import itertools
import numpy as np
import optuna
from pathlib import Path
import sys
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import box, shape

from deployment.Inference import SCDModel

def bbox_iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter)

class Tuner():

    def __init__(self,
                model=None,
                dem_path=None,
                device="cpu",
                resolutions=None,
                out_path=None,
                run_name=None,
                veto_model_state=None,
                hyperparams=None,
                to_veto=True,
                label_path=None,
                areas=None):

        if areas is None:
            if label_path is None or dem_path is None or model is None:
                raise ValueError(
                    "Single-area tuning needs model, dem_path and label_path."
                )
            areas = [{
                "name": "area",
                "model": model,
                "dem_path": dem_path,
                "device": device,
                "resolutions": resolutions,
                "out_path": out_path,
                "run_name": run_name,
                "veto_model_state": veto_model_state,
                "hyperparams": hyperparams,
                "to_veto": to_veto,
                "label_path": label_path,
            }]
        if not areas:
            raise ValueError("Tuner needs at least one test area.")

        self.out_path = out_path
        self._config_lock = Lock() # ensures that config is not modified while a deployer is running.
        self.area_models = []
        for area in areas:
            area_name = area.get("name", f"area-{len(self.area_models) + 1}")
            if area.get("label_path") is None:
                raise ValueError(f"Test area {area_name} needs a label_path.")

            labels = gpd.read_file(area["label_path"])
            with rasterio.open(area["dem_path"]) as dem:
                if labels.crs != dem.crs:
                    labels = labels.to_crs(dem.crs)

            self.area_models.append({
                "name": area_name,
                "model": SCDModel(
                    MaskRCNN_model_state=area["model"],
                    dem_path=area["dem_path"],
                    device=area["device"],
                    resolutions=area["resolutions"],
                    out_path=area["out_path"],
                    veto_model_state=area.get("veto_model_state"),
                    hyperparams=area.get("hyperparams"),
                    run_name=area.get("run_name"),
                    to_veto=area.get("to_veto", True),
                    requested_returns={"predictions", "transform", "crs"},
                ),
                "labels": labels,
            })

        self.model = self.area_models[0]["model"]
        self.labels = self.area_models[0]["labels"]

    def _run_area(self, area, params):
        # Deployment still has legacy module-global config consumers. Keep the
        # complete sweep under the lock so another area cannot replace the
        # active DEM/config while this area is running.
        with self._config_lock:
            area["model"].set_params(**params)
            deployer = area["model"].deployer
            results = deployer.sweep()

        labels = np.asarray(results["predictions"]).astype(np.int32)
        geometries, object_ids = [], []
        for geometry, object_id in shapes(
            labels,
            mask=labels > 0,
            transform=results["transform"],
            connectivity=8):

            geometries.append(shape(geometry))
            object_ids.append(int(object_id))

        # sieve out small objects that are below the minimum size threshold
        min_object_m2 = area["model"].config.min_object_m2
        if min_object_m2 > 0:
            with rasterio.open(area["model"].dem_path) as dem:
                pixel_area_m2 = abs(dem.transform.a * dem.transform.e)
            min_pixels = int(np.ceil(min_object_m2 / pixel_area_m2))
            filtered_geometries, filtered_object_ids = [], []
            for geom, obj_id in zip(geometries, object_ids):
                if geom.area >= min_pixels * pixel_area_m2:
                    filtered_geometries.append(geom)
                    filtered_object_ids.append(obj_id)
            geometries = filtered_geometries
            object_ids = filtered_object_ids

        return gpd.GeoDataFrame(
            {"object_id": object_ids, "geometry": geometries},
            crs=results["crs"],
        )

    def run(self, params):
        predictions = [self._run_area(a, params) for a in self.area_models]
        return predictions[0] if len(predictions) == 1 else predictions

    def _evaluate_area(self, preds, area):
        """Return a dict of object precision, recall and F1, followed by
        scene-wide pixel IoU, precision, recall, F1 and accuracy. Pixel metrics are
        computed on valid DEM pixels after rasterizing the prediction and
        label polygons onto the DEM grid.
        """

        matched, unmatched, FN = [], [], []
        labels = area["labels"]
        if preds.crs != labels.crs:
            if preds.crs is None or labels.crs is None:
                raise ValueError(
                    "Prediction and label CRS are required for object-level metrics."
                )
            labels = labels.to_crs(preds.crs)

        label_boxes = [box(*row.geometry.bounds) for _, row in labels.iterrows()]

        for idx, row in preds.iterrows():
            pred_box = box(*row.geometry.bounds)
            if any(bbox_iou(pred_box, lb) > 0.5 for lb in label_boxes):
                matched.append(idx)
            else:
                unmatched.append(idx)

        for idx, row in labels.iterrows():
            label_box = box(*row.geometry.bounds)
            if not any(bbox_iou(label_box, box(*preds.loc[i].geometry.bounds)) > 0.5 for i in matched):
                FN.append(idx)

        FNs = labels.loc[FN]

        # objet-level metrics

        precision = len(matched) / (len(matched) + len(unmatched)) if (len(matched) + len(unmatched)) > 0 else 0
        recall = len(matched) / (len(matched) + len(FNs)) if (len(matched) + len(FNs)) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # scene wide pixel-level metrics
        # Use the DEM grid as the common scene grid.  Rasterizing the complete
        # GeoDataFrames first makes overlapping object polygons count once.
        with rasterio.open(area["model"].dem_path) as dem:
            reference_crs = dem.crs
            valid_pixels = ~dem.read(1, masked=True).mask
            raster_shape = (dem.height, dem.width)
            transform = dem.transform

        def scene_mask(gdf):
            if gdf.empty:
                return np.zeros(raster_shape, dtype=bool)

            geometries = gdf
            if geometries.crs != reference_crs:
                if geometries.crs is None or reference_crs is None:
                    raise ValueError(
                        "Prediction and DEM CRS are required to calculate pixel metrics."
                    )
                geometries = geometries.to_crs(reference_crs)

            shapes = (
                (geometry, 1)
                for geometry in geometries.geometry
                if geometry is not None and not geometry.is_empty
            )
            return rasterize(
                shapes,
                out_shape=raster_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=False,
            ).astype(bool)

        prediction_mask = scene_mask(preds)
        label_mask = scene_mask(labels)

        true_positive = np.count_nonzero(prediction_mask & label_mask & valid_pixels)
        false_positive = np.count_nonzero(
            prediction_mask & ~label_mask & valid_pixels
        )
        false_negative = np.count_nonzero(
            ~prediction_mask & label_mask & valid_pixels
        )
        true_negative = np.count_nonzero(
            ~prediction_mask & ~label_mask & valid_pixels
        )

        pixel_precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0
        )
        pixel_recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0
        )
        pixel_f1 = (
            2 * true_positive
            / (2 * true_positive + false_positive + false_negative)
            if 2 * true_positive + false_positive + false_negative
            else 0
        )
        pixel_iou = (
            true_positive / (true_positive + false_positive + false_negative)
            if true_positive + false_positive + false_negative
            else 0
        )
        pixel_accuracy = (
            (true_positive + true_negative)
            / (true_positive + true_negative + false_positive + false_negative)
            if true_positive + true_negative + false_positive + false_negative
            else 0
        )

        out = {
            "object_precision": precision,
            "object_recall": recall,
            "object_f1": f1,
            "pixel_iou": pixel_iou,
            "pixel_precision": pixel_precision,
            "pixel_recall": pixel_recall,
            "pixel_f1": pixel_f1,
            "pixel_accuracy": pixel_accuracy,
        }

        return out

    def evaluate_results(self, preds):
        if not isinstance(preds, list):
            return self._evaluate_area(preds, self.area_models[0])

        if len(preds) != len(self.area_models):
            raise ValueError("Number of prediction sets must match test areas.")

        area_metrics = [
            self._evaluate_area(p, a) for p, a in zip(preds, self.area_models)
        ]

        n = len(area_metrics)
        return {
            "object_f1": sum(m["object_f1"] for m in area_metrics) / n,
            "pixel_iou": sum(m["pixel_iou"] for m in area_metrics) / n,
            "area_metrics": {
                a["name"]: m for a, m in zip(self.area_models, area_metrics)
            },
        }

    def tune(self, param_grid):
        best_object_f1 = -1
        best_params = None
        best_results = None

        keys = list(param_grid)

        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            print(f"Evaluating params: {params}")
            preds = self.run(params)
            metrics = self.evaluate_results(preds)
            object_f1 = metrics["object_f1"]

            print(f"Params: {params}, Object F1: {object_f1:.4f}")

            if object_f1 > best_object_f1:
                best_object_f1 = object_f1
                best_params = params
                best_results = metrics

        print(
            f"Best params: {best_params}, Best Object F1: "
            f"{best_object_f1:.4f}"
        )
        return best_params, best_results

    def tune_bayes(
        self,
        param_space,
        n_trials=50,
        seed=42,
        study_name=None,
        storage=None):

        """Tune parameter candidates or continuous ranges with Bayesian optimization.

        A sequence value is treated as a categorical candidate list for
        backwards compatibility. A mapping can define a continuous or integer
        range, for example ``{"type": "float", "low": 0.2, "high": 0.8}``.
        Float ranges may also specify ``step`` or ``log``.
        """
        if n_trials < 1:
            raise ValueError("n_trials must be at least 1.")
        if not param_space:
            raise ValueError("param_space must contain at least one parameter.")
        for name, space in param_space.items():
            if isinstance(space, Mapping):
                if space.get("type") not in {"float", "int"}:
                    raise ValueError(
                        f"{name} range type must be 'float' or 'int'."
                    )
                if "low" not in space or "high" not in space:
                    raise ValueError(f"{name} range needs 'low' and 'high'.")
                if space["low"] > space["high"]:
                    raise ValueError(f"{name} range has low greater than high.")
                if space.get("step") is not None and space.get("log", False):
                    raise ValueError(f"{name} cannot specify both step and log=True.")
            elif not space:
                raise ValueError(f"{name} must have at least one candidate value.")

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=study_name,
            storage=storage,
            load_if_exists=storage is not None and study_name is not None,
        )

        def objective(trial):
            params = {}
            for name, space in param_space.items():
                if not isinstance(space, Mapping):
                    params[name] = trial.suggest_categorical(name, list(space))
                    continue

                suggest = trial.suggest_float if space["type"] == "float" else trial.suggest_int
                kwargs = {
                    "low": space["low"],
                    "high": space["high"],
                }
                if space.get("step") is not None:
                    kwargs["step"] = space["step"]
                if space.get("log", False):
                    kwargs["log"] = True
                params[name] = suggest(name, **kwargs)

            print(f"Evaluating trial {trial.number}: {params}")
            preds = self.run(params)
            metrics = self.evaluate_results(preds)
            trial.set_user_attr("metrics", metrics)
            print(
                f"Trial {trial.number}: object F1 = "
                f"{metrics['object_f1']:.4f}")
            
            return metrics["object_f1"]

        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        best_results = best_trial.user_attrs.get("metrics")
        if best_results is None:
            preds = self.run(best_trial.params)
            best_results = self.evaluate_results(preds)

        print(f"Best params: {study.best_params}")
        print(f"Best Object F1: {study.best_value:.4f}")
        return study.best_params, best_results