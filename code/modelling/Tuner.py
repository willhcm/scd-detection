import itertools
import numpy as np
import optuna
from pathlib import Path
import sys
from collections.abc import Mapping
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from deployment.Inference import SCDModel

def bbox_iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter)

class Tuner():

    def __init__(self,
                model,
                dem_path, 
                device,
                resolutions,
                out_path,
                run_name=None,
                veto_model_state=None,
                hyperparams=None,
                to_veto=True,
                label_path=None):

        if label_path is None:
            raise ValueError("Tuner needs a label_path to evaluate against.")

        self.out_path = out_path
        self.labels = gpd.read_file(label_path)

        # the full checkpoint goes in (model_state_dict + norm_stats), Deployer unpacks it.
        # hyperparams are the fixed (non-swept) DeployConfig fields.
        self.model = SCDModel(MaskRCNN_model_state=model,
                            dem_path=dem_path,
                            device=device,
                            resolutions=resolutions,
                            out_path=out_path,
                            veto_model_state=veto_model_state,
                            hyperparams=hyperparams,
                            run_name=run_name,
                            to_veto=to_veto)

    def run(self, params):

        # rebuilds the Deployer, so score/mask thresholds reach the model wrapper too
        self.model.set_params(**params)
        deployer = self.model.deployer

        results = deployer.sweep()
        trial_path = self.out_path / "trial.shp"

        # predictions as gdf
        return deployer.merge_predictions(results['predictions'], results['transform'], results['crs'], trial_path)

    def evaluate_results(self, preds):
        """Return a dict of object precision, recall and F1, followed by
        scene-wide pixel IoU, precision, recall, F1 and accuracy. Pixel metrics are
        computed on valid DEM pixels after rasterizing the prediction and
        label polygons onto the DEM grid.
        """

        matched, unmatched, FN = [], [], []
        label_boxes = [box(*row.geometry.bounds) for _, row in self.labels.iterrows()]

        for idx, row in preds.iterrows():
            pred_box = box(*row.geometry.bounds)
            if any(bbox_iou(pred_box, lb) > 0.5 for lb in label_boxes):
                matched.append(idx)
            else:
                unmatched.append(idx)

        for idx, row in self.labels.iterrows():
            label_box = box(*row.geometry.bounds)
            if not any(bbox_iou(label_box, box(*preds.loc[i].geometry.bounds)) > 0.5 for i in matched):
                FN.append(idx)

        FNs = self.labels.loc[FN]

        # objet-level metrics

        precision = len(matched) / (len(matched) + len(unmatched)) if (len(matched) + len(unmatched)) > 0 else 0
        recall = len(matched) / (len(matched) + len(FNs)) if (len(matched) + len(FNs)) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # scene wide pixel-level metrics
        # Use the DEM grid as the common scene grid.  Rasterizing the complete
        # GeoDataFrames first makes overlapping object polygons count once.
        with rasterio.open(self.model.dem_path) as dem:
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
        label_mask = scene_mask(self.labels)

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

    def tune(self, param_grid):
        best_iou = -1
        best_params = None
        best_results = None

        keys = list(param_grid)

        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            print(f"Evaluating params: {params}")
            preds = self.run(params)
            metrics = self.evaluate_results(preds)
            iou = metrics['pixel_iou']

            print(f"Params: {params}, Pixel IoU: {iou:.4f}")

            if iou > best_iou:
                best_iou = iou
                best_params = params
                best_results = metrics

        print(f"Best params: {best_params}, Best Pixel IoU: {best_iou:.4f}")
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
                f"Trial {trial.number}: pixel IoU = "
                f"{metrics['pixel_iou']:.4f}"
            )
            return metrics["pixel_iou"]

        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial
        best_results = best_trial.user_attrs.get("metrics")
        if best_results is None:
            preds = self.run(best_trial.params)
            best_results = self.evaluate_results(preds)

        print(f"Best params: {study.best_params}")
        print(f"Best Pixel IoU: {study.best_value:.4f}")
        return study.best_params, best_results