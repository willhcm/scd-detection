import torch 
import numpy as np
from pathlib import Path
import sys
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

        self.model = model
        self.dem_path = dem_path
        self.device = device
        self.resolutions = resolutions
        self.out_path = out_path   
        self.run_name = run_name
        self.rgb_veto = False
        self.rgb_path = None
        self.veto_model_state = veto_model_state
        self.hyperparams = hyperparams
        self.to_veto = to_veto
        self.labels = gpd.read_file(label_path) if label_path is not None else None

        self.model = SCDModel(MaskRCNN_model_state=model['model_state_dict'],
                              rgb_veto=self.rgb_veto,
                              dem_path=self.dem_path,
                              device=self.device,
                              resolutions=self.resolutions,
                              out_path=self.out_path,
                              veto_model_state=self.veto_model_state,
                              rgb_path=self.rgb_path,
                              hyperparams=self.hyperparams,
                              run_name=self.run_name,
                              to_veto=self.to_veto)

    def run(self, params):

        self.model.deployer.score_threshold = params['score_thresh']
        self.model.deployer.mask_threshold = params['mask_thresh']
        self.model.deployer.detect_threshold = params['detect_thresh']
        self.model.deployer.veto_threshold = params['veto_thresh']

        results = self.model.predict()
        predictions = self.model.deployer.merge_predictions(results['predictions'], results['transform'], results['crs'])

        # predictions as gdf
        return predictions

    def evaluate_results(self, preds):
        """Return object metrics followed by scene-wide pixel metrics.

        The returned values are object precision, recall, and F1, followed by
        pixel IoU, precision, recall, F1, and accuracy. Pixel metrics are
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
        best_f1 = -1
        best_params = None
        best_results = None

        for score_thresh in param_grid['score_thresh']:
            for mask_thresh in param_grid['mask_thresh']:
                for veto_thresh in param_grid['veto_thresh']:
                    for detect_thresh in param_grid['detect_thresh']:
                        params = {
                            'score_thresh': score_thresh,
                            'mask_thresh': mask_thresh,
                            'veto_thresh': veto_thresh,
                            'detect_thresh': detect_thresh
                        }
                        print(f"Evaluating params: {params}")
                        preds = self.run(params)
                        metrics = self.evaluate_results(preds)
                        f1 = metrics[2]  # object-level F1

                        print(f"Params: {params}, Object F1: {f1:.4f}")

                        if f1 > best_f1:
                            best_f1 = f1
                            best_params = params
                            best_results = metrics

        print(f"Best params: {best_params}, Best Object F1: {best_f1:.4f}")
        return best_params, best_results