import os
from pathlib import Path

import numpy as np
import torch
from rasterio.features import shapes
from shapely.geometry import shape

import deployment.config as config
from deployment.Deployer import ResPredictor
from deployment.Test.InstancePostProcessor import InstancePostProcessor
from deployment.Test.ModelWrapper_Restructure import InstanceModelWrapper
import geopandas as gpd

class InstanceResPredictor(ResPredictor):
    """Reuse the tile generator while retaining each model instance."""

    def predict(self):
        resolution = self.res
        res_value = np.log1p(resolution) / np.log1p(
            self.norm_stats["max_res"]
        )
        dem_source = self._make_dem_source(resolution)
        instances = []

        try:
            h, w = dem_source.height, dem_source.width
            transform = dem_source.transform
            stride_px = max(1, int(round(self.tile_size * self.stride_frac)))
            stride_m = stride_px * resolution
            tile_w_m = self.tile_size * resolution
            minx, miny, maxx, maxy = dem_source.bounds
            x_positions = self._axis_positions(
                minx, maxx, tile_w_m, stride_m, self.overlap_m, resolution
            )
            y_positions = self._axis_positions(
                miny, maxy, tile_w_m, stride_m, self.overlap_m, resolution
            )
            valid_x = (
                (x_positions - self.overlap_m >= minx)
                & (x_positions + tile_w_m + self.overlap_m <= maxx)
            )
            valid_y = (
                (y_positions - self.overlap_m >= miny)
                & (y_positions + tile_w_m + self.overlap_m <= maxy)
            )
            x_positions = x_positions[valid_x]
            y_positions = y_positions[valid_y]

            batch_tiles, batch_locs = [], []

            def flush_batch():
                if not batch_tiles:
                    return
                stacked = np.stack(batch_tiles, axis=0)
                inputs = torch.from_numpy(stacked).float()
                resolutions = torch.full(
                    (len(batch_tiles), 1),
                    res_value,
                    dtype=torch.float32,
                )
                predictions = self.model.predict_instances(
                    inputs, resolutions, self.device
                )
                for image_instances, (row0, col0) in zip(
                    predictions, batch_locs
                ):
                    for prediction in image_instances:
                        rows, cols = np.nonzero(prediction["mask"])
                        if len(rows) == 0:
                            continue
                        instances.append({
                            "coords": np.column_stack(
                                (rows + row0, cols + col0)
                            ),
                            "score": prediction["score"],
                            "resolution": resolution,
                        })
                batch_tiles.clear()
                batch_locs.clear()

            with torch.inference_mode():
                for y0 in y_positions:
                    strip, strip_minx = self._make_row_strip(
                        dem_source, y0, x_positions, tile_w_m, resolution
                    )
                    for x0 in x_positions:
                        tile = self._tile_from_strip(
                            strip, strip_minx, x0, resolution
                        )
                        col0, row0 = ~transform * (x0, y0 + tile_w_m)
                        col0, row0 = int(round(col0)), int(round(row0))
                        row1, col1 = row0 + self.tile_size, col0 + self.tile_size
                        if (
                            tile is None
                            or row0 < 0
                            or col0 < 0
                            or row1 > h
                            or col1 > w
                        ):
                            continue
                        batch_tiles.append(self._prepare_tile(tile))
                        batch_locs.append((row0, col0))
                        if len(batch_tiles) >= self.batch_size:
                            flush_batch()
                flush_batch()
        finally:
            if dem_source.temp_path is not None and os.path.exists(
                dem_source.temp_path
            ):
                os.remove(dem_source.temp_path)

        return instances, transform, dem_source.crs, (h, w)

    def _make_dem_source(self, resolution):
        from loaders.Cogs import WindowedDataSource

        return WindowedDataSource.from_tiff_utm(
            self.dem_path,
            resolution=resolution,
        )


class DeployerRestructure:
    """Instance-level alternative to the weighted-merge Deployer."""

    def __init__(self, model_state_dict, veto_model_dict):
        self.cfg = config.get()
        self.norm_stats = model_state_dict["norm_stats"]
        self.model = InstanceModelWrapper(
            model_state_dict["model_state_dict"],
            self.cfg.score_threshold,
            self.cfg.mask_threshold,
        )
        self.veto_model = None
        self.veto_norm_stats = None
        if self.cfg.to_veto:
            if veto_model_dict is None:
                raise ValueError("to_veto is True but no veto model was given.")
            from modelling.Veto import Vetoer

            self.veto_model = Vetoer()
            self.veto_model.load_state_dict(veto_model_dict["state"])
            self.veto_norm_stats = veto_model_dict["norm_stats"]
            self.veto_model.to(self.cfg.device)
            self.veto_model.eval()

    @staticmethod
    def _stamp(canvas, instances):
        for object_id, instance in enumerate(instances, start=1):
            rows, cols = instance["coords"].T
            canvas[rows, cols] = object_id

    def sweep(self):
        all_instances = []
        raw = {}
        transform = crs = shape = None

        for resolution in self.cfg.resolutions:
            predictor = InstanceResPredictor(
                resolution,
                self.model,
                self.norm_stats,
            )
            instances, transform, crs, shape = predictor.predict()
            raw[resolution] = instances
            all_instances.extend(instances)

        if transform is None or crs is None or shape is None:
            raise RuntimeError("No resolution produced an inference grid.")

        unvetoed = np.zeros(shape, dtype=np.int32)
        self._stamp(unvetoed, all_instances)
        accepted = all_instances
        rejected = []
        veto_probabilities = []

        if self.cfg.to_veto:
            postprocessor = InstancePostProcessor(
                self.veto_model,
                self.veto_norm_stats,
            )
            accepted, rejected, veto_probabilities = (
                postprocessor.predict_instances(
                    all_instances,
                    transform,
                    crs,
                )
            )

        predictions = np.zeros(shape, dtype=np.int32)
        self._stamp(predictions, accepted)
        available = {
            "raw": raw,
            "unvetoed": unvetoed,
            "predictions": predictions,
            "transform": transform,
            "crs": crs,
            "rejected": rejected,
            "veto_probability_map": veto_probabilities,
        }
        requested = self.cfg.requested_returns
        return {
            key: value
            for key, value in available.items()
            if requested is None or key in requested
        }

    def save_predictions(self, results, output_path):
        """Polygonize only after instance-level score and veto decisions."""
        predictions = np.asarray(results["predictions"], dtype=np.int32)
        transform = results["transform"]
        crs = results["crs"]
        geometries, object_ids = [], []
        for geometry, object_id in shapes(
            predictions,
            mask=predictions > 0,
            transform=transform,
            connectivity=8,
        ):
            geometries.append(shape(geometry))
            object_ids.append(int(object_id))


        gdf = gpd.GeoDataFrame(
            {"object_id": object_ids, "geometry": geometries},
            crs=crs,
        )
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        gdf.to_file(output_path / "predictions.shp", driver="ESRI Shapefile")
        return gdf


Deployer = DeployerRestructure
