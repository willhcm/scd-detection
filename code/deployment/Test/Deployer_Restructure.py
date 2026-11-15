import os
from pathlib import Path

import numpy as np
import torch
from rasterio.features import shapes
from rasterio.warp import transform as warp_transform
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

            if len(x_positions) == 0 or len(y_positions) == 0:
                return [], transform, dem_source.crs, (h, w)

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

    duplicate_iou_threshold = 0.2

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
    def _canonicalize(
        instances,
        source_transform,
        source_crs,
        target_transform,
        target_crs,
        target_shape,
    ):
        canonical = []
        for instance in instances:
            rows, cols = instance["coords"].T
            xs, ys = source_transform * (cols + 0.5, rows + 0.5)
            if source_crs != target_crs:
                xs, ys = warp_transform(source_crs, target_crs, xs, ys)
            target_cols, target_rows = ~target_transform * (xs, ys)
            target_rows = np.floor(target_rows).astype(np.intp)
            target_cols = np.floor(target_cols).astype(np.intp)
            inside = (
                (target_rows >= 0)
                & (target_rows < target_shape[0])
                & (target_cols >= 0)
                & (target_cols < target_shape[1])
            )
            if inside.any():
                canonical.append({
                    **instance,
                    "coords": np.column_stack(
                        (target_rows[inside], target_cols[inside])
                    ),
                })
        return canonical

    @staticmethod
    def _resolve_overlaps(
        instances,
        shape,
        iou_threshold=0.5,
        suppress_contained=True,
        containment_threshold=0.99,
    ):
        """Suppress duplicate masks, then assign shared pixels by score."""
        resolved = np.zeros(shape, dtype=np.int32)
        ordered = sorted(
            instances,
            key=lambda instance: instance["score"],
            reverse=True,
        )
        pixels_by_instance = [
            np.unique(np.ravel_multi_index(
                (instance["coords"][:, 0], instance["coords"][:, 1]),
                shape,
            ))
            for instance in ordered
        ]
        suppressed = np.zeros(len(ordered), dtype=bool)

        for index, instance_pixels in enumerate(pixels_by_instance):
            for other_index, other_pixels in enumerate(pixels_by_instance):
                if index == other_index or not other_pixels.size:
                    continue
                intersection = np.intersect1d(
                    instance_pixels,
                    other_pixels,
                    assume_unique=True,
                ).size
                if not intersection:
                    continue

                if (
                    suppress_contained
                    and instance_pixels.size < other_pixels.size
                    and intersection / instance_pixels.size
                    >= containment_threshold
                ):
                    suppressed[index] = True
                    break

                union = instance_pixels.size + other_pixels.size - intersection
                if (
                    union
                    and intersection / union >= iou_threshold
                    and ordered[index]["score"] < ordered[other_index]["score"]
                ):
                    suppressed[index] = True
                    break

        kept = []
        for index, instance in enumerate(ordered):
            if suppressed[index]:
                continue
            rows, cols = instance["coords"].T
            available = resolved[rows, cols] == 0
            if not available.any():
                continue
            resolved[rows[available], cols[available]] = len(kept) + 1
            kept.append({
                **instance,
                "coords": np.column_stack(
                    (rows[available], cols[available])
                ),
            })
        return resolved, kept

    @staticmethod
    def _stamp(canvas, instances):
        for object_id, instance in enumerate(instances, start=1):
            rows, cols = instance["coords"].T
            canvas[rows, cols] = object_id

    def sweep(self):
        resolution_results = []
        raw = {}
        canonical_transform = canonical_crs = canonical_shape = None

        for resolution in self.cfg.resolutions:
            predictor = InstanceResPredictor(
                resolution,
                self.model,
                self.norm_stats,
            )
            instances, transform, crs, raster_shape = predictor.predict()
            raw[resolution] = instances
            resolution_results.append((instances, transform, crs, raster_shape))

        if not resolution_results:
            raise RuntimeError("No resolution produced an inference grid.")

        _, canonical_transform, canonical_crs, canonical_shape = (
            resolution_results[0]
        )
        all_instances = []
        accepted = []
        rejected = []
        veto_probabilities = []

        for instances, transform, crs, _ in resolution_results:
            if self.cfg.to_veto:
                postprocessor = InstancePostProcessor(
                    self.veto_model,
                    self.veto_norm_stats,
                )
                accepted_instances, rejected_instances, probabilities = (
                    postprocessor.predict_instances(instances, transform, crs)
                )
            else:
                accepted_instances = instances
                rejected_instances = []
                probabilities = []

            accepted.extend(self._canonicalize(
                accepted_instances,
                transform,
                crs,
                canonical_transform,
                canonical_crs,
                canonical_shape,
            ))
            rejected.extend(self._canonicalize(
                rejected_instances,
                transform,
                crs,
                canonical_transform,
                canonical_crs,
                canonical_shape,
            ))
            veto_probabilities.extend(probabilities)

            all_instances.extend(self._canonicalize(
                instances,
                transform,
                crs,
                canonical_transform,
                canonical_crs,
                canonical_shape,
            ))

        unvetoed, _ = self._resolve_overlaps(
            all_instances,
            canonical_shape,
            iou_threshold=self.duplicate_iou_threshold,
        )
        predictions, accepted = self._resolve_overlaps(
            accepted,
            canonical_shape,
            iou_threshold=self.duplicate_iou_threshold,
        )

        available = {
            "raw": raw,
            "unvetoed": unvetoed,
            "predictions": predictions,
            "transform": canonical_transform,
            "crs": canonical_crs,
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
