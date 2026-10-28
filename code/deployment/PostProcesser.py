from tqdm.auto import tqdm
from scipy.ndimage import label as ndi_label
from skimage.measure import regionprops
import numpy as np
import torch
import deployment.config as config
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from helpers.VetoHelpers import build_dem_context, normalise_dem_context
from loaders.WindowedDataSource import WindowedDataSource
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

class PostProcessor:
    """
    A class to handle post-processing of predictions, including vetoing false positives.
    Parameters (device, DEM path, thresholds, context sizes...) come from config.DeployConfig.

    Parameters
    ----------
    model: torch.nn.Module
        Trained Veto classifier.
    """

    def __init__(self, model):

        cfg = config.get()

        # veto model setup
        self.device = cfg.device
        self.model = model.to(self.device)
        self.model.eval()

        # data origins
        self.dem_path = cfg.dem_path
        self.dem_source = WindowedDataSource.from_tiff_utm(self.dem_path, type="DEM", resolution=cfg.native_res)

        # hyperparameters
        self.context_tile_size = cfg.veto_context_tile_size
        self.batch_size = cfg.veto_batch_size
        self.detect_threshold = cfg.detect_threshold
        self.veto_threshold = cfg.veto_threshold
        self.object_fraction = cfg.veto_object_fraction
        self.context_scale = cfg.context_scale
        self.min_context_width_m = cfg.min_context_width_m
        self.max_context_width_m = cfg.max_context_width_m

    @staticmethod
    def _object_mask_for_bounds(obj, source_transform, bounds, output_size):

        """
        Map the pixels belonging to an object into a new
        output grid covering `bounds`.
        """

        output_transform = from_bounds(*bounds, output_size, output_size)

        source_rows = (
            obj.coords[:, 0].astype(np.float64)
            + 0.5
        )

        source_cols = (
            obj.coords[:, 1].astype(np.float64)
            + 0.5
        )

        # raster indices to map coordinates
        xs, ys = source_transform * (source_cols,source_rows)

        # map coordinares to output raster indices
        output_cols, output_rows = (~output_transform) * (xs, ys)

        # need int indices
        output_rows = np.floor(output_rows).astype(np.int64)
        output_cols = np.floor(output_cols).astype(np.int64)

        # is object in output bounds?
        valid = (
            (output_rows >= 0)
            & (output_rows < output_size)
            & (output_cols >= 0)
            & (output_cols < output_size)
        )

        # build empty mask array
        mask = np.zeros((output_size, output_size),dtype=np.uint8)

        # where object is, stamp with 1
        mask[output_rows[valid], output_cols[valid]] = 1

        # Ensure a very small candidate does not disappear when
        # mapped into a large context tile.
        if not mask.any():
            centre = output_size // 2
            mask[centre, centre] = 1

        return mask

    def _make_tile(
        self,
        obj,
        transform,
        crs):

        row_c, col_c = obj.centroid
        
        cx, cy = transform * (col_c, row_c)

        # Candidate diameter from the deployment raster.
        prediction_pixel_res = abs(float(transform.a))

        # object size
        diameter_m = (obj.equivalent_diameter_area * prediction_pixel_res)
        crop_width_m = (diameter_m / self.object_fraction)
        
        # wider DEM context around the same candidate centre.
        # clip provides a minimum and maximum scene size to avoid tiny/massive context extents
        context_width_m = np.clip(self.context_scale * crop_width_m, self.min_context_width_m, self.max_context_width_m)

        context_half_width = context_width_m / 2.0

        context_bounds = (
            cx - context_half_width,
            cy - context_half_width,
            cx + context_half_width,
            cy + context_half_width,
        )

        dem = self.dem_source.reproject_to_shape(
            target_crs=crs,
            tile_bounds=context_bounds,
            out_width=self.context_tile_size,
            out_height=self.context_tile_size,
            resampling=Resampling.bilinear,
        )

        # similarly handles invalid DEM fetches
        if dem is None:
            return None

        dem = np.asarray(dem)

        if not np.isfinite(dem).any():
            return None

        # gets mask in zoomed out scale for dem context
        context_mask = self._object_mask_for_bounds(
            obj=obj,
            source_transform=transform,
            bounds=context_bounds,
            output_size=self.context_tile_size,
        )

        # build scalar features (e.g. candidate - annulus elevation delta)
        dem_context, scalar_features = build_dem_context(
            dem=dem,
            candidate_mask=context_mask,
            context_width_m=context_width_m,
            candidate_diameter_m=diameter_m,
        )

        # handles empty scalar features in case some error in calculation 
        if (dem_context is None or scalar_features is None):
            return None

        dem_context = np.asarray(
            dem_context,
            dtype=np.float32,
        )

        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        dem_context = normalise_dem_context(dem_context)

        return {
            "dem_context": np.ascontiguousarray(
                dem_context,
                dtype=np.float32,
            ),
            "scalar_features": np.ascontiguousarray(
                scalar_features,
                dtype=np.float32,
            ),
            "obj": obj,
            "context_bounds": context_bounds,
            "crop_width_m": crop_width_m,
            "context_width_m": context_width_m,
            "diameter_m": diameter_m,
        }

    @staticmethod
    def _stamp(canvas, obj, value):
        rows = obj.coords[:, 0]
        cols = obj.coords[:, 1]

        # stamps in place
        canvas[rows, cols] = value

    def predict(self, prob_raster, transform, crs,):

        # thresholds excluding NaN
        binary = (
            np.isfinite(prob_raster)
            & (
                prob_raster
                > self.detect_threshold
            )
        )

        # ndimage label
        labeled, n_objects = ndi_label(binary)

        del binary

        # gets region properties (i.e. area, centroid, bbox)
        objects = regionprops(labeled, intensity_image=prob_raster)

        # create empty arrays to stamp values onto
        stitched = np.zeros_like(prob_raster, dtype=np.float32)
        veto_map = np.zeros_like(prob_raster, dtype=np.int8) #int8, only positive 1 or 0.

        # Stores the veto probability for each assessed object.
        veto_probability_map = np.zeros_like(prob_raster, dtype=np.float16)

        # init tracking ints and lists for each batch 
        accepted = 0
        rejected = 0
        skipped = 0
        predicted = 0

        batch_dem = []
        batch_scalars = []
        batch_objs = []

        progress = tqdm(
            total=len(objects),
            desc="Vetoing predictions",
            unit="obj",
            dynamic_ncols=True)

        def flush_batch():
            nonlocal accepted
            nonlocal rejected
            nonlocal predicted

            # handles possible empty final batch
            if not batch_objs:
                return

            # make torch tensor from batch and send to device
            dem_tensor = torch.from_numpy(np.stack(batch_dem, axis=0)).to(self.device, dtype=torch.float32)
            scalar_tensor = torch.from_numpy(np.stack(batch_scalars, axis=0)).to(self.device, dtype=torch.float32)

            # predict
            with torch.inference_mode():

                logits = self.model(dem_tensor, scalar_tensor).view(-1)

                # sigmoid for probs
                probabilities = torch.sigmoid(logits).cpu().numpy()

            # stamp obj probability onto output arrays
            for obj, probability in zip(batch_objs, probabilities):
                probability = float(probability)

                self._stamp(veto_probability_map, obj, probability)

                if probability >= self.veto_threshold:
                    self._stamp(stitched, obj,obj.mean_intensity)
                    accepted += 1

                else:
                    self._stamp(veto_map, obj, -1.0)
                    rejected += 1

                predicted += 1

            # empty batch lists
            batch_dem.clear()
            batch_scalars.clear()
            batch_objs.clear()

        # make batches and send to model
        for obj in objects:
            tile = self._make_tile(obj=obj, transform=transform, crs=crs)

            # if cant build tile, just accept
            if tile is None:
                skipped += 1
                self._stamp(stitched, obj, obj.mean_intensity)
                accepted += 1

                # update progress
                progress.update(1)
                progress.set_postfix(
                    predicted=predicted,
                    accepted=accepted,
                    rejected=rejected,
                    skipped=skipped,
                )

                continue

            # if can build, add to batch lists
            batch_dem.append(tile["dem_context"])
            batch_scalars.append(tile["scalar_features"])
            batch_objs.append(obj)

            # when batch lists are full (i.e. at batch_size, send to model and predict)
            if len(batch_objs) >= self.batch_size:
                flush_batch()

                progress.set_postfix(
                    predicted=predicted,
                    accepted=accepted,
                    rejected=rejected,
                    skipped=skipped,
                )

            progress.update(1)

        # Process the final partial batch.
        flush_batch()

        if self.dem_source.temp_path is not None and os.path.exists(self.dem_source.temp_path):
            os.remove(self.dem_source.temp_path)

        progress.set_postfix(
            predicted=predicted,
            accepted=accepted,
            rejected=rejected,
            skipped=skipped,
        )

        progress.close()

        print(
            f"{accepted} objects retained, "
            f"{rejected} objects rejected, "
            f"{skipped} objects could not be assessed."
        )

        return (
            stitched,
            veto_map,
            veto_probability_map,
            transform,
            crs,
        )