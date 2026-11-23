from tqdm.auto import tqdm
from scipy.ndimage import label as ndi_label
from skimage.measure import regionprops
import numpy as np
import torch
import deployment.config as config
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from helpers.VetoHelpers import build_veto_data, normalise_veto_data, VETO_BAND_INDEX
from loaders.Cogs import WindowedDataSource
import os
from pathlib import Path
import gc

REPO_ROOT = Path(__file__).resolve().parent.parent


class Object():
    """ Takes a regionprops object and extracts the relevant attributes for vetoing. Allows addition of other attributes such as depth_width_ratio """
    def __init__(self, object):
        self.object = object
        self.depth_width_ratio = None
        self.depth = None
        self.veto_embedding = None
        self.circularity = (4 * np.pi * object.area) / (object.perimeter ** 2) if object.perimeter > 0 else 0.0

    def __getattr__(self, name):
        if name == 'object':
            raise AttributeError(name)
        return getattr(self.object, name)

    def as_dict(self):
        return {
            "circ": self.circularity,
            "depth_m": np.nan if self.depth is None else float(self.depth),
            "dw_ratio": np.nan if self.depth_width_ratio is None else float(self.depth_width_ratio),
            "elongation": self.object.axis_major_length / self.object.axis_minor_length if self.object.minor_axis_length > 0 else 0.0,
            "eccentricity": self.object.eccentricity,
            'embedding': self.veto_embedding if self.veto_embedding is not None else np.nan,
        }


class PostProcessor:
    """
    A class to handle post-processing of predictions, including vetoing false positives.
    Parameters (device, DEM path, thresholds, context sizes...) come from config.DeployConfig.

    Parameters
    ----------
    model: torch.nn.Module
        Trained Veto classifier.
    norm_stats: dict
        Normalization statistics for the veto data.
    """

    def __init__(self, model, norm_stats=None, cfg=None):

        cfg = cfg or config.get()
        self.cfg = cfg

        # veto model setup
        self.device = cfg.device
        self.model = model.to(self.device)
        self.model.eval()
        self.norm_stats = norm_stats

        # data origins
        self.dem_path = cfg.dem_path
        self.dem_source = WindowedDataSource.from_tiff_utm(self.dem_path, type="DEM", resolution=cfg.native_res)

        # hyperparameters
        self.context_tile_size = cfg.veto_context_tile_size
        self.batch_size = cfg.veto_batch_size
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
        candidate_width_m = float(diameter_m) * 2.0

        context_half_width = context_width_m / 2.0
        candidate_half_width = candidate_width_m / 2.0

        context_bounds = (
            cx - context_half_width,
            cy - context_half_width,
            cx + context_half_width,
            cy + context_half_width,
        )
        candidate_bounds = (
            cx - candidate_half_width,
            cy - candidate_half_width,
            cx + candidate_half_width,
            cy + candidate_half_width,
        )

        dem = self.dem_source.reproject_to_shape(
            target_crs=crs,
            tile_bounds=context_bounds,
            out_width=self.context_tile_size,
            out_height=self.context_tile_size,
            resampling=Resampling.bilinear,
        )
        candidate_dem = self.dem_source.reproject_to_shape(
            target_crs=crs,
            tile_bounds=candidate_bounds,
            out_width=self.context_tile_size,
            out_height=self.context_tile_size,
            resampling=Resampling.bilinear,
        )

        # similarly handles invalid DEM fetches
        if dem is None or candidate_dem is None:
            return None

        dem = np.asarray(dem)
        candidate_dem = np.asarray(candidate_dem)

        if not np.isfinite(dem).any() or not np.isfinite(candidate_dem).any():
            return None

        # gets mask in zoomed out scale for dem context
        context_mask = self._object_mask_for_bounds(
            obj=obj,
            source_transform=transform,
            bounds=context_bounds,
            output_size=self.context_tile_size,
        )
        candidate_mask = self._object_mask_for_bounds(
            obj=obj,
            source_transform=transform,
            bounds=candidate_bounds,
            output_size=self.context_tile_size,
        )

        # build scalar features (e.g. candidate - annulus elevation delta)
        image, candidate_image, scalar_features = build_veto_data(
            context=dem,
            context_mask=context_mask,
            context_width_m=context_width_m,
            candidate_diameter_m=diameter_m,
            candidate=candidate_dem,
            candidate_mask=candidate_mask,
        )

        # edit object, prior to normalisation
        candidate = Object(obj)
        candidate.depth_width_ratio = scalar_features[3]
        candidate.depth = scalar_features[3] * diameter_m

        # handles empty scalar features in case some error in calculation 
        if image is None or candidate_image is None or scalar_features is None:
            return None

        to_normalize = {
            'dem': image[VETO_BAND_INDEX['DEM']],
            'slope': image[VETO_BAND_INDEX['SLOPE']],
            'scalar_features': scalar_features,
            'candidate_dem': candidate_image[0],
        }

        dem, slope, scalar_features, candidate_dem = normalise_veto_data(
            to_normalize,
            norm_stats=self.norm_stats)
        image = np.stack([dem, slope, context_mask.astype(np.float32)], axis=0).astype(np.float32)
        candidate_image = np.stack(
            [candidate_dem, candidate_image[1].astype(np.float32)],
            axis=0,
        )

        image = np.asarray(image, dtype=np.float32)

        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        return {
            "image": np.ascontiguousarray(
                image,
                dtype=np.float32,
            ),
            "candidate": np.ascontiguousarray(
                candidate_image,
                dtype=np.float32,
            ),
            "scalar_features": np.ascontiguousarray(
                scalar_features,
                dtype=np.float32,
            ),
            "obj": candidate,
            "context_bounds": context_bounds,
            "crop_width_m": crop_width_m,
            "context_width_m": context_width_m,
            "diameter_m": diameter_m,
        }

    @staticmethod
    def _apply_lut(lut, labeled, rows=2048):
        # lookup in strips so the int64 index copy never covers the whole raster
        out = np.empty(labeled.shape, dtype=lut.dtype)
        for i in range(0, labeled.shape[0], rows):
            out[i:i + rows] = lut[labeled[i:i + rows]]
        return out

    def predict(self, results):
        """
        Apply the veto classifier to the merged predictions.
        """

        labeled = results["predictions"]
        transform = results["transform"]
        crs = results["crs"]

        # gets region properties (i.e. area, centroid, bbox)
        # cache=False so coords aren't held for every object
        objects = regionprops(labeled, cache=False)

        rejected_ids = []
        embeddings = {}
        features = {}

        # init tracking ints and lists for each batch 
        accepted = 0
        rejected = 0
        skipped = 0
        predicted = 0

        batch_dem = []
        batch_candidate = []
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
            candidate_tensor = torch.from_numpy(
                np.stack(batch_candidate, axis=0)
            ).to(self.device, dtype=torch.float32)
            scalar_tensor = torch.from_numpy(np.stack(batch_scalars, axis=0)).to(self.device, dtype=torch.float32)

            # predict
            with torch.inference_mode():

                logits, embeds = self.model(
                    dem_tensor,
                    candidate_tensor,
                    scalar_tensor,
                    return_embeddings=True
                )

                logits = logits.view(-1)

                embeds = embeds.cpu().numpy()

                # sigmoid for probs
                probabilities = torch.sigmoid(logits).cpu().numpy()

            for i, (obj, probability) in enumerate(zip(batch_objs, probabilities) ):
                features[obj.label] = obj.as_dict()
                if probability >= self.veto_threshold:
                    accepted += 1
                    embeddings[obj.label] = embeds[i]
                else:
                    embeddings[obj.label] = embeds[i]
                    rejected_ids.append(obj.label)
                    rejected += 1
                predicted += 1

            # empty batch lists
            batch_dem.clear()
            batch_candidate.clear()
            batch_scalars.clear()
            batch_objs.clear()

        # make batches and send to model
        for obj in objects:
            tile = self._make_tile(obj=obj, transform=transform, crs=crs)

            # if cant build tile, just accept
            if tile is None:
                skipped += 1
                accepted += 1

                embeddings[obj.label] = np.nan
                features[obj.label] = Object(obj).as_dict()

                # update progress
                progress.update(1)
                progress.set_postfix(
                    predicted=predicted,
                    accepted=accepted,
                    rejected=rejected,
                    skipped=skipped)

                continue

            # if can build, add to batch lists
            batch_dem.append(tile["image"])
            batch_candidate.append(tile["candidate"])
            batch_scalars.append(tile["scalar_features"])
            batch_objs.append(tile['obj'])

            # when batch lists are full (i.e. at batch_size, send to model and predict)
            if len(batch_objs) >= self.batch_size:
                flush_batch()

                progress.set_postfix(
                    predicted=predicted,
                    accepted=accepted,
                    rejected=rejected,
                    skipped=skipped)

            progress.update(1)

        # Process the final partial batch.
        flush_batch()

        if self.dem_source.temp_path is not None and os.path.exists(self.dem_source.temp_path):
            os.remove(self.dem_source.temp_path)

        progress.set_postfix(
            predicted=predicted,
            accepted=accepted,
            rejected=rejected,
            skipped=skipped)

        progress.close()

        print(
            f"{accepted} objects retained, "
            f"{rejected} objects rejected, "
            f"{skipped} objects could not be assessed.")

        # done with the objects, free them before the big allocations
        del objects
        gc.collect()

        requested_returns = self.cfg.requested_returns
        rejected_ids = np.asarray(rejected_ids, dtype=np.int64)

        # look up table (lut) to map old labels to new labels, with rejected objects mapped to 0
        lut = np.arange(labeled.max() + 1, dtype=np.int32)
        lut[rejected_ids] = 0

        out = {"predictions": self._apply_lut(lut, labeled), "transform": transform, "crs": crs, "features": features, "embeddings": embeddings}

        if 'rejected' in requested_returns:
            lut = np.zeros_like(lut)
            lut[rejected_ids] = rejected_ids
            out['rejected'] = self._apply_lut(lut, labeled)

        return out
