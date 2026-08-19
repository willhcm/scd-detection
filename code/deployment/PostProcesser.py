from tqdm.auto import tqdm
from scipy.ndimage import label as ndi_label
from skimage.measure import regionprops
import numpy as np
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from veto_helpers import build_dem_context
import rasterio as rio
from rasterio.warp import reproject

# windowed data source to limit RAM usage during deployment with huge rasters
class WindowedDataSource:

    def __init__(self, path, type="DEM", nominal_res=None):
        self.path = path
        self.type = type

        # only read metadata here
        with rio.open(path) as src:
            self.crs = src.crs
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.count = src.count

            # set res
            if nominal_res is None:
                self.res = float(abs(src.res[0]))
            else:
                self.res = float(nominal_res)

    @classmethod
    def from_tiff(cls, path, type="DEM"):
        return cls(path=path, type=type)

    @classmethod
    def from_tiff_utm(cls, path, native_res, type="DEM"):
        # no longer reproject whole raster in init
        # native_res is retained as the nominal pixel resolution.
        return cls(path=path, type=type, nominal_res=native_res)

    def reproject_to_shape(
        self,
        target_crs,
        tile_bounds,
        out_width,
        out_height,
        resampling=Resampling.bilinear):

        dst_transform = from_bounds(*tile_bounds, out_width, out_height)

        with rio.open(self.path) as src:

            if self.type == "RGB":

                out = np.full((3, out_height, out_width), np.nan, dtype=np.float32)

                for band_index in range(3):

                    reproject(
                        source=rio.band(src, band_index + 1),
                        destination=out[band_index],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                        dst_nodata=np.nan)
                    
            else:
                out = np.full((out_height, out_width), np.nan, dtype=np.float32)

                reproject(
                    source=rio.band(src, 1),
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                    dst_nodata=np.nan)

        return out

class PostProcessor:

    def __init__(
        self,
        tile_size,
        context_tile_size,
        device,
        model,
        rgb_path,
        dem_path,
        batch_size=32,
        detect_threshold=0.40,
        veto_threshold=0.90,
        min_crop_pixels=64,
        object_fraction=0.5,
        context_scale=4.0,
        min_context_width_m=768.0,
        max_context_width_m=4000.0):

        self.tile_size = tile_size
        self.context_tile_size = context_tile_size

        # veto model setup
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        # data origins
        self.rgb_path = rgb_path
        self.dem_path = dem_path

        # data source objects
        self.rgb_source = WindowedDataSource.from_tiff_utm(rgb_path,native_res=3, type="RGB")
        self.dem_source = WindowedDataSource.from_tiff(dem_path,type="DEM")

        # hyperparameters
        self.batch_size = batch_size
        self.detect_threshold = detect_threshold
        self.veto_threshold = veto_threshold
        self.min_crop_pixels = min_crop_pixels
        self.object_fraction = object_fraction
        self.context_scale = context_scale
        self.min_context_width_m = min_context_width_m
        self.max_context_width_m = max_context_width_m

    # finds the mask of an object to overlay onto the rgb/dem context images
    # static method -> doesnt need self
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

    @staticmethod
    def _prepare_rgb(rgb, mask):
        """
        Return local RGB + candidate mask in [C, H, W] form.
        """

        # set up vars
        rgb = np.asarray(rgb,dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        # Convert bands-first RGB to channels-last.
        if (rgb.ndim == 3 and rgb.shape[0] == 3):
            rgb = np.moveaxis(rgb, 0, -1)

        if mask.ndim == 3:
            mask = np.squeeze(mask)

        # clean and order channels correctly (ChatGPT help with dims)
        mask = (mask > 0).astype(np.float32)
        rgb_mask = np.concatenate([rgb, mask[..., None],], axis=-1)
        rgb_mask = np.moveaxis(rgb_mask,-1, 0)

        return np.ascontiguousarray(rgb_mask, dtype=np.float32)

    def _make_tile(
        self,
        obj,
        transform,
        crs,
    ):
        row_c, col_c = obj.centroid
        
        cx, cy = transform * (col_c, row_c)

        # Candidate diameter from the deployment raster.
        prediction_pixel_res = abs(float(transform.a))

        diameter_m = (obj.equivalent_diameter_area * prediction_pixel_res)

        # match planet rgb res
        rgb_pixel_res = self.rgb_source.res

        # get tile crop width (rgb)
        min_crop_width_m = (self.min_crop_pixels * rgb_pixel_res)

        object_crop_width_m = (diameter_m / self.object_fraction)

        local_crop_width_m = max( min_crop_width_m, object_crop_width_m)

        local_half_width = local_crop_width_m / 2.0

        local_bounds = (
            cx - local_half_width,
            cy - local_half_width,
            cx + local_half_width,
            cy + local_half_width,
        )

        rgb = self.rgb_source.reproject_to_shape(
            target_crs=crs,
            tile_bounds=local_bounds,
            out_width=self.tile_size,
            out_height=self.tile_size,
            resampling=Resampling.bilinear,
        )

        # handle objects outside of rgb scene
        # these are just accepted later on!
        if rgb is None:
            return None

        rgb = np.asarray(rgb)

        if not np.isfinite(rgb).any():
            return None

        if np.all(rgb == 0):
            return None
        
        
        # get mask to overlay over rgb.
        local_mask = self._object_mask_for_bounds(
            obj=obj,
            source_transform=transform,
            bounds=local_bounds,
            output_size=self.tile_size)

        # wider DEM context around the same candidate centre.
        # clip provides a minimum and maximum scene size to avoid tiny/massive context extents
        context_width_m = np.clip( self.context_scale * local_crop_width_m, self.min_context_width_m, self.max_context_width_m)

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

        rgb_local = self._prepare_rgb(
            rgb,
            local_mask,
        )

        return {
            "rgb_local": rgb_local,
            "dem_context": np.ascontiguousarray(
                dem_context,
                dtype=np.float32,
            ),
            "scalar_features": np.ascontiguousarray(
                scalar_features,
                dtype=np.float32,
            ),
            "obj": obj,
            "local_bounds": local_bounds,
            "context_bounds": context_bounds,
            "local_crop_width_m": local_crop_width_m,
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

        batch_rgb = []
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
            rgb_tensor = torch.from_numpy(np.stack(batch_rgb, axis=0)).to(self.device,dtype=torch.float32)
            dem_tensor = torch.from_numpy(np.stack(batch_dem, axis=0)).to(self.device, dtype=torch.float32)
            scalar_tensor = torch.from_numpy(np.stack(batch_scalars, axis=0)).to(self.device, dtype=torch.float32)

            # predict
            with torch.inference_mode():
                logits = self.model(rgb_tensor, dem_tensor, scalar_tensor).view(-1)

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
            batch_rgb.clear()
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
            batch_rgb.append(tile["rgb_local"])
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