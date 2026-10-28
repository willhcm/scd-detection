# inference at scale x:
# need to tile input DEM, with 4 channels at resolution x
# feed these into the model to generate predictions
# merge tiles back and generate prediction mask (vectorised for memory efficiency probably)

# ChatGPT assistance with these lines for importing whenever used

from pathlib import Path
import sys
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
import numpy as np

from rasterio.warp import reproject
import os
import numpy as np
import torch
import deployment.config as config
from loaders.Tiler import OVERLAP_SIGMA_MULTIPLIER
from rasterio.enums import Resampling
from scipy.ndimage import sobel, gaussian_filter, laplace
from deployment.ModelWrapper import ModelWrapper
from modelling.helpers import calculate_hillshade
from tqdm.auto import tqdm
from deployment.PostProcesser import PostProcessor
from loaders.WindowedDataSource import WindowedDataSource
import geopandas as gpd
from rasterio.features import shapes, sieve
from shapely.geometry import shape
from helpers.VetoHelpers import VETO_SCALAR_NAMES
from modelling.DEMVeto import DEM_Based_Vetoer
from scipy.ndimage import binary_dilation, label
from affine import Affine

REPO_ROOT = Path(__file__).resolve().parent.parent
TILE_ORDER = ['DEM', 'DEM_SLOPE', 'RR','LAPLACE']

class Deployer:

    """
    Deployer class for running inference on a DEM using a Mask R-CNN model and a Veto classifier.
    Handles loading the DEM, running predictions at multiple resolutions, merging the predictions,
    and applying the Veto classifier to filter out false positives.

    All run parameters (paths, resolutions, thresholds, tile sizes...) live in config.DeployConfig,
    so config.init(...) must be called before constructing this.

    Parameters
    ----------
    model_state_dict: dict
        Checkpoint of the trained Mask R-CNN model, containing 'model_state_dict' and 'norm_stats'.
    veto_model_dict: dict or None
        State dictionary of the trained Veto classifier model. Only needed if cfg.to_veto is True.
    """

    def __init__(self, model_state_dict, veto_model_dict):

        # snapshot, so changing the config mid-run can't alter this instance
        self.cfg = config.get()
        self.base_res = None

        self.norm_stats = model_state_dict['norm_stats']
        self.model = ModelWrapper(
            model_state_dict['model_state_dict'],
            self.cfg.score_threshold,
            self.cfg.mask_threshold)

        self.veto_model = None
        if self.cfg.to_veto:
            self.veto_model = DEM_Based_Vetoer()
            self.veto_model.load_state_dict(veto_model_dict)
            self.veto_model.to(self.cfg.device)
            self.veto_model.eval()

    def predict(self):
        """
        Predicts the probability maps for each resolution in cfg.resolutions using the Mask R-CNN model.
        """

        self.model.model.eval()
        self.model.model.to(self.cfg.device)
        preds = {}
        for res in self.cfg.resolutions:
            predictor = ResPredictor(res, self.model, self.norm_stats)

            prob_map, transform, crs = predictor.predict()

            preds[res] = {
                "prob": prob_map,
                "transform": transform,
                "crs": crs}

        return preds   


    def new_voting_scheme(self, preds):
        """
        Object-aware weighted voting.

        1. Generate a provisional merged map.
        2. Label candidate objects and estimate their sizes.
        3. Reproject predictions in bands and apply object-specific weights.

        Assumes larger resolution values represent coarser predictions.
        Object size is measured in base-grid pixels.
        """

        cfg = self.cfg
        band = cfg.merge_band

        base_res = min(preds)
        self.base_res = base_res

        base = preds[base_res]
        transform, crs = base["transform"], base["crs"]
        h, w = base["prob"].shape

        scratch_dir = REPO_ROOT / "scratch" / "merge"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: provisional predictions from the plain merge.
        provisional, _, _, _, _ = self.merge_predictions_pyramid(preds)

        labels_path = scratch_dir / "labels.npy"
        labels = np.lib.format.open_memmap(
            labels_path,
            mode="w+",
            dtype=np.int32,
            shape=(h, w))

        # Threshold the provisional map to identify candidate objects.
        # For very large rasters, connected-component labeling may need
        # to be replaced with a tiled implementation.
        mask = np.isfinite(provisional) & (
            provisional >= cfg.object_threshold)

        labels_array, n_objects = label(
            mask,
            structure=np.ones((3, 3), dtype=np.uint8))
        
        labels[:] = labels_array
        labels.flush()
        del labels_array, mask

        # Count pixels in each object.
        counts = np.bincount(
            np.asarray(labels).ravel(),
            minlength=n_objects + 1)

        areas = counts[1:] * (base_res ** 2)
        sizes = np.sqrt(areas) # eq diameter in metres

        # Resolution weights for each object size.
        resolutions = np.array(sorted(preds), dtype=np.float64)

        if np.any(resolutions <= 0):
            raise ValueError("Resolution values must be positive.")

        log_res = np.log(resolutions)

        # Preferred resolution increases with object size.
        # size_scale controls how quickly the preference shifts.

        # map object size to a target resolution, then compute weights based on distance from that target
        target_res = np.clip(
            (sizes * (1 / cfg.object_fraction)) / cfg.tile_size, 
            resolutions.min(),
            resolutions.max())

        # logits matrix. [i, j] is how far layer j is from the preferred resolution for object i, in log space.
        logits = -(
            log_res[None, :]
            - np.log(target_res[:, None])
        ) ** 2 / (2 * cfg.size_sigma**2)

        # subtract each row by its max, so that the best fit (max) is always exp(0) = 1, and the others are < 1.
        logits -= logits.max(axis=1, keepdims=True)

        # exp and normalise (divide weights by their row sum, so it sums to 1) to get weights for each object across resolutions.
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)

        # Pass 2: weighted merge, processed in bands.
        merged = np.lib.format.open_memmap(
            scratch_dir / "weighted_merged.npy",
            mode="w+",
            dtype=np.float32,
            shape=(h, w))

        for r0 in range(0, h, band):
            r1 = min(r0 + band, h)
            shape = (r1 - r0, w)
            band_transform = transform * Affine.translation(0, r0)

            band_labels = np.asarray(labels[r0:r1])

            numerator = np.zeros(shape, dtype=np.float32)
            denominator = np.zeros(shape, dtype=np.float32)
            coverage = np.zeros(shape, dtype=np.uint8)
            support = np.zeros(shape, dtype=np.uint8)
            best = np.full(shape, np.nan, dtype=np.float32)

            for j, (res, pred) in enumerate(
                sorted(preds.items())
            ):
                if res == base_res:
                    cur = np.asarray(
                        pred["prob"][r0:r1],
                        dtype=np.float32,
                    )
                else:
                    cur = np.full(shape, np.nan, dtype=np.float32)

                    reproject(
                        source=pred["prob"],
                        destination=cur,
                        src_transform=pred["transform"],
                        src_crs=pred["crs"],
                        src_nodata=np.nan,
                        dst_transform=band_transform,
                        dst_crs=crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                        init_dest_nodata=True,
                    )

                valid = np.isfinite(cur)

                coverage += valid
                support += valid & (cur >= cfg.support_threshold)
                best = np.fmax(best, cur)

                # Object-specific weight for this resolution.
                pixel_weights = weights[
                    np.clip(band_labels, 0, n_objects),
                    j,
                ].astype(np.float32, copy=False)

                effective_weights = pixel_weights * valid

                numerator += np.nan_to_num(cur, nan=0.0) * (
                    effective_weights
                )
                denominator += effective_weights

            result = np.full(shape, np.nan, dtype=np.float32)
            valid = denominator > 0

            result[valid] = (
                numerator[valid] / denominator[valid]
            )

            # Preserve your existing weak-support filtering.
            weak = (
                (coverage > 0)
                & (support < cfg.min_support)
                & (best < cfg.single_scale_keep)
            )
            result[weak] = 0.0

            merged[r0:r1] = result

        merged.flush()
        labels.flush()

        return merged, transform, crs
        

    def merge_predictions_pyramid(self, preds):

        cfg = self.cfg
        band = cfg.merge_band

        base_res = min(preds)
        self.base_res = base_res
        base = preds[base_res]
        transform, crs = base["transform"], base["crs"]
        h, w = base["prob"].shape

        scratch_dir = REPO_ROOT / 'scratch' / 'merge' 
        scratch_dir.mkdir(parents=True, exist_ok=True)
        merged = np.lib.format.open_memmap(
            scratch_dir / "merged.npy", mode="w+", dtype=np.float32, shape=(h, w))
        support = np.lib.format.open_memmap(
            scratch_dir / "support.npy", mode="w+", dtype=np.uint8, shape=(h, w))
        coverage = np.lib.format.open_memmap(
            scratch_dir / "coverage.npy", mode="w+", dtype=np.uint8, shape=(h, w))

        for r0 in range(0, h, band):
            r1 = min(r0 + band, h)
            shape = (r1 - r0, w)
            band_transform = transform * Affine.translation(0, r0)

            best = np.full(shape, np.nan, dtype=np.float32)
            sup = np.zeros(shape, dtype=np.uint8)
            cov = np.zeros(shape, dtype=np.uint8)

            for res, pred in preds.items():
                if res == base_res:
                    cur = np.array(pred["prob"][r0:r1], dtype=np.float32)
                else:
                    cur = np.full(shape, np.nan, dtype=np.float32)
                    reproject(
                        source=pred["prob"],
                        destination=cur,
                        src_transform=pred["transform"],
                        src_crs=pred["crs"],
                        src_nodata=np.nan,
                        dst_transform=band_transform,
                        dst_crs=crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                        init_dest_nodata=True)

                ok = np.isfinite(cur)
                cov += ok
                sup += ok & (cur >= cfg.support_threshold)
                best = np.fmax(best, cur)

            weak = (cov > 0) & (sup < cfg.min_support) & (best < cfg.single_scale_keep)
            best[weak] = 0.0
            best[cov == 0] = np.nan

            merged[r0:r1] = best
            support[r0:r1] = sup
            coverage[r0:r1] = cov

        for m in (merged, support, coverage):
            m.flush()

        return merged, transform, crs, support, coverage
    
    def veto(self, merged, transform, crs):
        """
        Applies the Veto classifier to the merged predictions to filter out false positives."""

        postproc = PostProcessor(self.veto_model)

        (stitched, rejected, veto_probability_map, out_transform, out_crs) = postproc.predict(merged,
                                                                                             transform,
                                                                                             crs)

        return (stitched, rejected, veto_probability_map, out_transform, out_crs)

    def sweep(self):

        # Docstring generation aided by Github Copilot Free.
        """
        Runs the full inference pipeline: predicts probability maps at multiple resolutions, merges them, and applies the Veto classifier if enabled.

        Returns
        -------
        dict with keys:
            raw: dict
                Predictions at each resolution.
            predictions: np.ndarray
                Merged probability map (Veto-cleaned if cfg.to_veto).
            transform: affine.Affine
            crs: rasterio.crs.CRS
            support: np.ndarray
                Support count for each pixel in the merged probability map.
            coverage: np.ndarray
                Coverage count for each pixel in the merged probability map.

        If cfg.to_veto is True, also:
            rejected: np.ndarray
                Rejected predictions by the Veto classifier.
            veto_probability_map: np.ndarray
                Probability map from the Veto classifier.
        """

        predictions = self.predict()

        merged, transform, crs= self.new_voting_scheme(predictions)

        if self.cfg.to_veto:
            accepted, rejected, veto_probability_map, out_transform, out_crs = self.veto(merged, transform, crs)

            out = {
                "raw": predictions,
                "predictions": accepted,
                "rejected": rejected,
                "veto_probability_map": veto_probability_map,
                "transform": out_transform,
                "crs": out_crs,
            }

        else:
            out = {
                "raw": predictions,
                "predictions": merged,
                "transform": transform,
                "crs": crs,
            }

        return out

    # merges all veto-cleaned predictions and then sieves remaining predictions
    # which are too physically small to be reasonably resolvable given the native DEM. 
    def merge_predictions(self, cleaned, transform, crs, output_path):
        """

        Merges the Veto-cleaned predictions into a single shapefile, sieving out predictions that are too small to be reasonably resolvable given the native DEM resolution.
        Exports the merged predictions as a shapefile to the specified output path.

        Parameters
        ----------

        cleaned: np.ndarray
            Veto-cleaned probability map.
        transform: affine.Affine
            Affine transform of the Veto-cleaned probability map.
        crs: rasterio.crs.CRS
            Coordinate reference system of the Veto-cleaned probability map.
        output_path: str    
            Path to save the merged shapefile.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame containing the merged predictions as polygons.
        """

        cfg = self.cfg

        # native res should determine smallest resolvable objects.
        min_m2 = (cfg.native_res ** 2) * cfg.min_native_pixels # resolution constrain, an area.
        min_pixels = int(np.ceil(max(min_m2, cfg.min_object_m2) / (self.base_res ** 2)))

        binary = (
            np.isfinite(cleaned)
            & (cleaned >= cfg.export_threshold)).astype(np.uint8)
        
        binary = sieve(
                binary,
                size=min_pixels,
                connectivity=8,
            )

        polygons = [shape(geometry) for geometry, value in shapes(binary,
                                                                mask=binary.astype(bool),
                                                                transform=transform,
                                                                connectivity=8) if value == 1]


        # to geodataframe for exporting
        gdf = gpd.GeoDataFrame(
            {
                "object_id": np.arange(1, len(polygons) + 1),
                "geometry": polygons,
            },
            crs=crs,
        )

        if not gdf.empty:
            gdf["area_m2"] = gdf.geometry.area

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # export
        gdf.to_file(
            output_path,
            driver="ESRI Shapefile",
            index=False,
        )

        print(f"Saved {len(gdf)} objects to {output_path}")

        return gdf


# self contained res predictor to decrease amount of calculations needed to be made repetitively!
class ResPredictor():

    def __init__(self, res, model, norm_stats):

        cfg = config.get()

        self.res = res
        self.model = model
        self.norm_stats = norm_stats

        self.device = cfg.device
        self.dem_path = cfg.dem_path
        self.tile_size = cfg.tile_size
        self.stride_frac = cfg.stride_frac
        self.batch_size = cfg.batch_size
        self.NODATA = cfg.nodata
        self.sigma_px = cfg.sigma_px

        self.overlap_px = max(
            4,
            int(round(
                OVERLAP_SIGMA_MULTIPLIER
                * self.sigma_px
            )),
        )

        self.overlap_m = self.overlap_px * res
        self.padded_size = (
            self.tile_size
            + 2 * self.overlap_px
        )
    # handles edges better than previous x and y position calculations. forces a tile at the edges.
    @staticmethod
    def _axis_positions(axis_min, axis_max, tile_width_m, stride_m, overlap_m, resolution):
        first = axis_min + overlap_m
        last = axis_max - overlap_m - tile_width_m

        if last < first:
            return np.empty(0, dtype=np.float64)

        positions = np.arange(
            first,
            last + 0.5 * resolution,
            stride_m,
            dtype=np.float64,
        )

        # Force a final tile against the valid far edge.
        if positions.size == 0:
            positions = np.array([first], dtype=np.float64)

        elif last - positions[-1] > 0.5 * resolution:
            positions = np.append(positions, last)

        return positions


    # cache derivaties and dem for entire x-extent of DEM region.
    # overlap is at 50%, so halves the amount of intensive calculations required.
    def _make_row_strip(self, dem_source, y0, x_positions, tile_w_m, resolution):

        res_value = np.log(resolution) / np.log(15.0)
        
        minx = x_positions[0] - self.overlap_m
        maxx = x_positions[-1] + tile_w_m + self.overlap_m
        strip_bounds = (minx, y0 - self.overlap_m, maxx, y0 + tile_w_m + self.overlap_m)

        strip_width_px = int(round((maxx - minx) / resolution))

        # target crs is source crs in this case! possibly redundant parameter, as source is reprojected within WindowedDataSource class.
        dem_padded = dem_source.reproject_to_shape(
            dem_source.crs, strip_bounds, strip_width_px, self.padded_size, resampling=Resampling.cubic
        ).astype(np.float32)

        invalid = np.isclose(dem_padded, self.NODATA) | ~np.isfinite(dem_padded)

        invalid = binary_dilation(
            invalid,
            iterations=int(np.ceil(3 * self.sigma_px))
        )

        sx = sobel(dem_padded, axis=0)
        sy = sobel(dem_padded, axis=1)
        slope = np.sqrt(sx ** 2 + sy ** 2)

        smoothed = gaussian_filter(dem_padded, sigma=self.sigma_px)
        rr = dem_padded - smoothed

        lap = laplace(dem_padded)

        # cut to size!
        s, e = self.overlap_px, self.overlap_px + self.tile_size

        dem = dem_padded[s:e, :]
        rr = rr[s:e, :]
        slope = slope[s:e, :]
        lap = lap[s:e, :]
        invalid = invalid[s:e, :]

        strip = {'DEM': dem, 'RR': rr, 'DEM_SLOPE': slope, 'LAPLACE': lap, 'RES': res_value, 'INVALID': invalid}
        return strip, minx

    # get next x tile from y-horizontal strip
    def _tile_from_strip(self, strip, strip_minx, x0, resolution):
        col0 = int(round((x0 - strip_minx) / resolution))
        col1 = col0 + self.tile_size

        invalid_tile = strip['INVALID'][:, col0:col1]

        if invalid_tile.any():
            return None

        return np.stack(
            [strip[name][:, col0:col1] for name in TILE_ORDER],
            axis=0)
    
    # changed for batching approach for optimisation - takes long time to predict 1000kms^2
    def predict(self):

        resolution = self.res
        res_value = np.log(resolution) / np.log(15.0)
        dem_source = WindowedDataSource.from_tiff_utm(self.dem_path, resolution=resolution)
        progress = None

        try:

            h, w = dem_source.height, dem_source.width
            transform = dem_source.transform

            # saved on file to reduce memory failure
            scratch = REPO_ROOT / 'scratch' / 'predict' / f"{resolution:g}m"
            scratch.mkdir(parents=True, exist_ok=True)
            tag = f"{resolution:g}m"
            prob_map = np.lib.format.open_memmap(scratch / f"prob_{tag}.npy", mode="w+", dtype=np.float32, shape=(h, w))
            weight_map = np.lib.format.open_memmap(scratch / f"weight_{tag}.npy", mode="w+", dtype=np.float32, shape=(h, w))

            # calculate streides and tile size in meetres
            stride_px = max(1, int(round(self.tile_size * self.stride_frac)))
            stride_m = stride_px * resolution
            tile_w_m = self.tile_size * resolution

            # calculate tile positions (using stride and res)
            minx, miny, maxx, maxy = dem_source.bounds

            print(f"bounds: {minx:g}, {miny:g}, {maxx:g}, {maxy:g}, tile size: {tile_w_m:g} m, stride: {stride_m:g} m, overlap: {self.overlap_m:g} m")
            x_positions = self._axis_positions(
                            minx,
                            maxx,
                            tile_w_m,
                            stride_m,
                            self.overlap_m,
                            resolution,
                        )

            y_positions = self._axis_positions(
                miny,
                maxy,
                tile_w_m,
                stride_m,
                self.overlap_m,
                resolution,
            )

            # moving valid checks from make_tile to save compute on skips.

            valid_x = (x_positions - self.overlap_m >= minx) & (x_positions + tile_w_m + self.overlap_m <= maxx)
            valid_y = (y_positions - self.overlap_m >= miny) & (y_positions + tile_w_m + self.overlap_m <= maxy)
            x_positions = x_positions[valid_x]
            y_positions = y_positions[valid_y]

            # progress bar
            total_tiles = len(x_positions) * len(y_positions)
            skipped_tiles = 0
            predicted_tiles = 0

            progress = tqdm(total=total_tiles, desc=f"Inference at {resolution:g} m", unit="tile", dynamic_ncols=True)
            weight_kernel = self._tile_weight_kernel()

            # temporary lists to hold input tiles before they are put onto device for prediction
            batch_tiles = []
            
            # corresponding list which stores the pixel-based locations for each tile.
            batch_locs = [] 

            # Claude assistance with this function. optimising and batching by-hand was something i hadnt done before!
            def flush_batch():
                # variable declared in nested function (has access to all internal variables.)
                nonlocal predicted_tiles
                if not batch_tiles:
                    return

                # create tensor from batch_tiles list.
                stacked = np.stack(batch_tiles, axis=0)
                x = torch.from_numpy(stacked).float()
                resolutions = torch.full((len(batch_tiles), 1), res_value, dtype=torch.float32)

                # outputs (prob map)
                masks = self.model.predict(x, resolutions, self.device) 

                # saves to correct locations in global prob_map and weight_map
                for mask, (row0, col0) in zip(masks, batch_locs):
                    pred = mask.numpy() if torch.is_tensor(mask) else np.asarray(mask)
                    row1, col1 = row0 + self.tile_size, col0 + self.tile_size
                    prob_map[row0:row1, col0:col1] += pred * weight_kernel
                    weight_map[row0:row1, col0:col1] += weight_kernel
                    predicted_tiles += 1

                # clears batch
                batch_tiles.clear()
                batch_locs.clear()

            with torch.inference_mode():

                for y0 in y_positions:
                    strip, strip_minx = self._make_row_strip(dem_source, y0, x_positions, tile_w_m, resolution)

                    for x0 in x_positions:

                        tile = self._tile_from_strip(strip, strip_minx, x0, resolution)

                        col0, row0 = ~transform * (x0, y0 + tile_w_m)
                        col0, row0 = int(round(col0)), int(round(row0))
                        row1, col1 = row0 + self.tile_size, col0 + self.tile_size

                        in_bounds = row0 >= 0 and col0 >= 0 and row1 <= h and col1 <= w

                        if tile is None or not in_bounds:
                            skipped_tiles += 1
                        else:
                            tile = self._prepare_tile(tile)
                            batch_tiles.append(tile)
                            batch_locs.append((row0, col0))

                            if len(batch_tiles) >= self.batch_size:
                                flush_batch()
                                progress.set_postfix(predicted=predicted_tiles, skipped=skipped_tiles)

                        progress.update(1)

                flush_batch()

            
            # normalise in row chunks so the memmaps never need to be fully in memory.
            # chunk step must equal chunk height or rows get divided more than once.
            chunk = 1024
            for r in range(0, h, chunk):
                p, wt = prob_map[r:r + chunk], weight_map[r:r + chunk]
                ok = wt > 0
                p[ok] /= wt[ok]
                p[~ok] = np.nan
            prob_map.flush()
            del weight_map

            print(f"Completed {resolution:g} m inference: {predicted_tiles}/{total_tiles} tiles predicted, {skipped_tiles} skipped.")

        finally:
            if progress is not None:
                progress.close()
            os.remove(dem_source.temp_path)
            
        return prob_map, transform, dem_source.crs

    @staticmethod
    def handle_outliers(band):
        p995 = np.nanpercentile(band, 99.5)
        p005 = np.nanpercentile(band, 0.5)
        band = np.clip(band, p005, p995) 

        return band
        
    def _normalise_band(self, band, name):
        band = band.astype(np.float32)

        band = self.handle_outliers(band)

        if name == "HILLSHADE":
            flat = np.cos(np.radians(45))
            band = flat - band
            band = np.power(band, 0.5)

        if name == 'DEM_SLOPE':
            band = np.power(band, 0.25)
            name = 'SLOPE'

        if name == 'DEM':
            band = band - np.median(band)

        band = band - self.norm_stats[name]['median']
        band = band / self.norm_stats[name]['iqr']

        return band
    
    def _prepare_tile(self, tile):

        for i, name in enumerate(TILE_ORDER):   
            tile[i] = self._normalise_band(tile[i], name)

        return tile
    
    # weight centre of tiles (reduces edge artefacts)
    def _tile_weight_kernel(self):
        # cached 
        if getattr(self, "_weight_kernel", None) is not None:
            return self._weight_kernel

        w1d = np.hanning(self.tile_size)
        # hanning hits 0 at the edges, which zeros out real predictions there
        # floored so edge pixels are downweighted but not thrown away
        w1d = np.clip(w1d, 0.05, None)
        kernel = np.outer(w1d, w1d).astype(np.float32)

        self._weight_kernel = kernel
        return kernel