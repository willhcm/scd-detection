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
from loaders.Cogs import OVERLAP_SIGMA_MULTIPLIER, WindowedDataSource
from rasterio.enums import Resampling
from scipy.ndimage import sobel, gaussian_filter, laplace
from deployment.ModelWrapper import ModelWrapper
from modelling.helpers import calculate_hillshade
from tqdm.auto import tqdm
from deployment.PostProcesser import PostProcessor
import geopandas as gpd
from rasterio.features import shapes, sieve
from shapely.geometry import shape
from helpers.VetoHelpers import VETO_SCALAR_NAMES
from modelling.Veto import Vetoer
from scipy.ndimage import binary_dilation, label
from affine import Affine
import rasterio
from scipy import ndimage
import shutil
import uuid

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

        self.norm_stats = model_state_dict['norm_stats'] # Normilisation stats for the Mask R-CNN model, used for normalizing input DEM tiles before prediction. 
        # normalisation stats are calculated in the training script from the training set (see modelling/Trainer.py and loaders/DatasetTools.py).

        self.model = ModelWrapper(
            model_state_dict['model_state_dict'],
            self.cfg.score_threshold,
            self.cfg.mask_threshold)

        self.veto_model = None
        if self.cfg.to_veto:
            self.veto_model = Vetoer()
            self.veto_model.load_state_dict(veto_model_dict['state'])
            self.veto_norm_stats = veto_model_dict['norm_stats'] 
            self.veto_model.to(self.cfg.device)
            self.veto_model.eval()
            self.postprocessor = PostProcessor(self.veto_model, self.veto_norm_stats, cfg=self.cfg)

        self.run_id = uuid.uuid4().hex
        self.scratch_root = REPO_ROOT / "scratch"
        self.scratch_dir = REPO_ROOT / "scratch" / "deploy" / self.run_id
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def predict(self):
        """
        Predicts the probability maps for each resolution in cfg.resolutions using the Mask R-CNN model. Calls ResPredictor for each resolution and returns a dictionary of predictions.
        """

        self.model.model.eval()
        self.model.model.to(self.cfg.device)
        preds = {}
        for res in self.cfg.resolutions:
            predictor = ResPredictor(
                res,
                self.model,
                self.norm_stats,
                cfg=self.cfg,
                run_id=self.run_id)

            prob_map, transform, crs, memmap_path = predictor.predict()

            preds[res] = {
                "prob": prob_map,
                "transform": transform,
                "crs": crs,
                "memmap_path": memmap_path}


        return preds   

    def object_scores(self, preds, labels, n_objects, base_res, transform, crs):
        """
        Max probability of each labelled object at each prediction resolution.

        preds: {res: {"prob", "transform", "crs"}}, resolutions sorted ascending
        labels: (h, w) int array of object ids, 0 = background (memmap is fine)
        n_objects: highest object id in labels
        base_res: resolution of the label grid
        transform, crs: georeferencing of the label grid

        Returns a float32 array of shape (n_objects + 1, n_res). Row i is
        object i, column j is the j-th resolution in sorted order. Row 0 is
        background and stays zero. NaN and out-of-extent pixels count as 0.

        Works in row bands of cfg.merge_band to keep RAM down.
        """

        # max prob per object per resolution -> (n_objects + 1, n_res), row 0 is background
        h, w = labels.shape
        band = self.cfg.merge_band
        items = sorted(preds.items())
        scores = np.zeros((n_objects + 1, len(items)), dtype=np.float32)

        # inverse transforms for same-crs preds, lets us skip reproject
        inv_tf = {res: ~p["transform"] for res, p in items if p["crs"] == crs}

        for r0 in range(0, h, band):
            r1 = min(r0 + band, h)
            band_labels = np.asarray(labels[r0:r1])

            # only care about pixels that are actually in an object
            rows, cols = np.nonzero(band_labels)
            if rows.size == 0:
                continue

            # sort by object id once, reused for every resolution
            pixel_ids = band_labels[rows, cols]
            sort_idx = np.argsort(pixel_ids)
            ids_sorted = pixel_ids[sort_idx]
            run_starts = np.flatnonzero(np.r_[True, ids_sorted[1:] != ids_sorted[:-1]])
            object_ids = ids_sorted[run_starts]

            # pixel centres in map coords (band is offset by r0 rows)
            map_x, map_y = transform * (cols + 0.5, rows + r0 + 0.5)
            band_tf = transform * Affine.translation(0, r0)

            for j, (res, pred) in enumerate(items):
                prob = pred["prob"]

                if res == base_res:
                    vals = np.asarray(prob[r0:r1])[rows, cols]

                elif res in inv_tf:
                    # nearest lookup by hand, same as reproject but much cheaper
                    src_col, src_row = inv_tf[res] * (map_x, map_y)
                    src_col = np.floor(src_col).astype(np.intp)
                    src_row = np.floor(src_row).astype(np.intp)
                    inside = ((src_col >= 0) & (src_col < prob.shape[1]) &
                              (src_row >= 0) & (src_row < prob.shape[0]))
                    vals = np.zeros(rows.size, dtype=np.float32)  # outside the source -> 0
                    vals[inside] = prob[src_row[inside], src_col[inside]]

                else:
                    # different crs, no way round the real reproject
                    warped = np.full(band_labels.shape, np.nan, dtype=np.float32)
                    reproject(
                        source=prob, destination=warped,
                        src_transform=pred["transform"], src_crs=pred["crs"],
                        src_nodata=np.nan, dst_transform=band_tf, dst_crs=crs,
                        dst_nodata=np.nan, resampling=Resampling.nearest,
                        init_dest_nodata=True)
                    vals = warped[rows, cols]

                # nan counts as 0, then max within each object's run
                vals_sorted = np.nan_to_num(vals[sort_idx], nan=0.0)
                run_max = np.maximum.reduceat(vals_sorted, run_starts)
                # max with existing so objects crossing band edges keep their best
                scores[object_ids, j] = np.maximum(scores[object_ids, j], run_max)

        return scores
    
    def weighted_vote(self, preds):
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

        # lowest resolution grid in predictions is used as the common grid shape for merged map and object labels. This is the "base" resolution.
        base_res = min(preds)
        self.base_res = base_res

        base = preds[base_res]
        transform, crs = base["transform"], base["crs"]
        h, w = base["prob"].shape


        # get maximum probabilty from all resolutions for each pixel (reprojecting coarser resolutions to common 'base' grid). This is the provisional merged map.
        # this merged map is then thresholded to identify candidate objects, which are then labelled and their sizes estimated.
        # the sizes are used for object-size-aware weighting of the predictions from each resolution, which are then merged to produce the final merged map.
        # an object size is measured in base-grid pixels, and the weighting is based on the resolution interval that an object of that size was trained to be seen at (see cfg.object_fraction, cfg.size_sigma).
        merged = self.max_prob(preds, base_res, transform, crs, (h, w), self.scratch_dir / "merged.npy")
        labels_path = self.scratch_dir / "labels.npy"

        # Threshold the provisional map to identify candidate objects.
        # For very large rasters, connected-component labeling may need
        # to be replaced with a tiled implementation.
        labels = np.lib.format.open_memmap(
            labels_path, mode="w+", dtype=np.int32, shape=(h, w))

        mask = np.isfinite(merged) & (merged >= cfg.object_threshold)
        n_objects = label(mask, structure=np.ones((3, 3), dtype=np.uint8), output=labels)
        labels.flush()
        del mask

        # banded for RAM.
        counts = np.zeros(n_objects + 1, dtype=np.int64)
        for r0 in range(0, h, band):
            counts += np.bincount(labels[r0:r0 + band].ravel(), minlength=n_objects + 1)

        areas = counts[1:] * (base_res ** 2)
        sizes = np.sqrt(areas) # eq diameter in metres

        resolutions = np.array(sorted(preds), dtype=np.float64) # preds is a dict of {res: {prob, transform, crs}}

        if np.any(resolutions <= 0):
            raise ValueError("Resolution values must be positive.")
        log_res = np.log(resolutions)

        # interval of resolutions an object of this size was trained to be seen at ([0.1, 0.35] object fractions)
        lo = np.log(sizes / (0.35 * cfg.tile_size))[:, None]
        hi = np.log(sizes / (0.10 * cfg.tile_size))[:, None]

        off = np.maximum(lo - log_res[None, :], 0) + np.maximum(log_res[None, :] - hi, 0) # offset of each resolution from the interval for each object
        weights = np.exp(-off ** 2 / (2 * cfg.size_sigma ** 2))   # 1 inside the interval 

        # matrix for max object prob per object per resolution
        stats = self.object_scores(preds, labels, n_objects, base_res, transform, crs)

        # no longer need dictionary of predictions from each resolution (was of form {res: {prob, transform, crs, memmap_path}}), so delete to free memory and storage
        if "raw" not in cfg.requested_returns:
            paths = []
            for res, items in preds.items():
                paths.append(items["memmap_path"])
                items['probs'] = None
            del base
            for f in paths:
                Path(f).unlink(missing_ok=True)

        # compute the weighted max probability for each object across resolutions, then threshold to keep or reject
        obj_score = (stats[1:] * weights).max(axis=1)
        keep = np.zeros(n_objects + 1, dtype=bool)
        keep[1:] = obj_score >= cfg.keep_threshold

        # write the final merged map and labels to disk, sieving out rejected objects
        for r0 in range(0, h, band):
            r1 = min(r0 + band, h)
            prov = merged[r0:r1]          # view into the file, edits go to disk
            lab = np.asarray(labels[r0:r1])
            kept = keep[lab]
            prov[~kept & ~np.isnan(prov)] = 0
            labels[r0:r1] = np.where(kept, lab, 0)

        merged.flush()
        labels.flush()
        return merged, labels, transform, crs, obj_score
        

    def max_prob(self, preds, base_res, transform, crs, shape, path):
        """
        Merges predictions from multiple resolutions into a single probability map by taking the maximum probability for each pixel across all resolutions.
        Pixel space is defined by the lowest resolution in preds (base_res). Coarser resolutions are reprojected to the base grid using nearest-neighbor resampling.
        """
        h, w = shape
        band = self.cfg.merge_band
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float16, shape=(h, w))

        for r0 in range(0, h, band):
            r1 = min(r0 + band, h)
            band_shape = (r1 - r0, w)
            band_tf = transform * Affine.translation(0, r0)
            best = np.full(band_shape, np.nan, dtype=np.float16)

            for res, pred in preds.items():
                if res == base_res:
                    cur = np.asarray(pred["prob"][r0:r1], dtype=np.float16)
                else:
                    tmp = np.full(band_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=pred["prob"], destination=tmp,
                        src_transform=pred["transform"], src_crs=pred["crs"],
                        src_nodata=np.nan, dst_transform=band_tf, dst_crs=crs,
                        dst_nodata=np.nan, resampling=Resampling.nearest,
                        init_dest_nodata=True)
                    cur = tmp.astype(np.float16)
                best = np.fmax(best, cur)

            out[r0:r1] = best

        out.flush()
        return out
    
    def veto(self, merged, labels, transform, crs):
        return self.postprocessor.predict(merged, labels, transform, crs)

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

        requested_returns = self.cfg.requested_returns
        predictions = self.predict()

        print("Inference complete; starting weighted vote", flush=True)
        merged, objects, transform, crs, obj_score = self.weighted_vote(predictions)

        if 'raw' not in requested_returns:
            del predictions
            scratch_dir = self.scratch_root / 'predict' / self.run_id
            if scratch_dir.exists():
                shutil.rmtree(scratch_dir)

        available = {
            "raw": predictions,
            "unvetoed": merged,
            "predictions": objects,
            "obj_score": obj_score,
            "transform": transform,
            "crs": crs,
        }

        if self.cfg.to_veto:
            print("Applying Veto classifier to merged predictions...", flush=True)
            veto_results = self.veto(merged, objects, transform, crs)
            available.update({
                "predictions": veto_results['predictions'],
                "transform": veto_results['transform'],
                "crs": veto_results['crs']})

            for k in ["rejected", "veto_probability_map"]:
                if k in veto_results:
                    available[k] = veto_results[k]

        return {k: v for k, v in available.items() if k in requested_returns}


    # merges all veto-cleaned predictions and then sieves remaining predictions
    # which are too physically small to be reasonably resolvable given the native DEM. 
    def polygonize(self, array, transform, crs, threshold=0.0):
        """
        Converts a binary mask array into a GeoDataFrame of polygons.

        Parameters
        ----------
        array: np.ndarray
            2D binary mask array where non-zero values represent the objects to be polygonized.
        transform: affine.Affine
            Affine transformation for the array.
        crs: rasterio.crs.CRS
            Coordinate reference system for the output GeoDataFrame.
        threshold: float
            The threshold value for the array. Values above this threshold will be considered as part of the objects to be polygonized.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame containing the polygons of the objects in the input array.
        """

        geoms, ids = [], []
        for geom, v in shapes(array.astype(np.int32), mask=array > threshold, transform=transform, connectivity=8):
            geoms.append(shape(geom))
            ids.append(int(v))

        gdf = gpd.GeoDataFrame({"object_id": ids, "geometry": geoms}, crs=crs)
        return gdf

    def results_to_gdf(self, results):
        cfg = self.cfg
        min_m2 = (cfg.native_res ** 2) * cfg.min_native_pixels
        min_pixels = int(np.ceil(max(min_m2, cfg.min_object_m2) / (self.base_res ** 2)))

        lab = np.asarray(results["predictions"]).astype(np.int32)
        counts = np.bincount(lab.ravel())
        lab[(counts < min_pixels)[lab]] = 0
        gdf = self.polygonize(lab, results["transform"], results["crs"])

        score = results.get("obj_score")
        if score is not None and not gdf.empty:
            gdf["score"] = score[gdf["object_id"].to_numpy() - 1]
        if not gdf.empty:
            gdf["area_m2"] = gdf.geometry.area
        return gdf

    def save_predictions(self, results, output_path):
        """
        Saves the merged predictions to disk in the specified format (shapefile, GeoTIFF, or NumPy array) based on the requested returns in the configuration.

        Parameters
        ----------

        results: dict
            Dictionary containing the results from the sweep method, including merged predictions, transform, and CRS.
        output_path: str or Path
            Path to save the merged predictions.

        Returns
        -------
        gdf: geopandas.GeoDataFrame
            GeoDataFrame containing the merged predictions as polygons.
        """

        cfg = self.cfg
        requested_returns = cfg.requested_returns

        gdf = self.results_to_gdf(results)
        transform, crs = results["transform"], results["crs"]

        # polygonize rejected objects (at veto stage)
        if "rejected" in results:
            rejected = results["rejected"]
            if not np.all(rejected == 0):
                rejected_gdf = self.polygonize(rejected, transform, crs, threshold=0.0)
                rejected_gdf["area_m2"] = rejected_gdf.geometry.area
                results["rejected"] = rejected_gdf
    
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        NPYS = ["raw", "provisional", "obj_score", 'unvetoed']
        GDFS = ['rejected']
        saved = {}

        for req in requested_returns:
            if req == "predictions":
                out_path = output_path / "predictions.shp"
                gdf.to_file(out_path, driver="ESRI Shapefile")
            elif req in GDFS:
                out_path = output_path / f"{req}.shp"
                results[req].to_file(out_path, driver="ESRI Shapefile")
            elif req in NPYS:
                out_path = output_path / f"{req}.npy"
                np.save(out_path, results[req])
            else:
                continue
            saved[req] = out_path

        print("Saved outputs:")
        for k, v in saved.items():
            print(f" {k} saved to: {v}")

        return gdf

    def cleanup(self):
        def report(func, path, exc_info):
            print(f"cleanup failed on {path}: {exc_info[1]}")

        for sub in ("deploy", "predict"):
            path = self.scratch_root / sub / self.run_id
            if path.exists():
                shutil.rmtree(path, onerror=report)

        if self.postprocessor is not None:
            if Path(self.postprocessor.dem_source.temp_path).exists():
                os.unlink(self.postprocessor.dem_source.temp_path) # remove the temporary DEM file created by the PostProcessor's WindowedDataSource

        print(f"Cleaned up scratch directory: {self.scratch_dir}")

# self contained res predictor to decrease amount of calculations needed to be made repetitively!
class ResPredictor():
    """
    Predicts probability maps for a single resolution using the Mask R-CNN model.
    Handles tiling the input DEM, running predictions on each tile, and merging the resullts.
    
    Internally calls ModelWrapper.predict() to run the model on each tile, and merges the results into a single probability map.
    """

    def __init__(self, res, model, norm_stats, run_id, cfg=None):

        cfg = cfg or config.get()

        self.res = res
        self.model = model
        self.norm_stats = norm_stats

        self.run_id = run_id
        self.device = cfg.device
        self.dem_path = cfg.dem_path
        self.tile_size = cfg.tile_size
        self.stride_frac = cfg.stride_frac
        self.batch_size = cfg.batch_size
        self.NODATA = cfg.nodata
        self.sigma_px = cfg.sigma_px
        self.base_res = min(cfg.resolutions)
        self.scratch_dir = REPO_ROOT / "scratch" / "predict" / self.run_id
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

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

        res_value = np.log1p(resolution) / np.log1p(self.norm_stats["max_res"])
        
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
        res_value = np.log1p(resolution) / np.log1p(self.norm_stats["max_res"])
        dem_source = WindowedDataSource.from_tiff_utm(self.dem_path, resolution=resolution)
        progress = None

        try:

            h, w = dem_source.height, dem_source.width
            transform = dem_source.transform

            tag = f"{resolution:g}m"
            dtype = np.float16 if resolution == self.base_res else np.float32 # can be f16 for base as it is never reprojected in GDAL. other resolutions are and GDL doesnt support float16
            memmap_path = self.scratch_dir / f"prob_{tag}.npy"
            prob_map = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=dtype, shape=(h, w))
            for r in range(0, h, 1024):
                prob_map[r:r + 1024] = np.nan

            ts = self.tile_size
            acc = wband = None
            band_top = None
            # calculate streides and tile size in meetres

            stride_px = max(1, int(round(self.tile_size * self.stride_frac)))
            stride_m = stride_px * resolution
            tile_w_m = self.tile_size * resolution

            # calculate tile positions (using stride and res)
            minx, miny, maxx, maxy = dem_source.bounds

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

                for mask, (row0, col0) in zip(masks, batch_locs):
                    pred = mask.numpy() if torch.is_tensor(mask) else np.asarray(mask)
                    r0 = row0 - band_top
                    acc[r0:r0 + ts, col0:col0 + ts] += pred * weight_kernel
                    wband[r0:r0 + ts, col0:col0 + ts] += weight_kernel
                    predicted_tiles += 1

                # clears batch
                batch_tiles.clear()
                batch_locs.clear()

            def finalize(lo, hi):
                # band rows [lo, hi) are done, so divide and write them out
                top = band_top + lo
                a, b = max(top, 0), min(band_top + hi, h)
                if b <= a:
                    return
                p = acc[a - band_top:b - band_top]
                wt = wband[a - band_top:b - band_top]
                out = np.full(p.shape, np.nan, dtype=np.float32)
                ok = wt > 0
                out[ok] = p[ok] / wt[ok]
                prob_map[a:b] = out

            def advance_band(r):
                # called at the start of each strip, r is its top row in the raster
                nonlocal acc, wband, band_top
                new_acc = np.zeros((ts, w), dtype=np.float32)
                new_w = np.zeros((ts, w), dtype=np.float32)
                if band_top is not None:
                    keep = max(0, min(r + ts, band_top + ts) - band_top)
                    finalize(keep, ts)
                    if keep > 0:
                        d = band_top - r
                        new_acc[d:d + keep] = acc[:keep]
                        new_w[d:d + keep] = wband[:keep]
                acc, wband, band_top = new_acc, new_w, r

            with torch.inference_mode():

                for y0 in y_positions:
                    strip, strip_minx = self._make_row_strip(dem_source, y0, x_positions, tile_w_m, resolution)

                    _, strip_row0 = ~transform * (x_positions[0], y0 + tile_w_m)
                    advance_band(int(round(strip_row0)))

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

            if band_top is not None:
                finalize(0, ts)
            prob_map.flush()
            
        finally:
            if progress is not None:
                progress.close()
            os.remove(dem_source.temp_path)

        return prob_map, transform, dem_source.crs, memmap_path

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
        elif name != 'DEM':
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