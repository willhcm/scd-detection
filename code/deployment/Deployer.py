# inference at scale x:
# need to tile input DEM, with 4 channels at resolution x
# feed these into the model to generate predictions
# merge tiles back and generate prediction mask (vectorised for memory efficiency probably)

# ChatGPT assistance with these lines for importing whenever used

from pathlib import Path
import sys
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from rasterio.warp import reproject
import os
import numpy as np
import torch
from loaders.Tiler import OVERLAP_SIGMA_MULTIPLIER
from rasterio.enums import Resampling
from scipy.ndimage import sobel, gaussian_filter, laplace
from deployment.ModelWrapper import ModelWrapper
from modelling.helpers import calculate_hillshade
from tqdm.auto import tqdm
from deployment.PostProcesser import PostProcessor, WindowedDataSource
from modelling.Veto import VetoClassifier
import geopandas as gpd
from rasterio.features import shapes, sieve
from shapely.geometry import shape
from helpers.VetoHelpers import VETO_SCALAR_NAMES
from modelling.DEMVeto import DEM_Based_Vetoer
from scipy.ndimage import binary_dilation
from affine import Affine

# AI assistance with conversion of DataStack logic to a deployment system. 
# dont need to export tiles as they will only be used once at inference
# unlike in training, where several epochs are run

TILE_ORDER = ['DEM',  'DEM_SLOPE', 'RR','LAPLACE', 'HILLSHADE']

class Deployer:

    # dosctring generated with assistance from Github Copilot Free.
    """
    Deployer class for running inference on a DEM and RGB dataset using a Mask R-CNN model and a Veto classifier.
    This class handles the loading of the DEM and RGB data, running predictions at multiple resolutions, merging the predictions, and applying the Veto classifier to filter out false positives.

    Parameters
    ----------
    dem_path: str
        Path to the DEM file.      
    rgb_path: str
        Path to the RGB file.
    model_state_dict: dict
        State dictionary of the trained Mask R-CNN model.
    veto_model_dict: dict
        State dictionary of the trained Veto classifier model.
    device: str
        Device to run the inference on (e.g., 'cpu' or 'cuda').
    resolutions: list
        List of resolutions to run the inference at.
    tile_size: int, optional
        Size of the tiles to use for inference. Default is 512.
    veto: bool, optional  
        Whether to apply the Veto classifier to filter out false positives. Default is True.
    veto_tile_size: int, optional  
        Size of the tiles to use for the Veto classifier. Default is 96.
    veto_context_tile_size: int, optional   
        Size of the context tiles to use for the Veto classifier. Default is 224.
    veto_batch_size: int, optional
        Batch size to use for the Veto classifier. Default is 16.
    veto_threshold: float, optional
        Threshold to use for the Veto classifier. Default is 0.90.
    context_scale: float, optional
        Scale to use for the context tiles in the Veto classifier. Default is 4.0.
    min_context_width_m: float, optional
        Minimum width of the context tiles in meters for the Veto classifier. Default is 768.0.
    max_context_width_m: float, optional
        Maximum width of the context tiles in meters for the Veto classifier. Default is 4000.0.
    native_res: float, optional 
        Native resolution of the DEM in meters. Default is 30.
    """

    def __init__(
        self,
        dem_path,
        model_state_dict,
        veto_model_dict,
        device,
        resolutions,
        rgb_path=None,
        tile_size=512,
        rgb_veto=False,
        NODATA=0,
        veto_tile_size=96, 
        veto_context_tile_size=224,
        veto_batch_size=16,
        veto_threshold=0.6, # tune
        score_threshold=0.65, # tune
        mask_threshold=0.50, # tune
        context_scale=4.0,
        min_context_width_m=500.0,
        max_context_width_m=4000.0,
        native_res = 30,
        stride_frac=0.75,
       to_veto=True,
    ):

        self.base_res = None
        self.NODATA = NODATA
        self.mask_threshold = mask_threshold
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

        self.model_dict = model_state_dict
        self.model = self.build_wrapper()
        del self.model_dict

        self.dem_path = dem_path
        self.rgb_path = rgb_path
        self.device = device
        self.resolutions = resolutions
        self.native_res = native_res
        self.stride_frac = stride_frac
        # Mask R-CNN deployment tile size.
        self.tile_size = tile_size
        self.rgb_veto = rgb_veto
        self.to_veto = to_veto

        # if using spectral data
        if self.rgb_veto:

            self.veto_model = VetoClassifier(
                scalar_dim=len(VETO_SCALAR_NAMES),
                dropout=0.3,
            )

            self.veto_model.load_state_dict(
                veto_model_dict
            )

            self.veto_model.to(self.device)
            self.veto_model.eval()

        # if not using spectral data 
        else:
            self.veto_model = DEM_Based_Vetoer()
            self.veto_model.load_state_dict(veto_model_dict)
            self.veto_model.to(self.device)
            self.veto_model.eval()

        self.veto_tile_size = veto_tile_size
        self.veto_context_tile_size = veto_context_tile_size
        self.veto_batch_size = veto_batch_size
        self.veto_threshold = veto_threshold

        self.context_scale = context_scale
        self.min_context_width_m = min_context_width_m
        self.max_context_width_m = max_context_width_m

    def build_wrapper(self):
        return ModelWrapper(self.model_dict, self.score_threshold, self.mask_threshold)

    def predict(self):
        """
        Predicts the probability maps for each resolution in self.resolutions using the Mask R-CNN model.
        """

        self.model.model.eval()
        self.model.model.to(self.device)
        preds = {}
        for res in self.resolutions:
            predictor = ResPredictor(
                res,
                self.dem_path,
                self.model,
                self.device,
                self.tile_size,
                stride_frac=self.stride_frac,
                NODATA=self.NODATA)

            prob_map, transform, crs = predictor.predict()

            preds[res] = {
                "prob": prob_map,
                "transform": transform,
                "crs": crs,
    }

        return preds   


    def merge_predictions_pyramid(
            self,
            preds,
            support_threshold=0.65,
            min_support=2,
            single_scale_keep=0.90,
            band=1024):

        base_res = min(preds)
        self.base_res = base_res
        base = preds[base_res]
        transform, crs = base["transform"], base["crs"]
        h, w = base["prob"].shape

        scratch_dir = Path("/scratch_root/wm722/scd-detection/") / 'merge' 
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
                sup += ok & (cur >= support_threshold)
                best = np.fmax(best, cur)

            weak = (cov > 0) & (sup < min_support) & (best < single_scale_keep)
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

        postproc = PostProcessor(
            context_tile_size=self.veto_context_tile_size,
            device=self.device,
            model=self.veto_model,
            rgb_path=self.rgb_path,
            dem_path=self.dem_path,
            batch_size=self.veto_batch_size,
            detect_threshold=self.score_threshold,
            veto_threshold=self.veto_threshold,
            min_crop_pixels=64,
            object_fraction=0.5,
            context_scale=self.context_scale,
            min_context_width_m=self.min_context_width_m,
            max_context_width_m=self.max_context_width_m,
            rgb=self.rgb_veto)

            
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

        If self.rgb_veto is False:
            predictions: dict
                Dictionary of predictions at each resolution.
            merged: np.ndarray
                Merged probability map.
            transform: affine.Affine
                Affine transform of the merged probability map.
            crs: rasterio.crs.CRS
                Coordinate reference system of the merged probability map.
            support: np.ndarray
                Support count for each pixel in the merged probability map.
            coverage: np.ndarray
                Coverage count for each pixel in the merged probability map.

        If self.rgb_veto is True:
            predictions: dict
                Dictionary of predictions at each resolution.
            cleaned: np.ndarray
                Merged and Veto-cleaned probability map.
            rejected: np.ndarray
                Rejected predictions by the Veto classifier.
            veto_probability_map: np.ndarray
                Probability map from the Veto classifier.
            out_transform: affine.Affine
                Affine transform of the Veto-cleaned probability map.
            out_crs: rasterio.crs.CRS
                Coordinate reference system of the Veto-cleaned probability map.
            support: np.ndarray
                Support count for each pixel in the merged probability map.
            coverage: np.ndarray
                Coverage count for each pixel in the merged probability map.
        """

        predictions = self.predict()

        (merged, transform, crs, support, coverage) = self.merge_predictions_pyramid(predictions)

        if self.to_veto:
            accepted, rejected, veto_probability_map, out_transform, out_crs = self.veto(merged, transform, crs)

            out = {
                "raw": predictions,
                "predictions": accepted,
                "rejected": rejected,
                "veto_probability_map": veto_probability_map,
                "transform": out_transform,
                "crs": out_crs,
                "support": support,
                "coverage": coverage
            }

        else:
            out = {
                "raw": predictions,
                "predictions": merged,
                "transform": transform,
                "crs": crs,
                "support": support,
                "coverage": coverage
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

        # native res should determine smallest resolvable objects.
        min_m2 = (self.native_res ** 2) * 10 # resolution constrain, an area.
        min_scd_size = 20 ** 2 # minimum SCD size reasonable wanted. 20m x 20m = 400m^2, an area.
        min_pixels = int(np.ceil(max(min_m2, min_scd_size) / (self.base_res ** 2)))

        binary = (
            np.isfinite(cleaned)
            & (cleaned >= 0.40)).astype(np.uint8)
        
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

    def __init__(self, res, dem_path, model, device, tile_size=512, stride_frac = 0.75, batch_size=16, NODATA=-3.4028234663852886e38):
        
        self.device = device
        self.dem_path = dem_path
        self.model = model
        self.res = res
        self.tile_size = tile_size
        self.stride_frac = stride_frac
        self.batch_size = batch_size
        self.NODATA = NODATA

        self.sigma_px = 12

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
        hillshade = calculate_hillshade(dem_padded)

        dem = dem_padded[s:e, :]
        rr = rr[s:e, :]
        slope = slope[s:e, :]
        lap = lap[s:e, :]
        hillshade = hillshade[s:e, :]
        invalid = invalid[s:e, :]

        strip = {'DEM': dem, 'RR': rr, 'DEM_SLOPE': slope, 'LAPLACE': lap, 'HILLSHADE': hillshade, 'RES': res_value, 'INVALID': invalid}
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

        # msame logic as before
        dem_source = WindowedDataSource.from_tiff_utm(self.dem_path, native_res=resolution)
        h, w = dem_source.height, dem_source.width
        transform = dem_source.transform

        # saved on file to reduce memory failure
        scratch = Path("/scratch_root/wm722/scd-detection/work") / 'predict' / f"{resolution:g}m"
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

        progress.close()

        # h is DEM height, r is rows, step is number of rows to split it up into.
        for r in range(0, h, 256):
            p, wt = prob_map[r:r + 1024], weight_map[r:r + 1024]
            ok = wt > 0
            p[ok] /= wt[ok]
            p[~ok] = np.nan
        prob_map.flush()
        del weight_map

        print(f"Completed {resolution:g} m inference: {predicted_tiles}/{total_tiles} tiles predicted, {skipped_tiles} skipped.")

        return prob_map, transform, dem_source.crs
    
        
    def _normalise_band(self, band, name):
        band = band.astype(np.float32)

        if name == "HILLSHADE":
            return (band - band.mean()) / (band.std() + 1e-6)

        if name in {"DEM_SLOPE",  "DEM"}:
            transformed = band
        else:
            transformed = np.sign(band) * np.log1p(np.abs(band))

        median = np.median(transformed)
        q75, q25 = np.percentile(transformed, [75, 25])
        iqr = q75 - q25
        scaled = (transformed - median) / (iqr + 1e-6)

        return scaled
    
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

