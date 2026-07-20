# inference at scale x:
# need to tile input DEM, with 4 channels at resolution x
# feed these into the model to generate predictions
# merge tiles back and generate prediction mask (vectorised for memory efficiency probably)

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np
import torch
from ScaleNormalisedDataStack import DataSource, _bounds_inside, OVERLAP_SIGMA_MULTIPLIER
from rasterio.enums import Resampling
from scipy.ndimage import sobel, gaussian_filter, laplace
from ModelWrapper import ModelWrapper
from helpers import calculate_hillshade, compute_tpi
from tqdm.auto import tqdm

# AI assistance with conversion of DataStack logic to a deployment system. 
# dont need to export tiles as they will only be used once at inference
# unlike in training, where several epochs are run

TILE_ORDER = ['DEM', 'RR', 'SLOPE', 'LAPLACE', 'HILLSHADE', 'TPI']

class Deployer():

    def __init__(self, dem_path, model_state_dict, device, resolutions, tile_size=512):
        self.dem_path = dem_path
        self.model_dict = model_state_dict
        self.model = self.build_wrapper()

        # clear memory of now duplicate state dict.
        del self.model_dict

        self.device = device
        self.resolutions = resolutions
        self.tile_size = tile_size

    def build_wrapper(self):
        return ModelWrapper(self.model_dict)

    def predict(self, stride_frac = 0.5):

        self.model.model.eval()
        self.model.model.to(self.device)
        preds = {}
        for res in self.resolutions:
            predictor = ResPredictor(res, self.dem_path, self.model, self.device, self.tile_size, stride_frac=stride_frac)
            prob_map, transform, crs = predictor.predict()
            preds[res] = {"prob": prob_map, "transform": transform, "crs": crs}

        return preds

    def merge_predictions_pyramid(self, preds, base_res=None):

        base_res = base_res or min(preds)
        base = preds[base_res]

        stack = [base["prob"]]

        for res, p in preds.items():
            if res == base_res:
                continue

            resampled = np.full_like(
                base["prob"],
                np.nan,
                dtype=np.float32,
            )

            reproject(
                source=p["prob"],
                destination=resampled,
                src_transform=p["transform"],
                src_crs=p["crs"],
                src_nodata=np.nan,
                dst_transform=base["transform"],
                dst_crs=base["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )

            stack.append(resampled)

        stacked = np.stack(stack, axis=0)

        finite = np.isfinite(stacked)
        filled = np.where(finite, stacked, -np.inf)

        merged = filled.max(axis=0)
        merged[~finite.any(axis=0)] = np.nan

        return merged, base["transform"], base["crs"]
    


# self contained res predictor to decrease amount of calculations needed to be made repetitively!
class ResPredictor():

    def __init__(self, res, dem_path, model, device, tile_size=512, stride_frac = 0.5, batch_size=16):
        
        self.device = device
        self.dem_path = dem_path
        self.model = model
        self.res = res
        self.sigma_m = 12.0 * res
        self.tile_size= tile_size
        self.sigma_px = max(1.0, self.sigma_m / res)
        self.overlap_px = max(4, int(round(OVERLAP_SIGMA_MULTIPLIER * self.sigma_px)))
        self.overlap_m = self.overlap_px * res
        self.padded_size = self.tile_size + 2 * self.overlap_px
        self.stride_frac = stride_frac
        self.batch_size = batch_size


    # cache derivaties and dem for entire x-extent of DEM region.
    # overlap is at 50%, so halves the amount of intensive calculations required.
    def _make_row_strip(self, dem_source, y0, x_positions, tile_w_m, resolution):
        minx = x_positions[0] - self.overlap_m
        maxx = x_positions[-1] + tile_w_m + self.overlap_m
        strip_bounds = (minx, y0 - self.overlap_m, maxx, y0 + tile_w_m + self.overlap_m)

        strip_width_px = int(round((maxx - minx) / resolution))

        dem_padded = dem_source.reproject_to_shape(
            dem_source.crs, strip_bounds, strip_width_px, self.padded_size, resampling=Resampling.cubic
        ).astype(np.float32)

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

        hillshade = calculate_hillshade(dem)
        tpi = compute_tpi(dem, 21)

        strip = {'DEM': dem, 'RR': rr, 'SLOPE': slope, 'LAPLACE': lap, 'HILLSHADE': hillshade, 'TPI': tpi}
        return strip, minx

    # get next x tile from y-horizontal strip
    def _tile_from_strip(self, strip, strip_minx, x0, resolution):
        col0 = int(round((x0 - strip_minx) / resolution))
        col1 = col0 + self.tile_size

        dem_tile = strip['DEM'][:, col0:col1]
        empty = (dem_tile == 0) | np.isnan(dem_tile)
        if empty.sum() / empty.size > 0.1:
            return None

        return np.stack([strip[name][:, col0:col1] for name in TILE_ORDER], axis=0)
    
    # changed for batching approach for optimisation - takes long time to predict 1000kms^2
    def predict(self):

        resolution = self.res

        # msame logic as before
        dem_source = DataSource.from_tiff_utm(self.dem_path, native_res=resolution)
        h, w = dem_source.height, dem_source.width
        transform = dem_source.transform

        # instantiate ooutputs
        prob_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # calculate streides and tile size in meetres
        stride_px = max(1, int(round(self.tile_size * self.stride_frac)))
        stride_m = stride_px * resolution
        tile_w_m = self.tile_size * resolution

        # calculate tile positions (using stride and res)
        minx, miny, maxx, maxy = dem_source.bounds
        x_positions = np.arange(minx, maxx - tile_w_m + 0.5 * resolution, stride_m)
        y_positions = np.arange(miny, maxy - tile_w_m + 0.5 * resolution, stride_m)

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

            # outputs (prob map)
            masks = self.model.predict(x, self.device) 

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

        valid = weight_map > 0
        prob_map[valid] /= weight_map[valid]
        prob_map[~valid] = np.nan

        print(f"Completed {resolution:g} m inference: {predicted_tiles}/{total_tiles} tiles predicted, {skipped_tiles} skipped.")

        return prob_map, transform, dem_source.crs
    
        
    def _normalise_band(self, band, name):
        band = band.astype(np.float32)

        if name == 'TPI':
            return band
        elif name in {"DEM_SLOPE",  "DEM"}:
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