# inference at scale x:
# need to tile input DEM, with 4 channels at resolution x
# feed these into the model to generate predictions
# merge tiles back and generate prediction mask (vectorised for memory efficiency probably)

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import numpy as np
import torch
from datastack import DataSource, _bounds_inside, OVERLAP_SIGMA_MULTIPLIER
from rasterio.enums import Resampling
from scipy.ndimage import sobel, gaussian_filter, laplace
from SegNet import SegNet
from MaskRCNN import MaskRCNN
from FPNCentreNet import FPNCentreNet
from ModelWrapper import ModelWrapper

# AI assistance with conversion of DataStack logic to a deployment system. 
# dont need to export tiles as they will only be used once at inference
# unlike in training, where several epochs are run

class Deployer():

    def __init__(self, dem_path, model_type, model_state_dict, device, resolutions, tile_size=512, stride_frac=0.75):
        self.dem_path = dem_path
        self.model_type = self.model_type
        self.model_dict = model_state_dict
        self.model = self.build_wrapper()

        # clear memory of now duplicate state dict.
        del self.model_dict

        self.device = device
        self.resolutions = resolutions
        self.tile_size = tile_size
        self.stride_frac = stride_frac

    def build_wrapper(self):
        return ModelWrapper(self.model_type, self.model_dict)

    def _make_tile(self, dem_source, bounds, res):

        sigma_m = 10.0 * res
        sigma_px = max(1.0, sigma_m / res)
        overlap_px = max(4, int(round(OVERLAP_SIGMA_MULTIPLIER * sigma_px)))
        overlap_m = overlap_px * res
        padded_size = self.tile_size + 2 * overlap_px

        # oversample DEM to reduce edge artefacts
        minx, miny, maxx, maxy = bounds
        padded_bounds = (minx - overlap_m, miny - overlap_m, maxx + overlap_m, maxy + overlap_m)

        if not _bounds_inside(padded_bounds, dem_source.bounds):
            return None

        dem_padded = dem_source.reproject_to_shape(
            dem_source.crs, padded_bounds, padded_size, padded_size, resampling=Resampling.cubic
        ).astype(np.float32)

        # gradient 
        sx = sobel(dem_padded, axis=0)
        sy = sobel(dem_padded, axis=1)
        slope = np.sqrt(sx ** 2 + sy ** 2)


        # residual relief (variable sigma depending on res)
        smoothed = gaussian_filter(dem_padded.astype(np.float64), sigma=sigma_px)
        rr = (dem_padded - smoothed).astype(np.float32)

        # laplace
        lap = laplace(dem_padded)

        # slice derivative layers to correct shape 
        s, e = overlap_px, overlap_px + self.tile_size
        slope, rr, lap = slope[s:e, s:e], rr[s:e, s:e], lap[s:e, s:e]

        dem = dem_source.reproject_to_shape(
            dem_source.crs, bounds, self.tile_size, self.tile_size, resampling=Resampling.cubic
        ).astype(np.float32)

        return np.stack([dem, rr, slope, lap], axis=0)

    def _predict_tile(self, tile):
        x = torch.from_numpy(tile).unsqueeze(0).float()
        preds = self.model.predict(x)

        return preds

    def predict_at_resolution(self, resolution):

        dem_source = DataSource.from_tiff_utm(self.dem_path, native_res=resolution)
        h, w = dem_source.height, dem_source.width
        transform = dem_source.transform

        # decide on weightings, centre should be weighting more highly but needs to depend on overlap size.
        prob_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        stride_m = int(self.tile_size * self.stride_frac) * resolution
        tile_w_m = self.tile_size * resolution
        minx, miny, maxx, maxy = dem_source.bounds

        x0 = minx
        while x0 < maxx:
            y0 = miny
            while y0 < maxy:
                bounds = (x0, y0, x0 + tile_w_m, y0 + tile_w_m)
                tile = self._make_tile(dem_source, bounds, resolution)

                if tile is not None:
                    pred = self._predict_tile(tile)
                    col0, row0 = ~transform * (x0, y0 + tile_w_m)
                    col0, row0 = int(round(col0)), int(round(row0))
                    row1, col1 = row0 + self.tile_size, col0 + self.tile_size

                    if row1 <= h and col1 <= w:
                        prob_map[row0:row1, col0:col1] += pred
                        weight_map[row0:row1, col0:col1] += 1.0

                y0 += stride_m
            x0 += stride_m

        weight_map[weight_map == 0] = 1.0
        prob_map /= weight_map

        return prob_map, transform, dem_source.crs

    def predict(self):

        preds = {}
        for res in self.resolutions:
            prob_map, transform, crs = self.predict_at_resolution(res)
            preds[res] = {"prob": prob_map, "transform": transform, "crs": crs}

        return preds

    def merge_predictions_pyramid(self, preds, base_res=None):

        base_res = base_res or min(preds)
        base = preds[base_res]
        stack = [base["prob"]]

        for res, p in preds.items():
            if res == base_res:
                continue
            resampled = np.empty_like(base["prob"])
            reproject(
                source=p["prob"],
                destination=resampled,
                src_transform=p["transform"],
                src_crs=p["crs"],
                dst_transform=base["transform"],
                dst_crs=base["crs"],
                resampling=Resampling.bilinear,
            )
            stack.append(resampled)

        # need to decide on how to stitch predictions, max works but maybe not optimal.
        return np.stack(stack, axis=0).max(axis=0), base["transform"], base["crs"]