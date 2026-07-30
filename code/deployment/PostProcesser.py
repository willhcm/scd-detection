# post processing

# want a neural network based off spectral bands (just RGB)
# this will only serve to reject/deny predictions: idea being it will see anthropogenic structures, etc and know they arent SCDs
# therefore increasing precision!
# this will not affect recall, the model won't be able to see new candidates, just already predicted ones, with the aim of screening 
# bad predictions out

# also: shape based screening -> angular predictions with one straight side (edge artefact) might be removable?
from tqdm.auto import tqdm
from scipy.ndimage import label as ndi_label
from skimage.measure import regionprops
import numpy as np
import torch
from rasterio.enums import Resampling
import cv2
from ScaleNormalisedDataStack import DataSource

class PostProcessor():

    def __init__(self, tile_size, device, model, rgb_path, batch_size=32,
                 detect_threshold=0.75, veto_threshold=0.85, min_crop_pixels=64, object_fraction=0.5):

        self.tile_size = tile_size
        self.device = device
        self.model = model
        self.rgb_path = rgb_path
        self.rgb_source = DataSource.from_tiff_utm(self.rgb_path, native_res=3, type='RGB')
        self.batch_size = batch_size
        self.detect_threshold = detect_threshold
        self.veto_threshold = veto_threshold
        self.min_crop_pixels = min_crop_pixels
        self.object_fraction = object_fraction

    def _make_tile(self, obj, labeled, transform):

        row_c, col_c = obj.centroid
        cx, cy = transform * (col_c, row_c)

        # size for calculating crop
        pixel_res = abs(transform.a)
        diameter = obj.equivalent_diameter_area * pixel_res

        # getting crop width and tile bounds
        min_crop_width_m = self.min_crop_pixels * pixel_res
        object_crop_width_m = diameter / self.object_fraction
        crop_width_m = max(min_crop_width_m, object_crop_width_m)
        half_width = crop_width_m / 2

        tile_bounds = (cx - half_width, cy - half_width, cx + half_width, cy + half_width)

        # reprojecitng rgb within bounds.
        rgb = self.rgb_source.reproject_to_shape(
            target_crs=self.rgb_source.crs,
            tile_bounds=tile_bounds,
            out_width=self.tile_size,
            out_height=self.tile_size,
            resampling=Resampling.bilinear,
        )

        # if no rgb, skip tile
        if np.all(rgb == 0):
            return None

        # calculate indices to crop the boolean blob straight out of the label array 
        col0 = int(round((tile_bounds[0] - transform.c) / transform.a))
        col1 = int(round((tile_bounds[2] - transform.c) / transform.a))
        row1 = int(round((transform.f - tile_bounds[1]) / -transform.e))
        row0 = int(round((transform.f - tile_bounds[3]) / -transform.e))

        # if invalid (out of bounds) return None
        h, w = labeled.shape
        if row0 < 0 or col0 < 0 or row1 > h or col1 > w:
            return None

        # actually crop the label (unit8 to save space)
        blob_crop = (labeled[row0:row1, col0:col1] == obj.label).astype(np.uint8)

        # resample the mask crop to tile_size with nearest-neighbour, matching rgb's resolution
        mask = self._resize_nearest(blob_crop, self.tile_size)

        return {"rgb": rgb, "mask": mask, "obj": obj, "bounds": tile_bounds}

    @staticmethod
    def _resize_nearest(arr, size):
        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)

    def _prepare_tile(self, tile):
        rgb = tile["rgb"]
        mask = tile["mask"]

        if rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if mask.ndim == 2:
            mask = mask[..., None]

        img = np.concatenate([rgb, mask], axis=-1)
        img = np.transpose(img, (2, 0, 1))

        return img

    def _stamp(self, canvas, obj, value):
        rows, cols = obj.coords[:, 0], obj.coords[:, 1]
        canvas[rows, cols] = value

    def predict(self, prob_raster, transform, crs):
        labeled, n_objects = ndi_label(prob_raster > self.detect_threshold)
        objects = regionprops(labeled, intensity_image=prob_raster)

        stitched = np.zeros_like(prob_raster, dtype=np.float32)
        veto_map = np.zeros_like(prob_raster, dtype=np.float32)

        accepted = 0
        rejected = 0
        skipped = 0
        predicted = 0

        batch_tiles = []
        batch_objs = []

        progress = tqdm(total=len(objects), desc="Vetoing predictions", unit="obj", dynamic_ncols=True)

        def flush_batch():
            nonlocal accepted, rejected, predicted
            if not batch_tiles:
                return

            x = torch.from_numpy(np.stack(batch_tiles, axis=0)).float()
            with torch.inference_mode():
                probs = self.model(x)

            for obj, p in zip(batch_objs, probs):
                p = p.item() if torch.is_tensor(p) else float(p)
                if p >= self.veto_threshold:
                    self._stamp(stitched, obj, obj.mean_intensity)
                    accepted += 1
                else:
                    self._stamp(veto_map, obj, -1)
                    rejected += 1
                predicted += 1

            batch_tiles.clear()
            batch_objs.clear()

        for obj in objects:
            tile = self._make_tile(obj, labeled, transform)
            if tile is None:
                skipped += 1
                progress.update(1)
                continue

            batch_tiles.append(self._prepare_tile(tile))
            batch_objs.append(obj)

            if len(batch_tiles) >= self.batch_size:
                flush_batch()
                progress.set_postfix(predicted=predicted, skipped=skipped)

            progress.update(1)

        flush_batch()  # catch the leftover partial batch
        progress.close()

        print(f'{accepted} tiles retained, {rejected} tiles rejected.')

        return stitched, veto_map, transform, crs