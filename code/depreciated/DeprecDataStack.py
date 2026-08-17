import numpy as np
import rasterio as rio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.warp import Resampling
from rasterio.transform import from_bounds
from code.depreciated.plotting import plot_stack
from scipy.ndimage import uniform_filter, sobel
from scipy.ndimage import grey_opening
from skimage.morphology import disk
import os
from code.depreciated.label_loader import make_labels
os.environ["PROJ_DATA"] = "/opt/anaconda3/envs/IRP/share/proj"

# redundant, keeping for reference for time being.


class DataSource():

    def __init__(self, type: str, data, res, crs, bounds, width, height, transform, band_idx=None):
        self.type = type
        self.data = data
        self.res = res
        self.crs = crs
        self.bounds = bounds
        self.width = width
        self.height = height
        self.transform = transform
        self.band_idx = band_idx

    def reproject(self, target_crs, target_bounds, target_res):
        minx, miny, maxx, maxy = target_bounds
        width = max(1, int((maxx - minx) / target_res))
        height = max(1, int((maxy - miny) / target_res))

        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        out = np.empty((height, width), dtype=self.data.dtype)

        rio.warp.reproject(
            source=self.data,
            destination=out,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest if self.type == 'LABELS' else Resampling.cubic,
        )
        return out

    @classmethod
    def from_tiff(cls, path, type, band_idx: int = 1):
        """reads pixel data and metadata from a tiff.
        Use band_idx for multi-band files (e.g. composite.tif).
        """
        with rio.open(path) as src:
            data = src.read(band_idx)
            return cls(
                type=type,
                data=data,
                res=src.res[0],
                crs=src.crs,
                bounds=src.bounds,
                width=src.width,
                height=src.height,
                transform=src.transform,
                band_idx=band_idx,
            )

    @classmethod
    def from_reference(cls, type, data, ref, band_idx: int = 1):
        """makes datasource using spatial metadata from another DataSource."""
        return cls(
            type=type,
            data=data,
            res=ref.res,
            crs=ref.crs,
            bounds=ref.bounds,
            width=ref.width,
            height=ref.height,
            transform=ref.transform,
            band_idx=band_idx,
        )
    
    @classmethod
    def labels_from_tiff(cls, path, out_path, height, width, transform, profile):

        make_labels(path, out_path, height, width, transform, profile)

        with rio.open(path) as src:
            data = src.read(1)
            return cls(
                type='LABELS',
                data=data,
                res=src.res[0],
                crs=src.crs,
                bounds=src.bounds,
                width=src.width,
                height=src.height,
                transform=src.transform,
                band_idx=1,
            )

# DataStack DEM derivatives
def compute_slope(dem: np.ndarray):
    sx = sobel(dem, axis=0)
    sy = sobel(dem, axis=1)
    return np.sqrt(sx**2 + sy**2).astype(np.float32)


def compute_tpi(dem: np.ndarray, radius: int):
    """ Topographic position index"""
    tpi  = dem - uniform_filter(dem, size=radius * 2 + 1)
    return tpi


def compute_dem_ground(dem: np.ndarray, radius: int = 15):
    """Change to median filter ? """
    return grey_opening(dem, footprint=disk(radius)).astype(np.float32)


def multidirectional_hillshade(dem, cell_size=1.0, altitude_deg=45.0, z_factor=1.0):
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]
    alt = np.radians(altitude_deg)
    dz_dx = np.gradient(dem * z_factor, cell_size, axis=1)
    dz_dy = np.gradient(dem * z_factor, cell_size, axis=0)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)
    hs = np.zeros_like(dem, dtype=np.float64)
    for az_deg in azimuths:
        az = np.radians(360 - az_deg + 90)
        hs += np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs / len(azimuths), 0, 1)

class DataStack():

    # acts an attribute 
    DEFAULT_LAYERS = [
    'DEM',
    'NDVI',
    'R',
    'G',
    'B',
    'DEM_SLOPE',
    'TPI_75',
    'TPI_150',
    'DEM_GROUND',
    'HILLSHADE',
]

    def __init__(self, sources: dict, crs, res, bounds, labelled=True, features=None):
        """
        Holds references to DataSource objects only — no reprojection or
        stacking happens at init time.heavy work is deferred to
        tile_and_export, keeping RAM usage low.
        """
        self.sources = sources
        self.target_crs = crs
        self.target_res = res
        self.target_bounds = bounds
        self.labelled = labelled

        # always for custom definition of what features to use - developed for hillshade model.
        self.layer_names = features or self.DEFAULT_LAYERS.copy()

        if self.labelled:
            self.layer_names.append('LABELS')

        self.layer_index = {
        name: i for i, name in enumerate(self.layer_names)
    }
        
        if any(k in self.layer_index for k in ['R', 'G', 'B']):
            # add all to set, shouldn't break if they aren't in layer_index either
            self.is_rgb = True
        
        self.is_rgb = False

        if self.is_rgb:
            self._rgb_percentiles = self._compute_rgb_percentiles()


    # change to only export certain tile types (e.g. has labels and centered i.e. without edge artefact?)
    def tile_and_export(
        self,
        tile_size: int,
        out_path: str,
        empty_threshold: float = 0.2,
        overlap: float = 0.15,
    ):

        Path(out_path).mkdir(parents=True, exist_ok=True)
        tile_bounds_list = list(self._generate_tile_bounds(tile_size, overlap))

        print(f"Exporting {len(tile_bounds_list)} tiles ")

        exported = 0
        skipped  = 0

        for tile_bounds in tile_bounds_list:
            name = self._tile_name(tile_bounds)
            tile_data = self._build_tile_stack(tile_bounds)

            if self._is_mostly_empty(tile_data, empty_threshold):
                del tile_data
                skipped += 1
                continue

            if self.labelled:
                label_idx = self.layer_index['LABELS']
                labels = tile_data[label_idx].astype(np.uint8)
                scd_present = bool(labels.sum() != 0)
                scd_pixel_fraction = float(labels.sum()) / float(labels.size)
                confirmed = True

                np.savez_compressed(
                    str(Path(out_path) / name),
                    image=tile_data,
                    labels=labels,
                    scd_present=np.array(scd_present),
                    scd_pixel_fraction=np.array(scd_pixel_fraction),
                    confirmed=np.array(confirmed),
                    tile_bounds=np.array(tile_bounds),
                    layer_names=np.array(self.layer_names),
                )

            else:

                np.savez_compressed(
                    str(Path(out_path) / name),
                    image=tile_data,
                    tile_bounds=np.array(tile_bounds),
                    layer_names=np.array(self.layer_names),
                )

            del tile_data
            exported += 1
            if exported % 10 == 0:
                print(f"  {exported} / {len(tile_bounds_list)} tiles saved...")

        print(f"Done. {exported} tiles exported, {skipped} skipped (>{empty_threshold:.0%} empty).")

    def view(self, figsize=(8, 8)):
        stack, layer_index = self._build_full_stack()
        plot_stack(stack, layer_index, figsize=figsize)
        del stack

    def view_region(self, cx: float, cy: float, size: int, figsize=(10, 10)):
        minx, miny, maxx, maxy = self.target_bounds
        half   = (size * self.target_res) / 2
        bounds = (
            max(cx - half, minx), max(cy - half, miny),
            min(cx + half, maxx), min(cy + half, maxy),
        )
        print(f"Viewing UTM region: {bounds}")
        tile_data = self._build_tile_stack(bounds)
        plot_stack(tile_data, self.layer_index, figsize=figsize)
        del tile_data

    def _generate_tile_bounds(self, tile_size: int, overlap: float = 0.0,
                               min_fraction: float = 0.5):
        """
        Generate tile bounding boxes across the full stack extent."""
        minx, miny, maxx, maxy = self.target_bounds
        tile_width = tile_size * self.target_res
        stride = tile_width * (1.0 - overlap)
        min_size = tile_width * min_fraction

        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                x1 = x
                y1 = y
                x2 = min(x + tile_width, maxx)
                y2 = min(y + tile_width, maxy)

                # Drop boundary slivers that are too small to be useful (i.e. outside raster bounds)
                if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
                    # help from AI on this function (wanted to make it clean and efficient, and a generator seemed like the
                    # best way of doing this)
                    yield (x1, y1, x2, y2)

                x += stride
            y += stride

    def _build_tile_stack(self, tile_bounds):
        # assistance from ChatGPT in making features selectable.
        # this runs very slowly still, need to optimise for future usage on large areas.
        # definetly dont need to redefine reproj every time ...
        def reproj(name):
            return self.sources[name].reproject(
                self.target_crs,
                tile_bounds,
                self.target_res
            ).astype(np.float32)

        # Always compute DEM
        dem = reproj('DEM')

        available = {'DEM': dem}

        if self.is_rgb:
            r   = reproj('R')
            g   = reproj('G')
            b   = reproj('B')
            nir = reproj('NIR')

            ndvi = np.where(
                nir + r == 0,
                0,
                (nir - r) / (nir + r + 1e-8)
            )

            available.update({'R': r, 'G': g, 'B': b, 'NIR': nir, 'NDVI': ndvi})

        # derived features (only compute if specified in __init__!)
        if 'DEM_SLOPE' in self.layer_names:
            available['DEM_SLOPE'] = compute_slope(dem)

        if 'TPI_75' in self.layer_names:
            available['TPI_75'] = compute_tpi(dem, radius=75)

        if 'TPI_150' in self.layer_names:
            available['TPI_150'] = compute_tpi(dem, radius=150)

        if 'DEM_GROUND' in self.layer_names:
            available['DEM_GROUND'] = compute_dem_ground(dem, radius=15)

        if 'HILLSHADE' in self.layer_names:
            available['HILLSHADE'] = multidirectional_hillshade(dem, 3)

        if 'FILL_DIFF' in self.layer_names:
            available['FILL_DIFF'] = reproj('FILL_DIFF')

        if 'FLOW_ACC' in self.layer_names:
            available['FLOW_ACC'] = reproj('FLOW_ACC')

        if self.labelled:
            available['LABELS'] = self.sources['LABELS'].reproject(
                self.target_crs,
                tile_bounds,
                self.target_res
            )

        layers = [available[name] for name in self.layer_names]

        return np.stack(layers, axis=0)
    

    def _build_full_stack(self):
        # very expensive for large region
        return self._build_tile_stack(self.target_bounds), self.layer_index

    def _empty_fraction(self, tile_data: np.ndarray):
        # changed from computing empty frac for optical and DEM to now being dynamic with layers used/selected.
        fractions = []

        # labels often 0, so need to ignore here.
        for name in self.layer_names:
            if name == 'LABELS':
                continue


            idx = self.layer_index[name]
            layer = tile_data[idx]
            empty = (layer == 0) | np.isnan(layer)
            fractions.append(float(empty.sum()) / empty.size)
        return max(fractions)

    def _is_mostly_empty(self, tile_data: np.ndarray, empty_threshold: float = 0.5):
        return self._empty_fraction(tile_data) > empty_threshold

    def _tile_name(self, tile_bounds: tuple):
        minx, miny, maxx, maxy = tile_bounds
        return f"tile_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}"

    def _compute_rgb_percentiles(self, sample_size=10_000):
        """Per-band scene-wide percentiles for display normalisation."""
        percentiles = {}
        for band in ('R', 'G', 'B'):
            data = self.sources[band].data.ravel().astype(np.float32)
            step = max(1, len(data) // sample_size)
            sample = data[::step]
            percentiles[band] = (np.percentile(sample, 2), np.percentile(sample, 98))
        return percentiles
    
    def view_tile(self, index=0):
        tile_generator = self._generate_tile_bounds(512)
        tiles = list(tile_generator)
        print(f"{len(tiles)} tiles available")
        tile_bounds = tiles[index]
        stack = self._build_tile_stack(tile_bounds)
        plot_stack(stack, self.layer_index, labels=False)
