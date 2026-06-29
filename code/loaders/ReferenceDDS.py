import numpy as np
import rasterio as rio
import geopandas as gpd
from pathlib import Path
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.crs import CRS
from rasterio.transform import from_bounds, array_bounds
from rasterio.features import rasterize
from shapely.geometry import box
from scipy.ndimage import uniform_filter, sobel, median_filter, gaussian_filter, laplace
from skimage.morphology import disk
from scipy.ndimage import label as scipy_label
from pyproj import Transformer
from sklearn.cluster import DBSCAN

# Claude help with ShapeLabels Class, code formatting and adaption of pre-exising DataStack.py to centre tiles around positive labels
# and cluster adjacent scds into one tile to avoid tile repetition.
# FEATURE REGISTRY idea also aided by GenAI. 

OVERLAP_SIGMA_MULTIPLIER = 3


class ShapeLabels:
    """
    Holds a GeoDataFrame of label polygons and rasterises on demand per tile.
    """

    def __init__(self, path):
        self.gdf = gpd.read_file(path)

    def centroids(self, target_crs):
        """(cx, cy) in target_crs for each polygon centroid."""
        gdf = self.gdf.to_crs(target_crs)
        return [(geom.centroid.x, geom.centroid.y)
                for geom in gdf.geometry if geom is not None]

    def rasterise(self, tile_bounds, tile_size, target_crs):
        minx, miny, maxx, maxy = tile_bounds
        transform = from_bounds(minx, miny, maxx, maxy, tile_size, tile_size)

        gdf = self.gdf.to_crs(target_crs)
        clipped = gdf[gdf.intersects(box(minx, miny, maxx, maxy))]

        if clipped.empty:
            return np.zeros((tile_size, tile_size), dtype=np.uint8)

        shapes = [(geom, 1) for geom in clipped.geometry if geom is not None]
        return rasterize(
            shapes=shapes,
            out_shape=(tile_size, tile_size),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )


def _utm_epsg(src_crs, bounds):
    """Infer the UTM EPSG code from a raster's bounds (in its native CRS)."""
    transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
    minx, miny, maxx, maxy = bounds
    lon = (minx + maxx) / 2
    lat = (miny + maxy) / 2
    lon_wgs, lat_wgs = transformer.transform(lon, lat)
    zone = int((lon_wgs + 180) / 6) + 1
    epsg = 32600 + zone if lat_wgs >= 0 else 32700 + zone
    return epsg


class DataSource:

    def __init__(self, type, data, res, crs, bounds, width, height, transform, band_idx=None):
        self.type = type
        self.data = data
        self.res = res
        self.crs = crs
        self.bounds = bounds
        self.width = width
        self.height = height
        self.transform = transform
        self.band_idx = band_idx

    def reproject(self, target_crs, tile_bounds, target_res):
        minx, miny, maxx, maxy = tile_bounds
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
            resampling=Resampling.cubic,
        )
        return out

    @classmethod
    def from_tiff(cls, path, type, band_idx=1):
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
    def from_tiff_utm(cls, path, target_res, band_idx=1):
        """Load a tiff and reproject to the appropriate UTM zone at target_res (metres)."""
        with rio.open(path) as src:
            utm_crs = CRS.from_epsg(_utm_epsg(src.crs, src.bounds))
            transform, width, height = calculate_default_transform(
                src.crs, utm_crs, src.width, src.height, *src.bounds,
                resolution=target_res,
            )
            data = np.empty((height, width), dtype=np.float32)
            reproject(
                source=rio.band(src, band_idx),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=utm_crs,
                resampling=Resampling.cubic,
            )

        bounds = array_bounds(height, width, transform)
        return cls(
            type='DEM',
            data=data,
            res=target_res,
            crs=utm_crs,
            bounds=bounds,
            width=width,
            height=height,
            transform=transform,
            band_idx=band_idx,
        )

    @classmethod
    def from_reference(cls, type, data, ref, band_idx=1):
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


def _slope(a):
    sx = sobel(a['DEM'], axis=0)
    sy = sobel(a['DEM'], axis=1)
    return np.sqrt(sx**2 + sy**2).astype(np.float32)


def _hillshade(cell_size=1.0, altitude_deg=45.0, z_factor=1.0):
    def _fn(a):
        dem = a['DEM']
        alt = np.radians(altitude_deg)
        dz_dx = np.gradient(dem * z_factor, cell_size, axis=1)
        dz_dy = np.gradient(dem * z_factor, cell_size, axis=0)
        slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect = np.arctan2(-dz_dy, dz_dx)
        hs = np.zeros_like(dem, dtype=np.float64)
        for az_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            az = np.radians(360 - az_deg + 90)
            hs += np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - aspect)
        return np.clip(hs / 8, 0, 1).astype(np.float32)
    return _fn


def _make_rr(sigma_px):
    """Residual relief: local DEM minus gaussian-smoothed regional trend."""
    def _fn(a):
        smoothed = gaussian_filter(a['DEM'].astype(np.float64), sigma=sigma_px)
        return (a['DEM'] - smoothed).astype(np.float32)
    return _fn


def _laplace(a):
    return laplace(a['DEM'])


def _build_feature_registry(sigma_px):
    return {
        'DEM_SLOPE': {'deps': ['DEM'],        'fn': _slope},
        'HILLSHADE': {'deps': ['DEM'],        'fn': _hillshade(cell_size=1.0)},
        'RR':        {'deps': ['DEM'],        'fn': _make_rr(sigma_px)},
        'LAPLACE': {'deps': ['DEM'],        'fn': _laplace},
    }


def _centred_bounds(cx, cy, tile_size, res, global_bounds):
    half = (tile_size * res) / 2
    minx, miny, maxx, maxy = global_bounds
    return (
        max(cx - half, minx), max(cy - half, miny),
        min(cx + half, maxx), min(cy + half, maxy),
    )


class DataStack:

    DEFAULT_LAYERS = [
        'DEM',
        'DEM_SLOPE',
        'HILLSHADE',
        'RR',
        'LAPLACE'
    ]

    def __init__(self, dem_source: DataSource, label_shp: ShapeLabels = None,
                 features=None, sigma_px: int = 10):

        self.dem_source = dem_source
        self.label_shp = label_shp
        self.labelled = label_shp is not None
        self.sigma_px = sigma_px
        self.feature_registry = _build_feature_registry(sigma_px)

        self.target_crs = dem_source.crs
        self.target_res = dem_source.res

        if label_shp is not None:
            self.target_bounds = tuple(label_shp.gdf.to_crs(self.target_crs).total_bounds)
        else:
            self.target_bounds = dem_source.bounds

        self.layer_names = features or self.DEFAULT_LAYERS.copy()
        if self.labelled:
            self.layer_names = [l for l in self.layer_names if l != 'LABELS']
            self.layer_names.append('LABELS')

        self.layer_index = {name: i for i, name in enumerate(self.layer_names)}

    def tile_and_export(
        self,
        tile_size: int,
        out_path: str,
        empty_threshold: float = 0.2,
        negative: float = 0.15,
        seed: int = 42,
        cluster = True
    ):
        Path(out_path).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        if self.labelled:
            self._export_labelled(tile_size, out_path, empty_threshold, negative, rng, cluster=cluster)
        else:
            self._export_unlabelled(tile_size, out_path, empty_threshold)

    def _cluster_centroids(self, tile_size, small_area_fraction=0.1):
        
        # small objects are clustered with DBSCAN so tight groups don't produce near-identical tiles.

        gdf = self.label_shp.gdf.to_crs(self.target_crs)
        tile_width = tile_size * self.target_res
        size_threshold = (tile_width * small_area_fraction) ** 2

        large = gdf[gdf.geometry.area >= size_threshold]
        small = gdf[gdf.geometry.area <  size_threshold]

        large_centroids = [(g.centroid.x, g.centroid.y) for g in large.geometry if g is not None]

        clustered_small = []
        if len(small) > 0:
            coords = np.array([(g.centroid.x, g.centroid.y) for g in small.geometry if g is not None])
            labels = DBSCAN(eps=tile_width, min_samples=1).fit(coords).labels_
            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                clustered_small.append(coords[mask].mean(axis=0).tolist())

        return large_centroids + clustered_small

    def _export_labelled(self, tile_size, out_path, empty_threshold, negative, rng, cluster=True):
        centroids = self.label_shp.centroids(self.target_crs)
        print(f"Found {len(centroids)} positive tiles")

        # somtimes i want to have repeated, translated tiles for training.
        if cluster:
            centroids = self._cluster_centroids(tile_size)
        
        exported_pos = skipped = 0
        for cx, cy in centroids:
            bounds = _centred_bounds(cx, cy, tile_size, self.target_res, self.dem_source.bounds)
            tile = self._build_tile(bounds, tile_size)

            if tile is None or self._is_mostly_empty(tile, empty_threshold):
                skipped += 1
                continue

            label = tile[self.layer_index['LABELS']]
            coverage = float(label.sum()) / float(label.size)

            if coverage > 0.6:
                skipped += 1
                continue

            edge_threshold = 0.2
            top    = label[0, :].sum()  / label.shape[1]
            bottom = label[-1, :].sum() / label.shape[1]
            left   = label[:, 0].sum()  / label.shape[0]
            right  = label[:, -1].sum() / label.shape[0]

            if max(top, bottom, left, right) > edge_threshold:
                skipped += 1
                continue

            self._save_tile(tile, bounds, out_path, self._tile_name(bounds, 'pos'))
            del tile
            exported_pos += 1

        n_neg = max(1, int(exported_pos * negative))
        print(f"Sampling {n_neg} negative tiles")

        exported_neg = attempts = 0
        while exported_neg < n_neg and attempts < n_neg * 20:
            attempts += 1
            bounds = self._random_tile_bounds(tile_size, rng)
            tile = self._build_tile(bounds, tile_size)

            if tile is None or self._is_mostly_empty(tile, empty_threshold):
                continue

            if tile[self.layer_index['LABELS']].sum() > 0:
                del tile
                continue

            self._save_tile(tile, bounds, out_path, self._tile_name(bounds, 'neg'))
            del tile
            exported_neg += 1

        print(f"Done. {exported_pos} positive, {exported_neg} negative.")

    def _export_unlabelled(self, tile_size, out_path, empty_threshold):
        bounds_list = list(self._grid_tile_bounds(tile_size))
        print(f"Exporting {len(bounds_list)} tiles (unlabelled)")

        exported = skipped = 0
        for bounds in bounds_list:
            tile = self._build_tile(bounds, tile_size)
            if tile is None or self._is_mostly_empty(tile, empty_threshold):
                skipped += 1
                continue
            self._save_tile(tile, bounds, out_path, self._tile_name(bounds))
            del tile
            exported += 1
            if exported % 10 == 0:
                print(f"  {exported} / {len(bounds_list)} saved...")

        print(f"Done. {exported} exported, {skipped} skipped.")

    def _build_tile(self, tile_bounds, tile_size):
        # Expand bounds by overlap so gaussian filter has context beyond tile edges
        # avoids edge artefacts
        overlap_px = OVERLAP_SIGMA_MULTIPLIER * self.sigma_px
        overlap_m  = overlap_px * self.target_res
        padded_size = tile_size + 2 * overlap_px

        minx, miny, maxx, maxy = tile_bounds
        padded_bounds = (
            minx - overlap_m, miny - overlap_m,
            maxx + overlap_m, maxy + overlap_m,
        )

        dem_padded = self.dem_source.reproject(
            self.target_crs, padded_bounds, self.target_res
        ).astype(np.float32)
        dem_padded = self._pad(dem_padded, padded_size)

        if dem_padded.size == 0:
            return None

        # Compute all features on the padded array
        available = {'DEM': dem_padded}
        for name in self.layer_names:
            if name in available or name == 'LABELS':
                continue
            if name not in self.feature_registry:
                raise ValueError(f"Unknown feature: {name}")
            available[name] = self.feature_registry[name]['fn'](available)

        # Crop back to tile_size 
        s, e = overlap_px, overlap_px + tile_size
        available = {k: v[s:e, s:e] for k, v in available.items()}

        # Replace padded DEM with a clean unpadded read for the actual DEM channel
        dem_clean = self.dem_source.reproject(
            self.target_crs, tile_bounds, self.target_res
        ).astype(np.float32)
        available['DEM'] = self._pad(dem_clean, tile_size)

        if self.labelled:
            available['LABELS'] = self.label_shp.rasterise(
                tile_bounds, tile_size, self.target_crs
            )

        return np.stack([available[name] for name in self.layer_names], axis=0)

    @staticmethod
    def _pad(arr, tile_size):
        h, w = arr.shape
        if h == tile_size and w == tile_size:
            return arr
        out = np.zeros((tile_size, tile_size), dtype=arr.dtype)
        out[:min(h, tile_size), :min(w, tile_size)] = arr[:tile_size, :tile_size]
        return out

    def _save_tile(self, tile_data, tile_bounds, out_path, name):
        kwargs = dict(
            image=tile_data,
            tile_bounds=np.array(tile_bounds),
            layer_names=np.array(self.layer_names),
            res=np.array(self.target_res),
        )
        if self.labelled:
            labels = tile_data[self.layer_index['LABELS']].astype(np.uint8)
            kwargs.update(
                labels=labels,
                scd_present=np.array(bool(labels.sum() != 0)),
                scd_pixel_fraction=np.array(float(labels.sum()) / float(labels.size)),
                confirmed=np.array(True),
            )
        np.savez_compressed(str(Path(out_path) / name), **kwargs)

    def _random_tile_bounds(self, tile_size, rng):
        minx, miny, maxx, maxy = self.dem_source.bounds
        half = (tile_size * self.target_res) / 2
        cx = rng.uniform(minx + half, maxx - half)
        cy = rng.uniform(miny + half, maxy - half)
        return _centred_bounds(cx, cy, tile_size, self.target_res, self.target_bounds)

    # add a jitter so that model doesnt learn centroids are always in the middle. 
    def _grid_tile_bounds(self, tile_size, min_fraction=0.5):
        minx, miny, maxx, maxy = self.target_bounds
        tile_w = tile_size * self.target_res
        min_size = tile_w * min_fraction
        y = miny
        while y < maxy:
            x = minx
            while x < maxx:
                x2 = min(x + tile_w, maxx)
                y2 = min(y + tile_w, maxy)
                if (x2 - x) >= min_size and (y2 - y) >= min_size:
                    yield (x, y, x2, y2)
                x += tile_w
            y += tile_w

    def _empty_fraction(self, tile_data):
        fractions = []
        for name in self.layer_names:
            if name == 'LABELS':
                continue
            layer = tile_data[self.layer_index[name]]
            empty = (layer == 0) | np.isnan(layer)
            fractions.append(float(empty.sum()) / empty.size)
        return max(fractions) if fractions else 0.0

    def _is_mostly_empty(self, tile_data, threshold):
        return self._empty_fraction(tile_data) > threshold

    @staticmethod
    def _tile_name(tile_bounds, prefix='tile'):
        minx, miny, maxx, maxy = tile_bounds
        return f"{prefix}_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}"