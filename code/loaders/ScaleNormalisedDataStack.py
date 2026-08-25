import numpy as np
import rasterio as rio
import geopandas as gpd
from pathlib import Path
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.crs import CRS
from rasterio.transform import from_bounds, array_bounds 
from shapely.geometry import box
from scipy.ndimage import sobel, gaussian_filter, laplace
from pyproj import Transformer
from rasterio.features import rasterize

OVERLAP_SIGMA_MULTIPLIER = 3


# AI Statement:
# I have used ChatGPT to help with portions of this code, specifically:
# 1. Grouping functionality e.g. centroid distance, UnionFind class. 
# 2. adapting datastack and build_tile to take more generalised requests (in terms of res), instead of being used at fixed-res
# 3. editing objects ShapeLabels Class to match the two above changes.

# ChatGPT assistance with gdf functionality.
class ShapeLabels:
    def __init__(self, path):
        self.gdf = gpd.read_file(path)
        self.gdf = self.gdf[self.gdf.geometry.notnull()].copy()
        self.gdf = self.gdf[~self.gdf.geometry.is_empty].copy()

    # get all objects from labels geodataframe. return dict of properties per scd label.
    def objects(self, target_crs):
        """Return label geometries and metadata in target CRS"""
        gdf = self.gdf.to_crs(target_crs).copy()
        objs = []
        for idx, geom in enumerate(gdf.geometry):
            if geom is None or geom.is_empty:
                continue
            area = float(geom.area)
            if area <= 0:
                continue
            minx, miny, maxx, maxy = geom.bounds
            bbox_w = maxx - minx
            bbox_h = maxy - miny
            equiv_d = 2.0 * np.sqrt(area / np.pi)
            diameter = max(equiv_d, bbox_w, bbox_h)

            # return gives geom (the key result, and other metadata used in grouping and tiling.)
            objs.append({
                "id": idx,
                "geom": geom,
                "cx": geom.centroid.x,
                "cy": geom.centroid.y,
                "area": area,
                "diameter": float(diameter),
                "bounds": (minx, miny, maxx, maxy),
            })
        return objs

    # rasterise labels within a given tile bounds (model needs images for masks for training!)
    def rasterise(self, tile_bounds, tile_size, target_crs):
        # rasterises labels from .shp (vector shapefile from hand labelling in QGIS.)
        minx, miny, maxx, maxy = tile_bounds
        transform = from_bounds(minx, miny, maxx, maxy, tile_size, tile_size)

        gdf = self.gdf.to_crs(target_crs)
        clipped = gdf[gdf.intersects(box(minx, miny, maxx, maxy))]

        if clipped.empty:
            return np.zeros((tile_size, tile_size), dtype=np.uint8)

        shapes = [(geom, 1) for geom in clipped.geometry if geom is not None and not geom.is_empty]
        return rasterize(
            shapes=shapes,
            out_shape=(tile_size, tile_size),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
# ChatGPT assistance with this function too!
# convertion to UTM from global DD
def _utm_epsg(src_crs, bounds):
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    minx, miny, maxx, maxy = bounds
    lon = (minx + maxx) / 2
    lat = (miny + maxy) / 2
    lon_wgs, lat_wgs = transformer.transform(lon, lat)
    zone = int((lon_wgs + 180) / 6) + 1
    return 32600 + zone if lat_wgs >= 0 else 32700 + zone

# same as previously, except reprojection function changes to use variables resolution instead of a fixed target_res throughout.
# again, for SNIP approach and scale normalisation.
class DataSource:
    def __init__(self, type, data, res, crs, bounds, width, height, transform):
        self.type = type
        self.data = data
        self.res = float(res)
        self.crs = crs
        self.bounds = tuple(bounds)
        self.width = width
        self.height = height
        self.transform = transform

    def reproject_to_shape(self, target_crs, tile_bounds, out_width, out_height, resampling=Resampling.cubic):
        minx, miny, maxx, maxy = tile_bounds

        transform = from_bounds(
            minx,
            miny,
            maxx,
            maxy,
            out_width,
            out_height,
        )

        if self.data.ndim == 2:
            out = np.empty(
                (out_height, out_width),
                dtype=np.float32,
            )
        else:
            out = np.empty(
                (self.data.shape[0], out_height, out_width),
                dtype=np.float32,
            )

        reproject(
            source=self.data,
            destination=out,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=resampling,
        )
        return out

    # handles both rgb and dem. changed from prior DEM only functionality before to work for veto classifier too.
    @classmethod
    def from_tiff(cls, path, type="DEM"):
        with rio.open(path) as src:
            if type == "DEM":
                data = src.read(1).astype(np.float32)
            elif type == "RGB":
                data = src.read([1, 2, 3]).astype(np.float32)
            # class method
            return cls(
                type=type,
                data=data,
                res=src.res[0],
                crs=src.crs,
                bounds=src.bounds,
                width=src.width,
                height=src.height,
                transform=src.transform,
            )

    # AI rewrite from taking only DEM to accepting RGB as needed for VetoDataset.py
    @classmethod
    def from_tiff_utm(cls, path, native_res, type="DEM"):
    
        with rio.open(path) as src:
            utm_crs = CRS.from_epsg(_utm_epsg(src.crs, src.bounds))

            # transform to UTM
            transform, width, height = calculate_default_transform(
                src.crs,
                utm_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=native_res)

            if type == "DEM":
                data = np.empty((height, width),dtype=np.float32)

                reproject(
                    source=rio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.cubic
                )

            elif type == "RGB":
                if src.count < 3:
                    # just to catch incorrect passing of type 'DEM'
                    raise ValueError(
                        f"RGB raster requires at least 3 bands, "
                        f"but {path} has {src.count}."
                    )

                data = np.empty((3, height, width), dtype=np.float32)

                for band_index in range(3):
                    reproject(
                        source=rio.band(src, band_index + 1),
                        destination=data[band_index],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=utm_crs,
                        resampling=Resampling.bilinear)

        bounds = array_bounds(height, width,transform)

        return cls(
            type=type,
            data=data,
            res=native_res,
            crs=utm_crs,
            bounds=bounds,
            width=width,
            height=height,
            transform=transform,
        )
 

# DEM-derived feature helpers
def _slope(a):
    sx = sobel(a["DEM"], axis=0)
    sy = sobel(a["DEM"], axis=1)
    return np.sqrt(sx ** 2 + sy ** 2).astype(np.float32)


# multi-directional hillshade. returns a function for registry
def _hillshade(cell_size=1.0, altitude_deg=45.0, z_factor=1.0):
    def _fn(a):
        dem = a["DEM"]
        alt = np.radians(altitude_deg)
        dz_dx = np.gradient(dem * z_factor, cell_size, axis=1)
        dz_dy = np.gradient(dem * z_factor, cell_size, axis=0)
        slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
        aspect = np.arctan2(-dz_dy, dz_dx)
        hs = np.zeros_like(dem, dtype=np.float64)
        for az_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            az = np.radians(360 - az_deg + 90)
            hs += np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - aspect)
        return np.clip(hs / 8, 0, 1).astype(np.float32)
    return _fn

# residual relief
def _make_rr(sigma_px, tile_res):

    def _fn(a):
        dem = a["DEM"].astype(np.float64)
        smooth = gaussian_filter(dem, sigma=sigma_px)
        return (dem - smooth).astype(np.float32)

    return _fn

# curvature
def _laplace(a):
    return laplace(a["DEM"]).astype(np.float32)


def _build_feature_registry(sigma_px, cell_size, tile_res):
    return {
        "DEM_SLOPE": {"deps": ["DEM"], "fn": _slope},
        "HILLSHADE": {"deps": ["DEM"], "fn": _hillshade(cell_size=cell_size)},
        "RR": {"deps": ["DEM"], "fn": _make_rr(sigma_px, tile_res)},
        "LAPLACE": {"deps": ["DEM"], "fn": _laplace},
    }

# ChatGPT assistance with grouping functionality, including the following 6 functions and UnionFind class.

# used for clustering of SCDs to generate tiles !!
# get bounds of a tile from its centre and resolution 
def _centred_bounds(cx, cy, tile_size, res):
    half = (tile_size * res) / 2.0
    return (cx - half, cy - half, cx + half, cy + half)

# check if one bounds is fully inside another
def _bounds_inside(inner, outer):
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 and iy0 >= oy0 and ix1 <= ox1 and iy1 <= oy1

# return bounds that cover different objects / bounds (for composite tiles)
def _combined_bounds(objs):
    b = np.array([o["bounds"] for o in objs], dtype=float)
    return (b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max())

# get centroid of a group of objects, weighted by area of scd
def _area_weighted_centroid(objs):
    areas = np.array([max(o["area"], 1e-6) for o in objs], dtype=float)
    xs = np.array([o["cx"] for o in objs], dtype=float)
    ys = np.array([o["cy"] for o in objs], dtype=float)
    return float((xs * areas).sum() / areas.sum()), float((ys * areas).sum() / areas.sum())

# get diamater of a group of scd labels
def _group_diameter(objs):
    minx, miny, maxx, maxy = _combined_bounds(objs)
    bbox_d = max(maxx - minx, maxy - miny)
    max_obj_d = max(o["diameter"] for o in objs)
    return float(max(bbox_d, max_obj_d))

# unionfind class 
# ChatGPT assistance with this class, used for grouping of SCDs to generate tiles.
class UnionFind:
    def __init__(self, n):
        # each object belongs to its own cluster initially, so parent is itself.
        self.parent = list(range(n))

    # find root of cluster containing object x.
    def find(self, x):
        # path compression: make each node on the path point directly to the root. dont care abt other intermediate nodes.
        while self.parent[x] != x:
            # parent of x becomes parent of its parent.
            self.parent[x] = self.parent[self.parent[x]]
            # x becomes its parent, moving up the tree.
            x = self.parent[x]
        # returns the root of the cluster containing x.
        return x

    # joins two clusters containing objects a and b.
    def union(self, a, b):
        # find the roots of the clusters containing a and b
        ra, rb = self.find(a), self.find(b)
        # join the clusters by making the root of one point to the root of the other. arbitary which one becomes the parent.
        if ra != rb:
            self.parent[rb] = ra

    # turns the parent list into a list of groups, where each group is a list of object indices that belong to the same cluster.
    def groups(self):
        out = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return list(out.values())
    
# smart tile bounds moving to avoid edge artefacts. 
# moves up, down, left, right depending on where the edge artefacts are.
def _edge_shift(label, tile_bounds, res, shift_px=32):

    minx, miny, maxx, maxy = tile_bounds

    dx = 0.0
    dy = 0.0

    # top
    if label[0].any():
        dy += shift_px * res

    # bottom
    if label[-1].any():
        dy -= shift_px * res

    # left
    if label[:, 0].any():
        dx += shift_px * res

    # right
    if label[:, -1].any():
        dx -= shift_px * res

    return (
        minx + dx,
        miny + dy,
        maxx + dx,
        maxy + dy,
    )


# GenAI assistance with 'request' format. The previous datastack.py was purpose-built for fixed res exportation,
# but that didnt fit the new SNIP approach I am attempting (and confidenw with), so i got assistance from ChatGPT on how to 
# pivot to variable res, which required some architectural change as opposed to just changing functions which i have done before.
class ScaleNormalisedDataStack:
    """ SNIP DataStack V1 

    No longer needs any unlabelled functionality. inference done using image pyramid.
    Just for scale normalised (and when appropriate, grouped) tile exportation.

    """
    DEFAULT_LAYERS = ["DEM", "DEM_SLOPE", "HILLSHADE", "RR", "LAPLACE"]

    # keep features functionality if want to change anything for experimentation.
    def __init__(self, dem_source: DataSource, label_shp: ShapeLabels = None,
                 features=None):
        
        self.dem_source = dem_source
        self.label_shp = label_shp
        self.target_crs = dem_source.crs
        self.dem_bounds = dem_source.bounds

        # Residual-relief smoothing scale is physical, not pixels, so it stays
        # comparable across variable-resolution tiles.
        # needs to depend on res
        # manual is probably overkill, just automatic 10 * res. can change later if needed.
        self.rr_sigma_m = 12.0 * dem_source.res

        self.layer_names = features or self.DEFAULT_LAYERS.copy()
        self.layer_names.append("LABELS")
        self.layer_index = {name: i for i, name in enumerate(self.layer_names)}

    # generates and exports tiles. 
    def tile_and_export(
        self,
        tile_size: int,
        out_path: str,
        obj_frac_range=(0.3, 0.55),
        negative: float = 0.15,
        seed: int = 42,
        empty_threshold: float = 0.2,
        min_res: float = None,
        max_res: float = None,
        cluster: bool = False,
        cluster_distance_m: float = None,
        cluster_factor: float = 0.75,
        jitter_frac: float = 0.10,
        edge_tolerance: float = 0.10,
        skip_edge_tiles: bool = True,
    ):
        """
        Export one scale-normalised positive tile per labelled object, or per
        conservative object group if cluster=True.

        Grouping behaviour:
        - cluster=False: one tile per individual SCD.
        - cluster=True and cluster_distance_m is provided: group objects whose
          centroids are within cluster_distance_m.
        - cluster=True and cluster_distance_m is None: group objects only when
          centroid distance <= cluster_factor * max(object diameters).

        """

        Path(out_path).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)

        objs = self.label_shp.objects(self.target_crs)
        print(f"Found {len(objs)} labelled objects")

        # if cluser, nearby SCDs are put into one tile to avoid multiple tiles of the same area (data leakage risk)
        # important as at inference, objects wont always be 'alone'
        # objects clustered using UnionFind.
        if cluster:
            groups = self._cluster_objects(
                objs,
                cluster_factor=cluster_factor,
                cluster_distance_m=cluster_distance_m,
            )
        # if not, all SCDs are treated as indiviiual tiles
        else:
            groups = [[o] for o in objs]

        # 
        n_grouped = sum(1 for g in groups if len(g) > 1)
        print(f"Using {len(groups)} object groups ({n_grouped} grouped, {len(groups) - n_grouped} individual)")

        exported_pos = skipped = 0
        positive_resolutions = []

        MAX_ATTEMPTS = 10

        # for group in cluster, make request around bounds and export tile if valid (several validity params)
        for group_id, group in enumerate(groups):

            success = False

            # initial request
            req = None

            for attempt in range(MAX_ATTEMPTS):

                if req is None:

                    # make a new request for a tile that covers the group, with a resolution that is scale-normalised to the group size
                    # and the target fraction of the tile that should be covered by the group.
                    req = self._make_scale_normalised_request(
                        group=group,
                        tile_size=tile_size,
                        rng=rng,
                        obj_frac_range=obj_frac_range,
                        min_res=min_res,
                        max_res=max_res,
                        jitter_frac=jitter_frac, # edge_shift is now solving same problem, so remove dynamic jf.
                        skip_edge_tiles=skip_edge_tiles,
                    )

                    if req is None:
                        continue

                # generate tile data from DataSources for specified bounds and res
                tile = self._build_tile(
                    req["bounds"],
                    tile_size,
                    req["res"],
                    skip_edge_tiles=skip_edge_tiles,
                )

                # if built tile has empty data, skip. 
                if tile is None or self._is_mostly_empty(tile, empty_threshold):
                    req = None
                    if tile is not None:
                        del tile
                    continue

                # get label
                label = tile[self.layer_index["LABELS"]]

                # if scd coverage is too low or too high, skip.
                coverage = float(label.sum()) / label.size

                if coverage <= 0 or coverage > 0.6:
                    req = None
                    del tile
                    continue

                # handle edge artefacts. if too many edge artefacts, move tile away from edge and try again.
                edge_frac = self._edge_label_fraction(label)

                if edge_frac <= edge_tolerance:

                    name = self._tile_name(req["bounds"], f"pos_g{group_id:04d}")

                    self._save_tile(
                        tile,
                        req,
                        out_path,
                        name,
                        positive=True,
                    )

                    exported_pos += 1
                    positive_resolutions.append(req["res"])

                    success = True
                    del tile
                    break

                # Failed edge tolerance
                # Move tile away from offending edge and try again

                req["bounds"] = _edge_shift(
                    label,
                    req["bounds"],
                    req["res"],
                    shift_px=32,
                )

                del tile

            if not success:
                skipped += 1

        n_neg = max(1, int(exported_pos * negative))
        print(f"Sampling {n_neg} negative tiles")

        # generate negatives as specified from negative_frac
        exported_neg = attempts = 0
        while exported_neg < n_neg and attempts < n_neg * 100:
            attempts += 1
            if positive_resolutions:
                # chooses a similar resolution for negative to whats already been chosen for positive tiles
                res = float(rng.choice(positive_resolutions))
            else:
                res = float(self.dem_source.res)

            req = self._random_negative_request(tile_size, res, rng)
            if req is None:
                continue

            tile = self._build_tile(req["bounds"], tile_size, req["res"], skip_edge_tiles=skip_edge_tiles)
            if tile is None or self._is_mostly_empty(tile, empty_threshold):
                continue

            if tile[self.layer_index["LABELS"]].sum() > 0:
                del tile
                continue

            name = self._tile_name(req["bounds"], "neg")
            self._save_tile(tile, req, out_path, name, positive=False)
            exported_neg += 1
            del tile

        print(f"Done. {exported_pos} positive, {exported_neg} negative, skipped {skipped}.")

    # ChatGPT assistance with grouping functionality and UnionFind.
    def _cluster_objects(self, objs, cluster_factor=0.75, cluster_distance_m=None):
        """
        Conservative graph clustering. Objects are joined only if their
        centroids are close enough.

        If cluster_distance_m is provided, it is used as the fixed maximum
        centroid distance for grouping. Otherwise, each pair uses:

            distance <= cluster_factor * max(diameter_i, diameter_j)

        With the default cluster_factor=0.75, most isolated SCDs remain
        individual tiles; only close neighbours are grouped.
        """
        n = len(objs)
        # custom UnionFind class to group objects based on centroid distance
        uf = UnionFind(n)

        # taken outsidew of loop for optimisation 
        if cluster_distance_m is not None:
            thresh = float(cluster_distance_m)
        else:
            thresh = float(cluster_factor) * max(objs[i]["diameter"], objs[j]["diameter"])

        # compare every pair of objects and join them if they are close enough based on the criteria above.
        # probably a faster way to do this, but this isnt the bottleneck overall and the code is only run once.
        for i in range(n):
            for j in range(i + 1, n):
                # compute distance between centroids of two objects
                dx = objs[i]["cx"] - objs[j]["cx"]
                dy = objs[i]["cy"] - objs[j]["cy"]
                dist = np.sqrt(dx * dx + dy * dy)

                # join if close enough
                if dist <= thresh:
                    uf.union(i, j)

        # return groups of objects based on the UnionFind structure, where each group is a list of objects that are nearby
        return [[objs[i] for i in inds] for inds in uf.groups()]

    def _make_scale_normalised_request(
        self,
        group,
        tile_size,
        rng,
        obj_frac_range,
        min_res,
        max_res,
        jitter_frac,
        skip_edge_tiles):

        group_d = _group_diameter(group)
        target_frac = float(rng.uniform(*obj_frac_range))

        # group_d / tile_width_m = target_frac => tile_width_m = group_d / target_frac
        tile_width_m = group_d / max(target_frac, 1e-6)
        res = tile_width_m / tile_size

        if min_res is not None:
            res = max(float(min_res), res)
        if max_res is not None:
            res = min(float(max_res), res)
        tile_width_m = res * tile_size

        # ChatGPT assistance with grouping functionality.
        cx, cy = _area_weighted_centroid(group)
        gx0, gy0, gx1, gy1 = _combined_bounds(group)

        # Jitter tile centre while trying to keep the whole group inside the tile.
        half = tile_width_m / 2.0
        margin = 0.02 * tile_width_m
        min_cx = gx1 + margin - half
        max_cx = gx0 - margin + half
        min_cy = gy1 + margin - half
        max_cy = gy0 - margin + half

        if min_cx <= max_cx:
            centre_x = rng.uniform(min_cx, max_cx)
        else:
            centre_x = cx

        if min_cy <= max_cy:
            centre_y = rng.uniform(min_cy, max_cy)
        else:
            centre_y = cy

        # Small jitter so model doesnt learn that SCDs are always in centre of tile shortcut.
        max_jit = half * jitter_frac
        centre_x += rng.uniform(-max_jit, max_jit)
        centre_y += rng.uniform(-max_jit, max_jit)
        if min_cx <= max_cx:
            centre_x = float(np.clip(centre_x, min_cx, max_cx))
        if min_cy <= max_cy:
            centre_y = float(np.clip(centre_y, min_cy, max_cy))

        bounds = _centred_bounds(centre_x, centre_y, tile_size, res)

        # handles data source edge tiles (cant zero pad instead, doesnt work)
        if skip_edge_tiles and not _bounds_inside(bounds, self.dem_bounds):
            return None

        # request information
        return {
            "bounds": bounds,
            "res": float(res),
            "tile_size": int(tile_size),
            "group_bool": [1 if len(group) > 1 else 0],
            "object_ids": np.array([o["id"] for o in group], dtype=np.int32),
        }

    def _random_negative_request(self, tile_size, res, rng):
        tile_w = tile_size * res
        minx, miny, maxx, maxy = self.dem_bounds
        if (maxx - minx) <= tile_w or (maxy - miny) <= tile_w:
            return None
        x0 = rng.uniform(minx, maxx - tile_w)
        y0 = rng.uniform(miny, maxy - tile_w)
        bounds = (x0, y0, x0 + tile_w, y0 + tile_w)

        # request information
        return {
            "bounds": bounds,
            "res": float(res),
            "tile_size": int(tile_size),
            "object_ids": np.array(np.nan)
        }

    def _build_tile(self, tile_bounds, tile_size, target_res, skip_edge_tiles=True):
        # Compute RR/slope/Laplace on a larger real-context window, then crop.
        sigma_px = max(1.0, self.rr_sigma_m / target_res)
        overlap_px = max(4, int(round(OVERLAP_SIGMA_MULTIPLIER * sigma_px)))
        padded_size = tile_size + 2 * overlap_px
        overlap_m = overlap_px * target_res

        minx, miny, maxx, maxy = tile_bounds
        padded_bounds = (
            minx - overlap_m,
            miny - overlap_m,
            maxx + overlap_m,
            maxy + overlap_m,
        )

        # handles tile being on edge of DEM (bad as zero padding ruins SCD context)
        if skip_edge_tiles and not _bounds_inside(padded_bounds, self.dem_bounds):
            return None

        # large DEM to reduce edge artefacts of the tile
        dem_padded = self.dem_source.reproject_to_shape(
            self.target_crs, padded_bounds, padded_size, padded_size, resampling=Resampling.cubic
        ).astype(np.float32)

        available = {"DEM": dem_padded}
        registry = _build_feature_registry(sigma_px=sigma_px, cell_size=target_res, tile_res=target_res)

        # build available from registry
        for name in self.layer_names:
            if name in available or name == 'LABELS':
                continue
            available[name] = registry[name]['fn'](available)

        s = overlap_px
        e = overlap_px + tile_size
        available = {k: v[s:e, s:e] for k, v in available.items()}

        # Clean DEM channel without overlap. This keeps DEM itself unfiltered while
        # derived features benefited from context.
        dem_clean = self.dem_source.reproject_to_shape(
            self.target_crs, tile_bounds, tile_size, tile_size, resampling=Resampling.cubic
        ).astype(np.float32)
        available["DEM"] = dem_clean

        available["LABELS"] = self.label_shp.rasterise(tile_bounds, tile_size, self.target_crs)

        return np.stack([available[name] for name in self.layer_names], axis=0).astype(np.float32)


    def _save_tile(self, tile_data, req, out_path, name, positive=True):
        
        # no longer need to worry about self.labelled, as this will not be used to produce deployment tiles!
        # that will be done using image pyramid.
        labels = tile_data[self.layer_index["LABELS"]].astype(np.uint8)
        image = tile_data[:self.layer_index["LABELS"]].astype(np.float32)

        np.savez_compressed(str(Path(out_path) / name),
                            image=image, 
                            layer_names=np.array(self.layer_names),
                            res=np.array(req["res"], dtype=np.float32),
                            labels=labels,
                            scd_pixel_fraction=np.array(float(labels.sum()) / float(labels.size)),
                            object_ids = np.array(req.get("object_ids", [])),
                            positive=np.array(positive))


    # how much of the tile edges have label=1
    def _edge_label_fraction(self, label):
        top = label[0, :].sum() / label.shape[1]
        bottom = label[-1, :].sum() / label.shape[1]
        left = label[:, 0].sum() / label.shape[0]
        right = label[:, -1].sum() / label.shape[0]
        return float(max(top, bottom, left, right))

    def _empty_fraction(self, tile_data):
        fractions = []
        for name in self.layer_names:
            if name == "LABELS":
                continue
            layer = tile_data[self.layer_index[name]]
            empty = (layer == 0) | np.isnan(layer)
            fractions.append(float(empty.sum()) / empty.size)
        return max(fractions) if fractions else 0.0

    def _is_mostly_empty(self, tile_data, threshold):
        return self._empty_fraction(tile_data) > threshold

    @staticmethod
    def _tile_name(tile_bounds, prefix="tile"):
        minx, miny, maxx, maxy = tile_bounds
        return f"{prefix}_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}"