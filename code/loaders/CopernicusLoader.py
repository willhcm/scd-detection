import math
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from osgeo import gdal
from pyproj import CRS

gdal.UseExceptions()

BUCKET = "copernicus-dem-30m"
NODATA = -9999


def tile_name(lat, lon):
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


class Cop30Loader:
    def __init__(self, tile_dir, workers=16):
        self.tile_dir = tile_dir
        self.workers = workers
        self.s3 = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=workers * 2)
        )
        self._have = None
        os.makedirs(tile_dir, exist_ok=True)

    def _listing(self):
        if self._have is None:
            body = self.s3.get_object(Bucket=BUCKET, Key="tileList.txt")["Body"]
            self._have = set(body.read().decode().split())
        return self._have

    def wanted(self, epsg, buffer_deg=1, lat_range=None):
        west, south, east, north = CRS.from_user_input(epsg).area_of_use.bounds
        if lat_range:
            south, north = max(south, lat_range[0]), min(north, lat_range[1])
        lons = range(math.floor(west) - buffer_deg, math.ceil(east) + buffer_deg)
        lats = range(math.floor(south), math.ceil(north))
        have = self._listing()
        names = [tile_name(la, (lo + 180) % 360 - 180) for la in lats for lo in lons]
        return [n for n in names if n in have]

    def _fetch(self, name):
        target = os.path.join(self.tile_dir, name + ".tif")
        if os.path.exists(target):
            return target
        for attempt in range(3):
            try:
                self.s3.download_file(BUCKET, f"{name}/{name}.tif", target)
                return target
            except Exception:
                if os.path.exists(target):
                    os.remove(target)
                if attempt == 2:
                    raise

    def download(self, names):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._fetch, names))

    def warp_zone(self, epsg, out_path, res=30, buffer_deg=1, lat_range=None):
        names = self.wanted(epsg, buffer_deg, lat_range)
        if not names:
            raise ValueError(f"no tiles for {epsg}")
        tiles = self.download(names)

        gdal.Warp(
            out_path,
            tiles,
            dstSRS=epsg,
            xRes=res,
            yRes=res,
            resampleAlg="cubic",
            srcNodata=NODATA,
            dstNodata=NODATA,
            outputType=gdal.GDT_Float32,
            multithread=True,
            warpMemoryLimit=4096,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
            creationOptions=[
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "COMPRESS=DEFLATE",
                "PREDICTOR=3",
                "BIGTIFF=IF_SAFER",
            ],
        )
        return out_path

loader = Cop30Loader("dem30/tiles")

for zone in range(38, 45):
    loader.warp_zone(f"EPSG:326{zone}", f"zones/z{zone}.tif", res=30, lat_range=(35, 56))