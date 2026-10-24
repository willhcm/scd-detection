import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from loaders.Tiler import _utm_epsg
import rasterio as rio
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

class WindowedDataSource:
    """
    A class to handle windowed reading and reprojection of raster data from a file.
    """

    def __init__(self, path, type="DEM", nominal_res=None, temp_path=None):
        self.path = path
        self.type = type

        # seperate temp path for reprojected raster, to be deleted at the end of the run.
        # will be the same as path when from_tiff class method is used, but saves deleting original raster if not.
        self.temp_path = temp_path

        # only read metadata here
        with rio.open(path) as src:
            self.crs = src.crs
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.count = src.count

            # set res
            if nominal_res is None:
                self.res = float(abs(src.res[0]))
            else:
                self.res = float(nominal_res)

    @classmethod
    def from_tiff(cls, path, type="DEM"):
        return cls(path=path, type=type)

    @classmethod
    def from_tiff_utm(cls, path, resolution, type="DEM"):

        with rio.open(path) as src:
            target_crs = CRS.from_epsg(_utm_epsg(src.crs, src.bounds))

        temp_path = cls.reproject_raster(src_path=path, dst_crs=target_crs, resolution=resolution)
        
        return cls(path=temp_path, type=type, nominal_res=resolution, temp_path=temp_path)

    # add to a helper .py later
    @staticmethod
    def reproject_raster(src_path, dst_crs, resolution=None):
        """
        Reproject a raster file to a new coordinate reference system (CRS).

        Parameters:
        - src_path: Path to the source raster file.
        - dst_crs: Target CRS in the format "EPSG:XXXX".

        Returns:
        - None
        """

        (REPO_ROOT / "temp").mkdir(exist_ok=True)
        fd, dst_path = tempfile.mkstemp(suffix=".tif", dir=REPO_ROOT / "temp")
        os.close(fd)

        with rio.open(src_path) as src:
            
            nodata = src.nodata if src.nodata is not None else -9999

            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                resolution=resolution)

            profile = src.profile.copy()
            profile.update(
                crs=dst_crs, transform=transform, width=width, height=height,
                dtype="float32", nodata=nodata,
                tiled=True, blockxsize=512, blockysize=512,
                compress="deflate", BIGTIFF="IF_SAFER")

            with rio.open(dst_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_nodata=nodata,
                        dst_nodata=nodata,
                        resampling=Resampling.cubic,
                        num_threads=4,
                    )

        return dst_path

    def reproject_to_shape(
        self,
        target_crs,
        tile_bounds,
        out_width,
        out_height,
        resampling=Resampling.cubic):

        dst_transform = from_bounds(*tile_bounds, out_width, out_height)

        with rio.open(self.path) as src:

            if self.type == "RGB":

                out = np.full((3, out_height, out_width), np.nan, dtype=np.float32)

                for band_index in range(3):

                    reproject(
                        source=rio.band(src, band_index + 1),
                        destination=out[band_index],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=resampling,
                        dst_nodata=np.nan)
            
            else:
                out = np.full((out_height, out_width), np.nan, dtype=np.float32)

                reproject(
                    source=rio.band(src, 1),
                    destination=out,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                    dst_nodata=np.nan)

        return out