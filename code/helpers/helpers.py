# Helper functions to use throughout notebooks.

import rasterio as rio 
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import glob
import geopandas as gpd
from pathlib import Path, PosixPath
import tempfile, os
import numpy as np
import rasterio
from rasterio.merge import merge
from pathlib import Path
import glob
from scipy.ndimage import uniform_filter, grey_opening
from skimage.morphology import disk


def plot_coverage(dirs: list):
    '''Plots a coverage map from a dictionary of tifs. Takes a few minutes to run, as it needs to open all tifs.
    
    Parameters
    --------------

    dirs: list
        list of directories to search in

    '''

    paths = set()
    bounds = []

    for dir in dirs:
        files = glob.glob(dir)
        for p in files:
            try:
                with rio.open(p) as src:
                    if p not in paths:
                        paths.update(p)
                        bounds.append(src.bounds)
            except:
                continue


    rects = [box(b.left, b.bottom, b.right, b.top) for b in bounds]
    coverage = unary_union(rects)

    total_extent = box(*coverage.bounds)
    gaps = total_extent.difference(coverage)

    _, ax = plt.subplots()
    gpd.GeoSeries([coverage]).plot(ax=ax, color='green', alpha=0.5, label='coverage')
    gpd.GeoSeries([gaps]).plot(ax=ax, color='red', alpha=0.5, label='gaps')
    ax.legend()
    plt.show()
    
def get_tifs(dir_path):
    file_names = set()
    paths = []
    duplicates = []

    files = glob.glob(dir_path)
    for p in files:
        with rasterio.open(p) as src:
            file_ = src.name[-17:-4]
            if file_ not in file_names:
                file_names.update([file_])
                paths.append(p)
            else:
                duplicates.append(p)

    return paths, file_names, duplicates


# assistance with tempfile from ChatGPT / claude
def merge_dems(dem_paths):
    NODATA = -9999
    cleaned_srcs = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for f_path in dem_paths:
            out_name = Path(f_path).stem + '_clean.tif'
            out_path = os.path.join(tmpdir, out_name)

            with rasterio.open(f_path) as src:
                # float32 ro save sapce 
                data = src.read(1).astype('float32')

                if src.nodata is not None:
                    data[data == src.nodata] = NODATA
                data[data < -9000] = NODATA # clean sea-level nodata which is often like -1e38 or something in DEFRA data.

                profile = src.profile.copy()
                profile.update(dtype='float32', nodata=NODATA, count=1)

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data, 1)

            # Open the cleaned temporary file for merging
            cleaned_srcs.append(rasterio.open(out_path))

        # merge
        mosaic, transform = merge(cleaned_srcs, nodata=NODATA, method='max')

        # Update metadata based on the merged mosaic
        # arbitrary choosing first source and editing to match new merged DEM
        meta = cleaned_srcs[0].profile.copy()
        meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "count": 1,
            "dtype": 'float32',
            "nodata": NODATA
        })

        # Close all opened cleaned source datasets
        for src in cleaned_srcs:
            src.close()

    # Return the merged data and the updated metadata
    return mosaic[0], meta
        

def tpi(dem, r):
  """Topographic Position Index (with radius r)"""
  neighbourhood_mean = uniform_filter(
        dem,
        size=r,
        mode="nearest")

  return dem - neighbourhood_mean

def read_meta(src):
    meta = {'crs': src.crs,
            'res': src.res,
            'bounds': src.bounds,
            'height': src.height,
            'profile': src.profile,
            'width': src.width,
            'transform': src.transform}
    
    return meta

def dem_ground(dem, radius=15):
    return grey_opening(dem, footprint=disk(radius)).astype(np.float32)

