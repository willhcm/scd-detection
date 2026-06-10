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
from scipy.ndimage import uniform_filter


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
                # float64 not needed
                data = src.read(1).astype('float32')

                if src.nodata is not None:
                    data[data == src.nodata] = NODATA
                data[data < -9000] = NODATA 

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

def multidirectional_hillshade(dem, cell_size=1.0, altitude_deg=45.0, z_factor=1.0):
    """Calculates muldirectional hillshade for a DEM array"""
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



