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


def merge_tifs(dir_path: list, out_path: str, file_name: str):
    '''
    add docstring
    '''

    if not isinstance(out_path, PosixPath):
        try:
            out_path = Path(out_path)
        except:
            raise TypeError('out_path must be type PosixPath or string!')


    dem_files = glob.glob(dir_path)

    out_path.mkdir(parents=True, exist_ok=True)

    NODATA = -9999
    clipped_paths = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in dem_files:
            out_name = Path(f).stem + '_clean.tif'
            out_path = os.path.join(tmpdir, out_name)

            with rasterio.open(f) as src:
                data = src.read(1).astype('float32')

                if src.nodata is not None:
                    data[data == src.nodata] = NODATA

                data[data < -9000] = NODATA

                profile = src.profile.copy()
                profile.update(dtype='float32', nodata=NODATA)

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data, 1)

            clipped_paths.append(out_path)

        # Merge
        datasets = [rasterio.open(p) for p in clipped_paths]
        mosaic, transform = merge(datasets, nodata=NODATA, method='max')

        profile = datasets[0].profile
        profile.update(
            width=mosaic.shape[2],
            height=mosaic.shape[1],
            transform=transform,
            dtype='float32',
            nodata=NODATA,
        )

        with rasterio.open(out_path / file_name, 'w', **profile) as dst:
            dst.write(mosaic)

        for ds in datasets:
            ds.close()

    print("Saved")
        




