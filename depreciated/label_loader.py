import os
os.environ["PROJ_DATA"] = "/opt/anaconda3/envs/IRP/share/proj"

from rasterio.features import rasterize
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../../helpers')
from helpers import read_meta

# will add into dataloader as a class method in time, just simple for now. 

def make_labels(SHAPE_PATH, OUTPUT_PATH, height, width, transform, profile):

    gdf = gpd.read_file(SHAPE_PATH)

    if gdf.crs != profile["crs"]:
        gdf = gdf.to_crs(profile["crs"])

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

    label_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )

    print("Unique values:", np.unique(label_raster))
    print("Positive pixels:", label_raster.sum())

    plt.imshow(label_raster)
    plt.show()

    with rio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(label_raster, 1)

def open_labels(OUTPUT_PATH):
    with rio.open(OUTPUT_PATH) as src:
        labels = src.read(1)
        meta = read_meta(src)
        return labels, meta

def get_labels(SHAPE_PATH, OUTPUT_PATH, meta):

    try:
        labels, meta = open_labels(OUTPUT_PATH)
        return (labels, meta)
    except:
        make_labels(SHAPE_PATH, OUTPUT_PATH, meta['height'], meta['width'], meta['transform'], meta['profile'])
        labels = open_labels(OUTPUT_PATH)
        return (labels, meta)