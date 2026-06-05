from rasterio.features import rasterize
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt

def make_labels(SHAPE_PATH, OUTPUT_PATH, height, width, transform, profile):

    gdf = gpd.read_file(SHAPE_PATH)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

    label_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,              # background
        dtype="uint8",
        all_touched=False    # set True if features are thin
    )

    plt.imshow(label_raster)

    with rio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(label_raster, 1)
