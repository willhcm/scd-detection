from rasterio.features import rasterize
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt

# will add into dataloader as a class method in time, just simple for now. 

def make_labels(SHAPE_PATH, OUTPUT_PATH, height, width, transform, profile):

    gdf = gpd.read_file(SHAPE_PATH)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

    label_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0, 
        dtype="uint8",
        all_touched=False
    )

    plt.imshow(label_raster)

    with rio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(label_raster, 1)
