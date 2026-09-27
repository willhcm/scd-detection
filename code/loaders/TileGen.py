# creates a tile from centre coordinates (DMS) and inputted rasters (DEM, labels: optional)

import sys
from pathlib import Path
CODE_DIR = Path("../../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
from loaders.ScaleNormalisedDataStack import DataSource, ShapeLabels, _bounds_inside, _build_feature_registry, OVERLAP_SIGMA_MULTIPLIER, _centred_bounds
import numpy as np
from rasterio.warp import Resampling
from pathlib import Path 
import matplotlib.pyplot as plt

class TileGenerator():

    def __init__(self, dem, label_path=None):
        self.dem = dem 
        self.target_crs = self.dem.crs
        self.sigma_px = 12
        self.layer_names = ["DEM", "DEM_SLOPE", "HILLSHADE", "RR", "LAPLACE"]

        if label_path is not None:
            self.labelled = True
            self.labels = ShapeLabels(label_path)
        else:
            self.labelled = False
        
        self.tiles = []

    def _build_tile(self, tile_bounds, tile_size, target_res):
        # Compute RR/slope/Laplace on a larger real-context window, then crop.
        
        overlap_px = max(4, int(round(OVERLAP_SIGMA_MULTIPLIER * self.sigma_px)))
        padded_size = tile_size + 2 * overlap_px
        overlap_m = overlap_px * target_res

        minx, miny, maxx, maxy = tile_bounds
        padded_bounds = (
            minx - overlap_m,
            miny - overlap_m,
            maxx + overlap_m,
            maxy + overlap_m,
        )

        # large DEM to reduce edge artefacts of the tile
        dem_padded = self.dem.reproject_to_shape(
            self.target_crs, padded_bounds, padded_size, padded_size, resampling=Resampling.cubic
        ).astype(np.float32)

        available = {"DEM": dem_padded}
        registry = _build_feature_registry(sigma_px=self.sigma_px, cell_size=target_res, tile_res=target_res)

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
        dem_clean = self.dem.reproject_to_shape(
            self.target_crs, tile_bounds, tile_size, tile_size, resampling=Resampling.cubic
        ).astype(np.float32)
        available["DEM"] = dem_clean

        if self.labelled:
            available["LABELS"] = self.labels.rasterise(tile_bounds, tile_size, self.target_crs)
        else:
            available["LABELS"] = np.zeros((tile_size, tile_size))

        return np.stack([available[name] for name in self.layer_names], axis=0).astype(np.float32)

    @staticmethod
    def _tile_name(tile_bounds, prefix="tile"):
        minx, miny, maxx, maxy = tile_bounds
        return f"{prefix}_{int(minx)}_{int(miny)}_{int(maxx)}_{int(maxy)}"


    def export(self, out_path):

        for tile in self.tiles:
            labels = tile['data'][self.layer_index["LABELS"]].astype(np.uint8)
            image = tile['data'][:self.layer_index["LABELS"]].astype(np.float32)

            np.savez_compressed(str(Path(out_path) / tile['name']),
                    image=image, 
                    layer_names=np.array(self.layer_names),
                    res=np.array(tile["res"], dtype=np.float32),
                    layer_index = self.layer_names,
                    labels=labels,
                    scd_pixel_fraction=np.array(float(labels.sum()) / float(labels.size)),
                    bounds=np.array(tile['bounds'], dtype=np.float32))


    def generate_tile(self, centre, res):
        x, y = centre
        bounds = _centred_bounds(x, y, 512, res)
        tile = self._build_tile(bounds, 512, res)

        tile_dict = {'data': tile,
                     'name': self._tile_name(bounds),
                     'bounds': bounds,
                     'positive': 0 if self.tile[self.layer_index["LABELS"]].sum() == 0 else 1,
                     'res': res}
        self.tiles.append(tile_dict)
        self.plot_tile(tile_dict)

    def plot_tile(self, tile):

        fig, axs = plt.subplots(1, len(self.layer_names), figsize=(15, 5))
        for i, name in enumerate(self.layer_names):
            axs[i].imshow(tile['data'][i], cmap='gray')
            axs[i].set_title(name)
            axs[i].axis('off')
        plt.show()

