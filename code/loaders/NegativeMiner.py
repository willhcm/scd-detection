import numpy as np
from pathlib import Path
import sys
CODE_DIR = Path("../../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
from loaders.ScaleNormalisedDataStack import DataSource
from loaders.TileGen import TileGenerator
from dataclasses import dataclass

@dataclass
class ResBin:
    name: str
    res: float
    count: int

class NegativeMiner:

    def __init__(self, dem: DataSource, resolutions: list[float]):
        self.dem = dem
        self.resolutions = resolutions
        self.coverage = np.zeros(dem.data.shape, dtype=np.float32)
        self.generator = TileGenerator(dem=dem)
        self.report_capacity()

    def _pixel_window(self, tile_bounds):
        minx, miny, maxx, maxy = tile_bounds
        col0, row0 = ~self.dem.transform * (minx, maxy)
        col1, row1 = ~self.dem.transform * (maxx, miny)
        return int(row0), int(row1), int(col0), int(col1)

    def _update_coverage(self, tile_bounds):
        r0, r1, c0, c1 = self._pixel_window(tile_bounds)
        self.coverage[r0:r1, c0:c1] += 1.0

    def _is_tile_covered(self, tile_bounds):
        r0, r1, c0, c1 = self._pixel_window(tile_bounds)
        return np.any(self.coverage[r0:r1, c0:c1] > 0)

    def report_capacity(self):

        px_res = self.dem.res
        free_px = self.coverage.size
        free_area = free_px * px_res ** 2

        print(f"free area: {free_area:,.0f} m^2 ({free_px:,} px, "
            f"{100 * free_px / self.coverage.size:.1f}% of DEM)\n")

        self.bins = []
        for res in sorted(self.resolutions):
            tile_area = (512 * res) ** 2
            tile_count = int((free_area / tile_area) * 0.7)
            print(f"res {res:>5.2f}m  tiles = {tile_count:.1f}")

            self.bins.append(ResBin(name=f"{res:.2f}m", res=res, count=tile_count))

    def _grid_candidates(self, tile_side, jitter=0.15):
        minx, miny, maxx, maxy = self.dem.bounds
        xs = np.arange(minx, maxx - tile_side, tile_side)
        ys = np.arange(miny, maxy - tile_side, tile_side)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # jitter off-grid so tiles aren't all perfectly aligned - keeps some
        # placement diversity without falling back to pure random search
        coords += np.random.uniform(-tile_side * jitter, tile_side * jitter, coords.shape)
        np.random.shuffle(coords)
        return coords

    def _find_tile_positions(self):
        results = {}
        count_ceiling = min([b.count for b in self.bins])
        
        for b in sorted(self.bins, key=lambda b: -b.res):  # largest footprint first, least flexible
            tile_side = 512 * b.res
            placed = []

            for x0, y0 in self._grid_candidates(tile_side):
                if len(placed) >= b.count:
                    break
                if len(placed) >= count_ceiling:
                    break

                bounds = (x0, y0, x0 + tile_side, y0 + tile_side)
                if self._is_tile_covered(bounds):
                    continue
                self._update_coverage(bounds)
                placed.append(bounds)

            if len(placed) < b.count:
                print(f"{b.name}: placed {len(placed)}/{b.count}, candidates exhausted")
            results[b.name] = placed

        return results

    def generate_tiles(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        tile_positions = self._find_tile_positions()

        for res, bounds_list in tile_positions.items():
            res_str = f'{res}m'

            for i, bounds in enumerate(bounds_list):
                tile = self.generator._build_tile(bounds, 512, self.dem.res)
                np.savez_compressed(out_dir / f"res_{res_str}_tile_{i:05d}.npz", data=tile)

        self.report_statistics(tile_positions)

    def report_statistics(self, tile_positions):
        for res_name, bounds_list in tile_positions.items():
            print(f"{res_name}: {len(bounds_list)} tiles placed")

