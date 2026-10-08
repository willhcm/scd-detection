import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from deployment.PostProcesser import WindowedDataSource
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from ScaleNormalisedDataStack import ShapeLabels, DataSource
from IPython.display import clear_output
import numpy as np
import ipywidgets as widgets

from pathlib import Path
import sys

CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from VetoHelpers import build_dem_context, VETO_SCALAR_NAMES

class DEMVetoLoader:

    def __init__(
        self,
        shp_path,
        dem_path,
        target_crs = None,
        context_scale=4.0,
        min_width_m=750.0,
        max_width_m=4000.0,
        tile_size=224):

        self.predictions = ShapeLabels(shp_path)
        self.dem_source = DataSource.from_tiff(
            dem_path,
            type="DEM")
        self.dem_res = self.dem_source.res
        self.target_crs = target_crs
        self.tiles = []
        self.tile_size = tile_size
        self.context_scale = context_scale
        self.min_width_m = min_width_m
        self.max_width_m = max_width_m

    # build tile from candidate object, including local RGB and DEM context
    def _make_tile(self, obj, tile_size=96, min_crop_pixels=64, object_fraction=0.5):

        # centroid 
        cx = obj["cx"]
        cy = obj["cy"]

        # object size
        diameter_m = (obj.equivalent_diameter_area * 1)
        object_crop_width_m = (diameter_m / object_fraction)

        # DEM crop width
        # Four-times object box width, clipped to fixed limits.
        context_width_m = float(
            np.clip(
                self.context_scale * object_crop_width_m,
                self.min_width_m,
                self.max_width_m,
            )
        )

        # build bbox around centre
        context_half_width = context_width_m / 2.0
        context_bounds = (
            cx - context_half_width,
            cy - context_half_width,
            cx + context_half_width,
            cy + context_half_width,
        )

        dem = self.dem_source.reproject_to_shape(
            target_crs=self.target_crs,
            tile_bounds=context_bounds,
            out_width=self.tile_size,
            out_height=self.tile_size,
            resampling=Resampling.bilinear,
        )

        if dem is None:
            return None

        dem_array = np.asarray(dem)

        if not np.isfinite(dem_array).any():
            return None

        context_transform = from_bounds(
            *context_bounds,
            self.tile_size,
            self.tile_size,
        )

        context_mask = rasterize(
            [(obj["geom"], 1)],
            out_shape=(
                self.tile_size,
                self.tile_size,
            ),
            transform=context_transform,
            fill=0,
            dtype=np.uint8,
        )

        dem_context, scalar_features = build_dem_context(
            dem=dem,
            candidate_mask=context_mask,
            context_width_m=context_width_m,
            candidate_diameter_m=obj["diameter"],
        )

        if dem_context is None:
            return None

        return {
            "dem_context": dem_context,
            "scalar_features": scalar_features,
            "id": obj["id"],
            "area": obj["area"],
            "diameter": obj["diameter"],
            "bounds": context_bounds,
            "width_m": context_width_m,
        }

    # show for labelling
    def _show_tile(self, tile):

        # stack not in seperate channels (last dim for rgb plotting)
        rgb = np.moveaxis(tile["rgb"], 0, -1,)

        # clipping for correct colours to show visually for manual labelling
        low = np.nanpercentile(rgb, 2, axis=(0, 1))
        high = np.nanpercentile(rgb, 98, axis=(0, 1))

        rgb_display = np.clip((rgb - low) / (high - low + 1e-6), 0, 1)

        relative_dem = tile["dem_context"][0]
        context_mask = tile["dem_context"][2]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # plot rgb
        axes[0].imshow(rgb_display, interpolation="nearest")
        # display mask
        axes[0].contour(tile["mask"],levels=[0.5], colors="red")

        # plotting params
        axes[0].set_title("Local RGB")
        axes[0].axis("off")

        # plot dem
        axes[1].imshow(relative_dem,cmap="terrain")
        # display mask
        axes[1].contour(context_mask, levels=[0.5], colors="red")
        axes[1].set_title(f"DEM context: {tile['context_width_m']:.0f} m")
        axes[1].axis("off")

        plt.suptitle(f"Candidate {tile['id']}")
        plt.tight_layout()
        plt.show()

    # for manual labelling of candidates, with export to npz files for training / validation of veto classifier
    def label(self, min_area=10):

        objs = self.predictions.objects(self.target_crs)

        print(f"{len(objs)} candidate objects found.")

        for obj in objs:

            if obj["area"] < min_area:
                continue

            tile = self._make_tile(obj)

            if tile is None:
                continue

            clear_output(wait=True)
            self._show_tile(tile)

            print(f"Object diameter: {tile['diameter']:.1f} m\n"
                  f"Local width: {tile['local_crop_width_m']:.1f} m\n"
                f"Context width: {tile['context_width_m']:.1f} m"
            )

            answer = input(
                "r=RETAIN, x=REJECT, s=SKIP, q=QUIT: "
            ).strip().lower()

            if answer == "q":
                break

            if answer == "s":
                continue

            if answer == "r":
                tile["label"] = 1
                tile["label_name"] = "RETAIN"

            elif answer == "x":
                tile["label"] = 0
                tile["label_name"] = "REJECT"

            else:
                print("Invalid input; candidate skipped.")
                continue

            self.tiles.append(tile)

            print(f"{tile['label_name']} | " f"{len(self.tiles)} candidates labelled")

    # export tiles labelled in label()
    def export(self, output_directory):

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        for tile in self.tiles:

            label_directory = output_directory / tile["label_name"]

            label_directory.mkdir(parents=True, exist_ok=True)

            output_path = label_directory / f"candidate_{tile['id']:05d}.npz"

            np.savez_compressed(
                output_path,
                rgb=tile["rgb"],
                mask=tile["mask"],
                dem_context=tile["dem_context"],
                scalar_features=tile["scalar_features"],
                scalar_names=np.asarray(VETO_SCALAR_NAMES),
                label=np.int64(tile["label"]),
                candidate_id=np.int64(tile["id"]),
                bounds=np.asarray(tile["bounds"], dtype=np.float64),
                context_bounds=np.asarray(tile["context_bounds"], dtype=np.float64),
                local_crop_width_m=np.float32(tile["local_crop_width_m"]),
                context_width_m=np.float32(tile["context_width_m"]))

        print(f"Exported {len(self.tiles)} candidates.")

