import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from ScaleNormalisedDataStack import ShapeLabels, DataSource
from IPython.display import clear_output
import ipywidgets as widgets
from IPython.display import display, clear_output
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
    def _make_tile(self, obj, object_fraction=0.5):

        # centroid 
        cx = obj["cx"]
        cy = obj["cy"]

        # object size
        diameter_m = (obj["diameter"] * self.dem_source.res)
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
            candidate_diameter_m=diameter_m,
        )

        if dem_context is None:
            return None

        return {
            "dem_context": dem_context,
            "scalar_features": scalar_features,
            "id": obj["id"],
            "area": obj["area"],
            "diameter_m": diameter_m,
            "bounds": context_bounds,
            "context_bounds": context_bounds,
            "local_crop_width_m": object_crop_width_m,
            "context_width_m": context_width_m,
        }

    # show for labelling
    def _show_tile(self, tile):


        relative_dem = tile["dem_context"][0]
        context_mask = tile["dem_context"][2]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # plot dem
        ax.imshow(relative_dem,cmap="terrain")
        # display mask
        ax.contour(context_mask, levels=[0.5], colors="red")
        ax.set_title(f"DEM context: {tile['context_width_m']:.0f} m")
        ax.axis("off")

        plt.suptitle(f"Candidate {tile['id']}")
        plt.tight_layout()
        plt.show()

    # for manual labelling of candidates, with export to npz files for training / validation of veto classifier

    def label(self, min_area=10):

        objs = self.predictions.objects(self.target_crs)

        print(f"{len(objs)} candidate objects found.")

        # Filter candidates first
        candidates = []

        for obj in objs:
            if obj["area"] < min_area:
                continue

            tile = self._make_tile(obj)

            if tile is not None:
                candidates.append(tile)

        print(f"{len(candidates)} candidates available for labelling.")

        # State
        state = {
            "index": 0,
            "running": True,
        }

        # ------------------------------------------------------------------
        # Widgets
        # ------------------------------------------------------------------

        retain_button = widgets.Button(
            description="RETAIN",
            button_style="success",
            icon="check",
            layout=widgets.Layout(width="120px", height="45px")
        )

        reject_button = widgets.Button(
            description="REJECT",
            button_style="danger",
            icon="times",
            layout=widgets.Layout(width="120px", height="45px")
        )

        skip_button = widgets.Button(
            description="SKIP",
            button_style="warning",
            icon="forward",
            layout=widgets.Layout(width="120px", height="45px")
        )

        quit_button = widgets.Button(
            description="QUIT",
            button_style="",
            icon="stop",
            layout=widgets.Layout(width="120px", height="45px")
        )

        buttons = widgets.HBox(
            [
                retain_button,
                reject_button,
                skip_button,
                quit_button,
            ],
            layout=widgets.Layout(
                justify_content="center",
                gap="10px"
            )
        )

        output = widgets.Output()

        # ------------------------------------------------------------------
        # Display candidate
        # ------------------------------------------------------------------

        def show_candidate():

            if state["index"] >= len(candidates):
                state["running"] = False

                with output:
                    clear_output(wait=True)
                    print("Finished labelling all candidates.")

                return

            tile = candidates[state["index"]]

            with output:
                clear_output(wait=True)

                self._show_tile(tile)

                print(
                    f"Object diameter: {tile['diameter_m']:.1f} m\n"
                    f"Local width: {tile['local_crop_width_m']:.1f} m\n"
                    f"Context width: {tile['context_width_m']:.1f} m\n\n"
                    f"Candidate {state['index'] + 1} / {len(candidates)}\n"
                    f'Depth: {tile["scalar_features"][0]:.1f} m\n'
                )

        # ------------------------------------------------------------------
        # Label callback
        # ------------------------------------------------------------------

        def label_candidate(label, label_name):

            if not state["running"]:
                return

            tile = candidates[state["index"]]

            tile["label"] = label
            tile["label_name"] = label_name

            self.tiles.append(tile)

            state["index"] += 1

            # Move immediately to the next candidate
            show_candidate()


        # ------------------------------------------------------------------
        # Button callbacks
        # ------------------------------------------------------------------

        def on_retain(button):
            label_candidate(1, "RETAIN")

        def on_reject(button):
            label_candidate(0, "REJECT")

        def on_skip(button):
            if not state["running"]:
                return

            state["index"] += 1
            show_candidate()

        def on_quit(button):
            state["running"] = False

            with output:
                clear_output(wait=True)
                print(
                    f"Labelling stopped. "
                    f"{len(self.tiles)} candidates labelled."
                )

        retain_button.on_click(on_retain)
        reject_button.on_click(on_reject)
        skip_button.on_click(on_skip)
        quit_button.on_click(on_quit)


        display(output)
        display(buttons)

        show_candidate()

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
                mask = tile["dem_context"][2],
                dem_context=tile["dem_context"],
                scalar_features=tile["scalar_features"],
                scalar_names=np.asarray(VETO_SCALAR_NAMES),
                label=np.int64(tile["label"]),
                candidate_id=np.int64(tile["id"]),
                bounds=np.asarray(tile["bounds"], dtype=np.float64),
                context_bounds=np.asarray(tile["context_bounds"], dtype=np.float64),
                diameter_m=np.float32(tile["diameter_m"]),
                local_crop_width_m=np.float32(tile["local_crop_width_m"]),
                context_width_m=np.float32(tile["context_width_m"]))

        print(f"Exported {len(self.tiles)} candidates.")
