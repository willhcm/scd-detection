import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from loaders.Cogs import ShapeLabels, WindowedDataSource
from IPython.display import clear_output
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import ipywidgets as widgets
import geopandas as gpd

from pathlib import Path
import sys

CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from helpers.VetoHelpers import VETO_SCALAR_NAMES, build_veto_data, x_y_to_long_lat

def hillshade(dem, azimuth=315, angle_altitude=45):
    """
    Compute hillshade from DEM.
    """
    x, y = np.gradient(dem)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.
    altituderad = angle_altitude * np.pi / 180.

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(slope) * np.cos(azimuthrad - aspect)
    return (shaded + 1) / 2  # Normalize to [0, 1]

class DEMVetoLabeller:

    def __init__(
        self,
        dem_path,
        region,
        pred_path = None,
        retain_path = None,
        reject_path = None,
        target_crs = None,
        context_scale=8.0,
        min_width_m=512.0,
        max_width_m=40000.0,
        tile_size=512):

        self.labels = None


        self.retains = ShapeLabels(retain_path) if retain_path is not None else None
        self.rejects = ShapeLabels(reject_path) if reject_path is not None else None

        self.from_shps = self.retains is not None and self.rejects is not None
        if self.from_shps:
            print(f"Labeller will build tiles from retain/reject shapefiles: {retain_path}, {reject_path}")

        if pred_path is not None and (retain_path is not None and reject_path is not None):
            self.predictions = ShapeLabels(pred_path)
            print(f"Loaded {len(self.predictions.objects(target_crs))} candidate objects from predictions shapefile.")

        if pred_path is None and (retain_path is None and reject_path is None):
            raise ValueError("Either pred_path or both retain_path and reject_path must be provided.")
        
        self.dem_source = WindowedDataSource.from_tiff_utm(
            dem_path,
            type="DEM")
        
        self.dem_res = self.dem_source.res
        self.target_crs = self.dem_source.crs
        self.tiles = []
        self.region = region
        self.tile_size = tile_size
        self.context_scale = context_scale
        self.min_width_m = min_width_m
        self.max_width_m = max_width_m

    # build tile from candidate object, including DEM context
    def _make_tile(self, obj):

        # centroid 
        cx = obj["cx"]
        cy = obj["cy"]

        # geographic centre of candidate object, in dms
        dms_centre = x_y_to_long_lat(cx, cy, self.target_crs, "EPSG:4326")

        # object size
        diameter_m = obj['diameter'] 

        # DEM crop width
        context_width_m = float(
            np.clip(
                self.context_scale * diameter_m,
                self.min_width_m,
                self.max_width_m))

        # build bbox around centre
        context_half_width = context_width_m / 2.0
        context_bounds = (
            cx - context_half_width,
            cy - context_half_width,
            cx + context_half_width,
            cy + context_half_width)

        candidate_width_m = float(diameter_m) * 2
        candidate_bounds = (
            cx - candidate_width_m / 2.0,
            cy - candidate_width_m / 2.0,
            cx + candidate_width_m / 2.0,
            cy + candidate_width_m / 2.0)

        dem_context = self.dem_source.reproject_to_shape(
            target_crs=self.target_crs,
            tile_bounds=context_bounds,
            out_width=self.tile_size,
            out_height=self.tile_size,
            resampling=Resampling.bilinear)

        candidate = self.dem_source.reproject_to_shape(
            target_crs=self.target_crs,
            tile_bounds=candidate_bounds,
            out_width=self.tile_size,
            out_height=self.tile_size,
            resampling=Resampling.bilinear)

        if dem_context is None or candidate is None:
            return None

        dem_array = np.asarray(dem_context)
        candidate_array = np.asarray(candidate)

        if not np.isfinite(dem_array).any() or not np.isfinite(candidate_array).any():
            return None

        context_transform = from_bounds(
            *context_bounds,
            self.tile_size,
            self.tile_size)

        candidate_transform = from_bounds(
            *candidate_bounds,
            self.tile_size,
            self.tile_size)

        context_mask = rasterize(
            [(obj["geom"], 1)],
            out_shape=(
                self.tile_size,
                self.tile_size),
            transform=context_transform,
            fill=0,
            dtype=np.uint8,
        )

        candidate_mask = rasterize(
            [(obj["geom"], 1)],
            out_shape=(
                self.tile_size,
                self.tile_size),
            transform=candidate_transform,
            fill=0,
            dtype=np.uint8,
        )

        context_image, candidate_image, scalar_features = build_veto_data(
            context=dem_context,
            context_mask=context_mask,
            candidate=candidate,
            candidate_mask=candidate_mask,
            context_width_m=context_width_m,
            candidate_diameter_m=diameter_m)

        if context_image is None or candidate_image is None or scalar_features is None:
            return None
    
        return {
            "context": context_image,
            "candidate": candidate_image,
            "scalar_features": scalar_features,
            "id": obj["id"],
            "centre_dms": dms_centre,
            "area": obj["area"],
            "diameter_m": diameter_m,
            "bounds": context_bounds,
            "context_bounds": context_bounds,
            "context_width_m": context_width_m,
        }

    # show for labelling
    def _show_tile(self, tile):

        relative_dem = tile["context"][0]
        context_mask = tile["context"][2]
        tile_hillshade = hillshade(relative_dem)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # plot dem
        ax.imshow(tile_hillshade, cmap="gray")
        # display mask
        ax.contour(context_mask, levels=[0.5], colors="red")
        ax.set_title(f"DEM context: {tile['context_width_m']:.0f} m")
        ax.axis("off")

        plt.suptitle(f"Candidate {tile['id']}")
        plt.tight_layout()
        plt.show()

    # for manual labelling of candidates, with export to npz files for training / validation of veto classifier

    def label(self, min_area=1):

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
                    f"Context width: {tile['context_width_m']:.1f} m\n\n"
                    f"Candidate {state['index'] + 1} / {len(candidates)}\n"
                    f'Depth_width: {tile["scalar_features"][-1]:.4f} m\n')

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

    def _build_from_shps(self):

        to_retain_objs = self.retains.objects(self.target_crs)
        to_reject_objs = self.rejects.objects(self.target_crs)

        tiles = []

        for obj in to_retain_objs:
            tile = self._make_tile(obj)
            if tile is not None:
                tile["label"] = 1
                tile["label_name"] = "RETAIN"
                tiles.append(tile)

        for obj in to_reject_objs:
            tile = self._make_tile(obj)
            if tile is not None:
                tile["label"] = 0
                tile["label_name"] = "REJECT"
                tiles.append(tile)

        return tiles

    # export tiles labelled in label()
    def export(self, output_directory):

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        if self.from_shps:
            self.tiles = self._build_from_shps()

        for tile in self.tiles:

            output_directory.mkdir(parents=True, exist_ok=True)

            output_path = output_directory / f"{self.region}_candidate_{tile['id']:05d}.npz"

            np.savez_compressed(
                output_path,
                mask = tile["image"][2],
                image=tile["image"],
                scalar_features=tile["scalar_features"],
                scalar_names=np.asarray(VETO_SCALAR_NAMES),
                region=str(self.region),
                label=np.int64(tile["label"]),
                centre_in_dms=tile["centre_dms"],
                bounds=np.asarray(tile["bounds"], dtype=np.float64),
                context_bounds=np.asarray(tile["context_bounds"], dtype=np.float64),
                diameter_m=np.float32(tile["diameter_m"]),
                context_width_m=np.float32(tile["context_width_m"]),
                crs=str(self.target_crs))

        print(f"Exported {len(self.tiles)} candidates.")
