import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from ScaleNormalisedDataStack import ShapeLabels, DataSource
from IPython.display import clear_output


class VetoLoader:

    def __init__(self, shp_path, rgb_path):

        self.predictions = ShapeLabels(shp_path)
        self.spectral = DataSource.from_tiff(
            rgb_path,
            type="RGB",
        )

        self.target_crs = self.spectral.crs
        self.tiles = []

    def _make_tile(
        self,
        obj,
        tile_size=96,
        min_crop_pixels=64,
        object_fraction=0.5,
    ):


        cx = obj["cx"]
        cy = obj["cy"]

        # Minimum crop extent at native Planet resolution.
        min_crop_width_m = min_crop_pixels * self.spectral.res

        # Ensure the object occupies at most 50% of the crop width.
        object_crop_width_m = obj["diameter"] / object_fraction

        crop_width_m = max(
            min_crop_width_m,
            object_crop_width_m,
        )

        half_width = crop_width_m / 2

        tile_bounds = (
            cx - half_width,
            cy - half_width,
            cx + half_width,
            cy + half_width,
        )

        rgb = self.spectral.reproject_to_shape(
            target_crs=self.target_crs,
            tile_bounds=tile_bounds,
            out_width=tile_size,
            out_height=tile_size,
            resampling=Resampling.bilinear,
        )

        if np.all(rgb == 0):
            return None

        tile_transform = from_bounds(
            *tile_bounds,
            tile_size,
            tile_size,
        )

        mask = rasterize(
            [(obj["geom"], 1)],
            out_shape=(tile_size, tile_size),
            transform=tile_transform,
            fill=0,
            dtype=np.uint8,
        )

        return {
            "rgb": rgb,
            "mask": mask,
            "id": obj["id"],
            "area": obj["area"],
            "diameter": obj["diameter"],
            "bounds": tile_bounds,
            "mask_fraction": float(mask.mean()),
        }

    # for colab manual labelling
    def _show_tile(self, tile):

        rgb = np.moveaxis(tile["rgb"], 0, -1)

        # Display scaling only.
        low = np.nanpercentile(rgb, 2, axis=(0, 1))
        high = np.nanpercentile(rgb, 98, axis=(0, 1))

        rgb_display = np.clip(
            (rgb - low) / (high - low + 1e-6),
            0,
            1,
        )

        plt.figure(figsize=(5, 5))
        plt.imshow(rgb_display, interpolation='nearest')

        # Show the Mask R-CNN candidate outline.
        plt.contour(
            tile["mask"],
            levels=[0.5],
            colors="red",
        )

        plt.title(f"Candidate {tile['id']}")
        plt.axis("off")
        plt.show()

    # for colab manual labelling again 
    def label(self, min_area=10):
        
        objs = self.predictions.objects(
            self.target_crs
        )
        print(len(objs))
        for obj in objs:

            if obj["area"] < min_area:
                continue

            tile = self._make_tile(obj)

            if tile is None:
                continue

            clear_output(wait=True)
            self._show_tile(tile)

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

            print(
                f"{tile['label_name']} | "
                f"{len(self.tiles)} candidates labelled"
            )

    def export(self, output_directory):

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        for tile in self.tiles:
            label_directory = (output_directory / tile["label_name"])

            label_directory.mkdir(parents=True, exist_ok=True)

            output_path = (label_directory / f"candidate_{tile['id']:05d}.npz")

            np.savez_compressed(
                output_path,
                rgb=tile["rgb"],
                mask=tile["mask"],
                label=np.int64(tile["label"]),
                candidate_id=np.int64(tile["id"]),
                bounds=np.asarray(tile["bounds"], dtype=np.float64))

        print(f"Exported {len(self.tiles)} candidates.")