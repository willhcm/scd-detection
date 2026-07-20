import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from ScaleNormalisedDataStack import ShapeLabels, DataSource

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
        tile_size=224,
        padding=1.0,
    ):
        """
        Extract an RGB crop centred on one predicted candidate.
        """

        cx = obj["cx"]
        cy = obj["cy"]

        # padding=1.0 gives approximately one object diameter
        # of context around each side.
        crop_width = obj["diameter"] * (1 + 2 * padding)

        half_width = crop_width / 2

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

        tile_transform = from_bounds(
            *tile_bounds,
            tile_size,
            tile_size,
        )

        # Rasterise only the current candidate, not every candidate in the bounds
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
        plt.imshow(rgb_display)

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
    def label(self, tile_size=224, padding=1.0, min_area=20):
        
        objs = self.predictions.objects(
            self.target_crs
        )

        for obj in objs:

            if obj["area"] < min_area:
                continue

            tile = self._make_tile(
                obj,
                tile_size=tile_size,
                padding=padding,
            )

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