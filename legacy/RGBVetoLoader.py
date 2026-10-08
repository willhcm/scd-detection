## Legacy

class RGBVetoLoader:

    def __init__(
        self,
        shp_path,
        dem_path,
        rgb_flag = False,
        rgb_path = None,
        target_crs = None,
        context_scale=4.0,
        min_context_width_m=768.0,
        max_context_width_m=4000.0,
        context_tile_size=224):

        self.predictions = ShapeLabels(shp_path)

        self.rgb_flag = rgb_flag

        if self.rgb_flag:
            self.spectral = DataSource.from_tiff(
                            rgb_path,
                            type="RGB")
            self.rgb_res = self.rgb_source.res
        else:
            self.rgb_res = 3 # planet labs backup

        self.dem_source = DataSource.from_tiff(
            dem_path,
            type="DEM")

        self.target_crs = target_crs
        self.tiles = []

        self.context_scale = context_scale
        self.min_context_width_m = min_context_width_m
        self.max_context_width_m = max_context_width_m
        self.context_tile_size = context_tile_size

    # build tile from candidate object, including local RGB and DEM context
    def _make_tile(self, obj, tile_size=96, min_crop_pixels=64, object_fraction=0.5):

        # centroid 
        cx = obj["cx"]
        cy = obj["cy"]

        # object size
        diameter_m = (obj.equivalent_diameter_area * 1)
        object_crop_width_m = (diameter_m / object_fraction)
        min_crop_width_m = (min_crop_pixels * 1)
        local_crop_width_m = max(min_crop_width_m, object_crop_width_m)
        local_half_width = local_crop_width_m / 2.0

        if self.rgb_flag:
            local_bounds = (
                cx - local_half_width,
                cy - local_half_width,
                cx + local_half_width,
                cy + local_half_width,
            )

            rgb = self.spectral.reproject_to_shape(
                target_crs=self.target_crs,
                tile_bounds=local_bounds,
                out_width=tile_size,
                out_height=tile_size,
                resampling=Resampling.bilinear,
            )

            if not np.isfinite(rgb).any() or np.all(rgb == 0):
                return None

            local_transform = from_bounds(
                *local_bounds,
                tile_size,
                tile_size,
            )

            local_mask = rasterize(
                [(obj["geom"], 1)],
                out_shape=(tile_size, tile_size),
                transform=local_transform,
                fill=0,
                dtype=np.uint8,
        )

        # DEM crop width
        # Four-times wider context, clipped to fixed limits.
        context_width_m = float(
            np.clip(
                self.context_scale * local_crop_width_m,
                self.min_context_width_m,
                self.max_context_width_m,
            )
        )

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
            out_width=self.context_tile_size,
            out_height=self.context_tile_size,
            resampling=Resampling.bilinear,
        )

        if dem is None:
            return None

        dem_array = np.asarray(dem)

        if not np.isfinite(dem_array).any():
            return None

        context_transform = from_bounds(
            *context_bounds,
            self.context_tile_size,
            self.context_tile_size,
        )

        context_mask = rasterize(
            [(obj["geom"], 1)],
            out_shape=(
                self.context_tile_size,
                self.context_tile_size,
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
            "rgb": rgb.astype(np.float32),
            "mask": local_mask,
            "dem_context": dem_context,
            "scalar_features": scalar_features,
            "id": obj["id"],
            "area": obj["area"],
            "diameter": obj["diameter"],
            "bounds": local_bounds,
            "context_bounds": context_bounds,
            "local_crop_width_m": local_crop_width_m,
            "context_width_m": context_width_m,
            "mask_fraction": float(local_mask.mean()),
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
