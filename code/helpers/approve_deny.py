import glob
import shutil # AI assistance with this more efficient copying files and managing directories
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SRC_DIR = Path('/Users/willmcmahon/Library/CloudStorage/GoogleDrive-will.mcmahon04@gmail.com/My Drive/IRP/Tiles/NativeRes')
APPROVED_DIR = Path("../../../data/DatasetDecisions/Approved")
REJECTED_DIR = Path("../../../data/DatasetDecisions/Rejected")

APPROVED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)

paths = glob.glob('/Users/willmcmahon/Library/CloudStorage/GoogleDrive-will.mcmahon04@gmail.com/My Drive/IRP/SNIPCluster/*/*.npz')

print(f"Found {len(paths)} candidate tiles")

def show_tile(path):
    d = np.load(path, allow_pickle=True)

    image = d["image"]
    labels = d["labels"]

    layer_names = list(d["layer_names"])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # show dem 
    axes[0].imshow(image[layer_names.index("DEM")], cmap="terrain")
    axes[0].set_title("DEM")

    # show rr
    if "RR" in layer_names:
        axes[1].imshow(image[layer_names.index("RR")], cmap="coolwarm")
        axes[1].set_title("Residual relief")

    # show label
    axes[2].imshow(labels, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Label")

    # Overlay dem on labels
    axes[3].imshow(image[layer_names.index("DEM")], cmap="terrain")
    axes[3].contour(labels, levels=[0.5], colors="red", linewidths=1)

    for ax in axes:
        ax.axis("off")

    title = path.name
    if "res" in d:
        title += f" | res={float(d['res']):.2f} m/px"
    if "target_fraction" in d:
        title += f" | frac={float(d['target_fraction']):.2f}"
    if "scd_pixel_fraction" in d:
        title += f" | pix={float(d['scd_pixel_fraction']):.3f}"

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.show()

for path in paths:
    show_tile(path)

    decision = input("[y] approve, [n] reject, [s] skip, [q] quit: ").strip().lower()

    if decision == "q":
        break

    elif decision == "y":
        shutil.copy2(path, APPROVED_DIR / path.name)
        print(f"approved: {path.name}")

    elif decision == "n":
        shutil.copy2(path, REJECTED_DIR / path.name)
        print(f"rejected: {path.name}")

    else:
        print(f"skipped: {path.name}")