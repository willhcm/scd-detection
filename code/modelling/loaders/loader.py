from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

# these will be incorporated into the __init__ eventually, just not got round to it yet!
_BANDS_TO_LOAD = ['HILLSHADE']

# Bands normalised per-tile (z-score)
_PER_TILE_NORM = {'HILLSHADE'}

class TestSCDDataset(Dataset):

    def __init__(self, paths, tile_size=512, skip_partial=True):
        self.good_paths = sorted(paths)
        print(f"Found {len(self.good_paths)} tiles")

    def __len__(self):
        return len(self.good_paths)

    def __getitem__(self, idx):
        d = np.load(self.good_paths[idx], allow_pickle=True)

        full_image = d["image"].astype(np.float32) 
        image = full_image[0:1]  # customised for hillshade model right now, will change and make flexible.

        # normalise
        band = image[0]
        mean = np.nanmean(band)
        std = np.nanstd(band)
        image[0] = (band - mean) / (std + 1e-6)

        image = np.nan_to_num(image, nan=0.0)

        return (torch.from_numpy(np.ascontiguousarray(image)), self.good_paths[idx])
            
def get_test_loader(test_set, batch_size=8):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)