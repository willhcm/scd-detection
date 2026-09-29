# veto classifier which uses only DEM data, no RGB needed. 
# useful for deployment in regions where SCDs are likely too small for worthwhile use of lower res (3m) PlanetLabs data. 
# approach also removes need for two datasources, reduces running time, and makes deployment over large areas considerably easier to set up

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import sys
import cv2 # computer vision, used for resizing images and masks.

CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from helpers.VetoHelpers import VETO_SCALAR_NAMES

class DEM_Based_Vetoer(nn.Module):
    """
    A neural network model to screen out false positives from Mask R-CNN predictions.
    """
    # 3 encoder branches and shared classifier
    def __init__(self, scalar_dim=len(VETO_SCALAR_NAMES), dropout=0.3):
        super().__init__()

        # Relative DEM + slope + candidate mask. 3 channels, no replace conv needed
        self.dem_encoder = resnet18(weights=None)

        dem_feature_dim = (self.dem_encoder.fc.in_features)
        self.dem_encoder.fc = nn.Identity()

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # shared after fusing
        self.classifier = nn.Sequential(
            nn.Linear(dem_feature_dim + 32,
                      256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1))

        # defaults make scalar standardisation a no-op until statistics are assigned
        self.register_buffer("scalar_mean", torch.zeros(scalar_dim))
        self.register_buffer("scalar_std", torch.ones(scalar_dim))

    def set_scalar_statistics(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)

        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6) # no 0 divide

        self.scalar_mean.copy_(mean)
        self.scalar_std.copy_(std)

    def forward(self, dem_context, scalar_features):

        # get feature encodings
        dem_features = self.dem_encoder(dem_context)

        # prep scalars, and get encodings
        scalar_features = (scalar_features - self.scalar_mean) / self.scalar_std.clamp_min(1e-6)
        scalar_features = self.scalar_encoder(scalar_features)

        # concat encodings
        fused = torch.cat([dem_features, scalar_features,], dim=1)

        # classify and return
        return self.classifier(fused).squeeze(1)    


    # AI assistance in handling shapes and channels for VetoDataset
class DEM_Based_Veto_Dataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing data for the Veto classifier.
    Each item in the dataset consists of local RGB data, DEM context, scalar features and a binary label"""

    def __init__(self, paths, dem_size=224):
        self.paths = list(paths)
        self.dem_size = dem_size

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):

        path = self.paths[idx]

        with np.load(path) as data:

            mask = data["mask"].astype(np.float32, copy=True)

            dem_context = data["dem_context"].astype(np.float32, copy=True)

            scalar_features = data["scalar_features"].astype(np.float32, copy=True)

            label = float(data["label"])

        mask = (mask > 0.5).astype(np.float32)

        # DEM-context 
        # handles different context sizes in training set.
        if dem_context.shape[1:] != (self.dem_size, self.dem_size):
            # Resize in channels-last form.
            dem_hwc = np.moveaxis(dem_context, 0, -1)

            dem_hwc = cv2.resize(dem_hwc, (self.dem_size, self.dem_size), interpolation=cv2.INTER_LINEAR)

            # Restore the candidate-mask channel using
            # nearest-neighbour interpolation.
            context_mask = cv2.resize(dem_context[2], (self.dem_size, self.dem_size), interpolation=cv2.INTER_NEAREST)

            dem_hwc[..., 2] = (context_mask > 0.5).astype(np.float32)

            dem_context = np.moveaxis(dem_hwc, -1, 0)

        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        # to tensors
        dem_context = torch.from_numpy(np.ascontiguousarray(dem_context, dtype=np.float32))
        scalar_features = torch.from_numpy(np.ascontiguousarray(scalar_features, dtype=np.float32))

        # 1 is accept, 0 is reject
        label = torch.tensor(label, dtype=torch.float32)

        return (dem_context, scalar_features, label)