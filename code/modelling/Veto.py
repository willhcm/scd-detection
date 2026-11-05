import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import sys
import cv2 # computer vision, used for resizing images and masks.
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import random

CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from helpers.VetoHelpers import VETO_SCALAR_NAMES, VETO_BAND_INDEX, normalise_veto_data

class Vetoer(nn.Module):
    """
    A neural network model to screen out false positives from Mask R-CNN predictions.
    """
    # 3 encoder branches and shared classifier
    def __init__(self, scalar_dim=len(VETO_SCALAR_NAMES), dropout=0.3):
        super().__init__()

        # Relative DEM + slope + candidate mask. 3 channels, no replace conv needed
        self.dem_encoder = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        dem_feature_dim = self.dem_encoder.classifier[0].in_features
        self.dem_encoder.classifier = nn.Identity()  # Remove the original classifier

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True))

        # squeezed so the signal does not dominate the scalars. Final encodings are [64,] for image and [32,] for scalars.
        self.image_squeezer = nn.Sequential(
            nn.Linear(dem_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))  

        # shared after fusing
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, image, scalar_features):

        # get feature encodings
        image_features = self.dem_encoder(image)

        # get encodings
        scalar_features = self.scalar_encoder(scalar_features)
        image_features = self.image_squeezer(image_features)

        # concat encodings
        fused = torch.cat([image_features, scalar_features,], dim=1)

        # classify and return
        return self.classifier(fused).squeeze(1)     


# AI assistance in handling shapes and channels for VetoDataset
class VetoDataset(Dataset):
    """
    A PyTorch Dataset class for loading and preprocessing data for the Veto classifier.
    Each item in the dataset consists of local RGB data, DEM context, scalar features and a binary label"""

    def __init__(
        self,
        paths,
        norm_stats,
        augment=False,
        dem_size=400):

        self.paths = list(paths)
        self.dem_size = dem_size
        self.norm_stats = norm_stats
        self.augment = augment

    def __len__(self):
        return len(self.paths)
    
    def _augment(self, image, scalars, sigma=0.5):
        """
        Apply identical spatial transformations to DEM, Slope and Mask Channels. Add small guassian noise to scalars.
        """

        image = image.copy()
        scalars = scalars.copy()

        # flip
        if random.random() > 0.5:
            image = image[:, :, ::-1]

        # flip
        if random.random() > 0.5:
            image = image[:, ::-1, :]

        # rotate 0, 90, 180, 270.
        k = random.choice([0, 1, 2, 3])
        if k:
            image = np.rot90(image, k, axes=(1, 2))

        # scale sigma to the range of the scalars, to avoid overwhelming the signal with noise.
        for i, scalar in enumerate(scalars):
            if np.isfinite(scalar):
                scalars[i] += np.random.normal(0, sigma * abs(scalar) * 0.1)

        return (np.ascontiguousarray(image), scalars)

    def __getitem__(self, idx):

        path = self.paths[idx]

        # open path and fetch data for this sample
        with np.load(path) as data:

            image = data["image"].astype(np.float32, copy=True)
            scalar_features = data["scalar_features"].astype(np.float32, copy=True)
            label = float(data["label"])    

        # unpack image parts (DEM, SLOPE, Mask from Mask R-CNN proposal model)
        mask = image[VETO_BAND_INDEX['MASK']] > 0.5 # threshold mask in place ...
        slope = image[VETO_BAND_INDEX['SLOPE']]
        dem = image[VETO_BAND_INDEX['DEM']] 

        # reshape (why?)
        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        to_normalize = {
            'dem': dem,
            'slope': slope,
            'scalar_features': scalar_features}

        # handle outliers, before normalising
        dem, slope, scalar_features = normalise_veto_data(to_normalize, norm_stats=self.norm_stats)
        image = np.stack([dem, slope, mask.astype(np.float32)], axis=0).astype(np.float32)

        # augment sample to increase training diversity and mitigate overfitting to small dataset.
        if self.augment:
            image, scalar_features = self._augment(image, scalar_features)

        # to tensors for numpy
        image = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        scalar_features = torch.from_numpy(np.ascontiguousarray(scalar_features, dtype=np.float32))
        label = torch.tensor(label, dtype=torch.float32) # 1 is accept, 0 is reject

        return (image, scalar_features, label)