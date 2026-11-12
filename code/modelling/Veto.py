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
        self.context_encoder = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.candidate_encoder = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        candidate_conv = self.candidate_encoder.features[0][0]
        self.candidate_encoder.features[0][0] = nn.Conv2d(
            2,
            candidate_conv.out_channels,
            kernel_size=candidate_conv.kernel_size,
            stride=candidate_conv.stride,
            padding=candidate_conv.padding,
            bias=candidate_conv.bias is not None,
        )
        with torch.no_grad():
            self.candidate_encoder.features[0][0].weight.copy_(
                candidate_conv.weight[:, :2]
            )
            if candidate_conv.bias is not None:
                self.candidate_encoder.features[0][0].bias.copy_(candidate_conv.bias)

        dem_feature_dim = self.context_encoder.classifier[0].in_features
        self.context_encoder.classifier = nn.Identity()  # Remove the original classifier
        self.candidate_encoder.classifier = nn.Identity()  # Remove the original classifier
        
        # squeezed so the signal does not dominate the scalars. Final encodings are [64,] for image and [32,] for scalars.
        self.squeezer = nn.Sequential(
            nn.Linear(2 * dem_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 32),  
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

        # shared after fusing
        self.classifier = nn.Sequential(
            nn.Linear(32 + len(VETO_SCALAR_NAMES), 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 1))

    def forward(self, context, candidate, scalar_features):

        # get feature encodings
        context_features = self.context_encoder(context)
        candidate_features = self.candidate_encoder(candidate)

        image_features = torch.cat([context_features, candidate_features], dim=1)
        # squeeze the image features
        image_features = self.squeezer(image_features)

        # concat encodings
        fused = torch.cat([image_features, scalar_features], dim=1)

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
    
    def _augment(self, context, candidate, scalars, sigma=0.5):
        """
        Apply identical spatial transformations to DEM, Slope and Mask Channels. Add small guassian noise to scalars.
        """

        context = context.copy()
        candidate = candidate.copy()
        scalars = scalars.copy()

        flip_x = random.random() > 0.5
        flip_y = random.random() > 0.5
        k = random.choice([0, 1, 2, 3])

        def transform(image):
            if flip_x:
                image = image[:, :, ::-1]
            if flip_y:
                image = image[:, ::-1, :]
            if k:
                image = np.rot90(image, k, axes=(1, 2))
            return image

        context = transform(context)
        candidate = transform(candidate)

        # scale sigma to the range of the scalars, to avoid overwhelming the signal with noise.
        for i, scalar in enumerate(scalars):
            if np.isfinite(scalar):
                scalars[i] += np.random.normal(0, sigma * abs(scalar) * 0.1)

        return (np.ascontiguousarray(context), np.ascontiguousarray(candidate), scalars)

    def __getitem__(self, idx):

        path = self.paths[idx]

        # open path and fetch data for this sample
        with np.load(path) as data:

            context = data["context"].astype(np.float32, copy=True)
            candidate = data["candidate"].astype(np.float32, copy=True)
            scalar_features = data["scalar_features"].astype(np.float32, copy=True)
            label = float(data["label"])    

        # unpack image parts (DEM, SLOPE, Mask from Mask R-CNN proposal model)
        mask = context[VETO_BAND_INDEX['MASK']] > 0.5 # threshold mask in place
        slope = context[VETO_BAND_INDEX['SLOPE']]
        dem = context[VETO_BAND_INDEX['DEM']]
        candidate_dem = candidate[0]
        candidate_mask = candidate[1] > 0.5 # threshold mask in place

        # reshape (why?)
        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        to_normalize = {
            'dem': dem,
            'slope': slope,
            'scalar_features': scalar_features,
            'candidate_dem': candidate_dem}
        
        # handle outliers, before normalising
        dem, slope, scalar_features, candidate_dem = normalise_veto_data(to_normalize, norm_stats=self.norm_stats)
        context = np.stack([dem, slope, mask.astype(np.float32)], axis=0).astype(np.float32)
        candidate = np.stack([candidate_dem, candidate_mask.astype(np.float32)], axis=0).astype(np.float32)
        
        # augment sample to increase training diversity and mitigate overfitting to small dataset.
        if self.augment:
            context, candidate, scalar_features = self._augment(context, candidate, scalar_features)

        # to tensors for numpy
        context = torch.from_numpy(np.ascontiguousarray(context, dtype=np.float32))
        candidate = torch.from_numpy(np.ascontiguousarray(candidate, dtype=np.float32))
        scalar_features = torch.from_numpy(np.ascontiguousarray(scalar_features, dtype=np.float32))
        label = torch.tensor(label, dtype=torch.float32) # 1 is accept, 0 is reject

        return (context, candidate, scalar_features, label)