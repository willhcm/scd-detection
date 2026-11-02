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

def replace_first_conv(
    model,
    in_channels,
    copy_pretrained=True):

    old_conv = model.conv1

    # replace conv (new inputs compared to pretrained model!)
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False)

    # use mean weights for extra channels
    if copy_pretrained:
        with torch.no_grad():

            channels_to_copy = min(3, in_channels)

            new_conv.weight[:, :channels_to_copy,].copy_(old_conv.weight[:, :channels_to_copy,])

            if in_channels > 3:
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)

                new_conv.weight[:, 3:].copy_(mean_weight.repeat(1, in_channels - 3, 1, 1))

    model.conv1 = new_conv
    return model


class VetoClassifier(nn.Module):
    """
    A neural network model to screen out false positives from Mask R-CNN predictions.
    """
    # 3 encoder branches and shared classifier
    def __init__(self, scalar_dim=len(VETO_SCALAR_NAMES), dropout=0.3):
        super().__init__()

        # Local Planet RGB + candidate mask.
        self.rgb_encoder = resnet18(ResNet18_Weights.DEFAULT)
        self.rgb_encoder = replace_first_conv(self.rgb_encoder, in_channels=4,copy_pretrained=True)
        rgb_feature_dim = (self.rgb_encoder.fc.in_features)
        self.rgb_encoder.fc = nn.Identity()

        # Relative DEM + slope + candidate mask.
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

        # change to KNN?? see performance, will be quicker probs afeer encoder.
        # shared after fusing
        self.classifier = nn.Sequential(
            nn.Linear(rgb_feature_dim + dem_feature_dim + 32,
                      256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1))
        

    def forward(self, rgb_local, dem_context, scalar_features):

        # get feature encodings
        rgb_features = self.rgb_encoder(rgb_local)
        dem_features = self.dem_encoder(dem_context)

        # prep scalars, and get encodings
        scalar_features = (scalar_features - self.scalar_mean) / self.scalar_std.clamp_min(1e-6)
        scalar_features = self.scalar_encoder(scalar_features)

        # concat encodings
        fused = torch.cat([rgb_features, dem_features, scalar_features,], dim=1)

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
        dem_size=400):

        self.paths = list(paths)
        self.dem_size = dem_size
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.paths)

    def normalise(self, dem_context):

        dem_band = dem_context[0]
        slope_band = dem_context[1]

        dem_band = (dem_band - self.norm_stats['DEM']['median']) / self.norm_stats['DEM']['IQR']
        slope_band = (slope_band - self.norm_stats['SLOPE']['median']) / self.norm_stats['SLOPE']['IQR']
            
    def __getitem__(self, idx):

        path = self.paths[idx]

        with np.load(path) as data:

            mask = data["mask"].astype(np.float32, copy=True)
            dem_context = data["dem_context"].astype(np.float32, copy=True)
            scalar_features = data["scalar_features"].astype(np.float32, copy=True)
            label = float(data["label"])

        mask = (mask > 0.5).astype(np.float32)
        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        # to tensors
        dem_context = self.normalise(dem_context)
        dem_context = torch.from_numpy(np.ascontiguousarray(dem_context, dtype=np.float32))
        scalar_features = torch.from_numpy(np.ascontiguousarray(scalar_features, dtype=np.float32))

        # 1 is accept, 0 is reject
        label = torch.tensor(label, dtype=torch.float32)

        return (dem_context, scalar_features, label)