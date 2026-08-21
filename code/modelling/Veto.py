import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import sys
import cv2

CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from helpers.veto_helpers import VETO_SCALAR_NAMES


def replace_first_conv(
    model,
    in_channels,
    copy_pretrained=True,
):
    old_conv = model.conv1

    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

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

        # defaults make scalar standardisation a no-op until statistics are assigned
        self.register_buffer("scalar_mean", torch.zeros(scalar_dim))
        self.register_buffer("scalar_std", torch.ones(scalar_dim))

    def set_scalar_statistics(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)

        std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6) # no 0 divide

        self.scalar_mean.copy_(mean)
        self.scalar_std.copy_(std)

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
        rgb_size=96,
        dem_size=224,
    ):
        self.paths = list(paths)
        self.rgb_size = rgb_size
        self.dem_size = dem_size

    def __len__(self):
        return len(self.paths)

    @staticmethod
    # make sure rgb is [h, w, band]
    def _ensure_rgb_hwc(rgb):
        rgb = np.asarray(rgb)

        if (rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3):
            rgb = np.moveaxis(rgb, 0, -1)

        return rgb

    def __getitem__(self, idx):

        path = self.paths[idx]

        with np.load(path) as data:

            rgb = data["rgb"].astype(np.float32, copy=True)

            mask = data["mask"].astype(np.float32, copy=True)

            dem_context = data["dem_context"].astype(np.float32, copy=True)

            scalar_features = data["scalar_features"].astype(np.float32, copy=True)

            label = float(data["label"])


        # local RGB + candidate-mask branch
        rgb = self._ensure_rgb_hwc(rgb)

        mask = (mask > 0.5).astype(np.float32)

        # add mask to rgb
        rgb_local = np.concatenate([rgb, mask[..., None],], axis=-1,)

        # bands to front
        rgb_local = np.moveaxis(rgb_local, -1, 0)

        # wider DEM-context branch

        if dem_context.shape[1:] != (self.dem_size, self.dem_size):
            # Resize in channels-last form.
            dem_hwc = np.moveaxis(dem_context,0,-1, )

            dem_hwc = cv2.resize(dem_hwc, (self.dem_size, self.dem_size), interpolation=cv2.INTER_LINEAR)

            # Restore the candidate-mask channel using
            # nearest-neighbour interpolation.
            context_mask = cv2.resize(dem_context[2], (self.dem_size, self.dem_size), interpolation=cv2.INTER_NEAREST)

            dem_hwc[..., 2] = (context_mask > 0.5).astype(np.float32)

            dem_context = np.moveaxis(dem_hwc, -1, 0)

        scalar_features = np.asarray(scalar_features, dtype=np.float32).reshape(-1)

        # to tensors
        rgb_local = torch.from_numpy(np.ascontiguousarray(rgb_local, dtype=np.float32))
        dem_context = torch.from_numpy(np.ascontiguousarray(dem_context, dtype=np.float32))
        scalar_features = torch.from_numpy(np.ascontiguousarray(scalar_features, dtype=np.float32))

        # 1 is accept, 0 is reject
        label = torch.tensor(label, dtype=torch.float32)

        return (rgb_local, dem_context, scalar_features, label)