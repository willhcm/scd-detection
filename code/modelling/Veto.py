import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from code.helpers.veto_helpers import VETO_SCALAR_NAMES


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

                new_conv.weight[:, 3:].copy_( mean_weight.repeat(1, in_channels - 3, 1, 1))

    model.conv1 = new_conv
    return model


class VetoClassifier(nn.Module):

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