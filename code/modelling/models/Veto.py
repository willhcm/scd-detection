import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

class VetoTrainingDataset(nn.Module):

    
    def __init__(
        self,
        all_paths,
        tile_size=512,
        skip_partial=True,
        augment=False,
        min_instance_area=20,
    ):
        self.augment = augment
        self.paths = all_paths
        self.min_instance_area = min_instance_area


    def __len__(self):
        return len(self.paths)

    def __len__(self):
        ...

    def __getitem__(self):
        ...


class VetoClassifier(nn.Module):

    def __init__(self, dropout=0.2, in_channels=4):
        super().__init__()

        base_model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        n_features = base_model.fc.in_features

        if in_channels > 3:
            old_conv1 = base_model.conv1
            new_conv1 = nn.Conv2d(
                in_channels, old_conv1.out_channels,
                kernel_size=old_conv1.kernel_size,
                stride=old_conv1.stride,
                padding=old_conv1.padding,
                bias=old_conv1.bias is not None,
            )
            with torch.no_grad():
                new_conv1.weight[:, :3] = old_conv1.weight
                new_conv1.weight[:, 3:] = old_conv1.weight.mean(dim=1, keepdim=True)
            base_model.conv1 = new_conv1

        # ResNet now returns the pooled [B, 512] so i can run my own classifier. (replaces fully connected layer with identity matrix)
        base_model.fc = nn.Identity()

        self.backbone = base_model

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)

        # removes extra dim
        return logits.squeeze(1)
