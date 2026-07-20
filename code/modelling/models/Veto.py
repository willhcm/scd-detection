import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

class VetoDataset(nn.Module):

    def __init__(self):
        super().__init__()
        ...

    def __len__(self):
        ...

    def __getitem__(self):
        ...


class VetoClassifier(nn.Module):

    def __init__(self, dropout=0.2):
        super().__init__()

        base_model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        n_features = base_model.fc.in_features

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
        
