import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Formalisation of basic building blocks used in UNET, UNET++, CentreNetStrided architectures.

class DoubleConv(nn.Module):
    """Back bone UNET. 2 * (Conv2d + Batch + Relu)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """MaxPool2x2 followed by DoubleConv, one encoder step."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel=2, stride=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)

# test transpose as opposed to bilinear up
# might get checkboard artefacts but more parameters ?
class Up(nn.Module):
    """Bilinear upsample, concatenate skip, DoubleConv. one decoder step"""

    def __init__(self, x_ch: int, skip_ch, out_ch: int):
        super().__init__()
        # in_ch comes from (upsampled features + skip features)
       #self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up = nn.ConvTranspose2d(
            x_ch,
            x_ch,
            kernel_size=2,
            stride=2
        )
        self.conv = DoubleConv(x_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if the skip tensor is slightly larger (odd input dimensions)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                           0, skip.shape[-2] - x.shape[-2]])

        x = torch.cat([skip, x], dim=1) # channel-wise concat
        return self.conv(x)

class ASPP(nn.Module):
    # analyses image at multiple scales in parralell.
    # in theory, it helps to detects both small and large SCDs

    def __init__(self, in_ch, out_ch, dilations=(1, 3, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.project = nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))
    
class _AugmentedSubset(Dataset):
    """Wraps a subset of dataset with its own augment flag."""

    def __init__(self, dataset, indices: list, augment: bool):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original = self.dataset.augment
        self.dataset.augment = self.augment
        try:
            return self.dataset[self.indices[idx]]
        finally:
            self.dataset.augment = original