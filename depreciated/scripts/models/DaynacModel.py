# recreating the model from Daynac 2026 (manuscript in pre-print), which uses Segmentation style with same input channels as 
# the models I am developing. Using as a literature-based baseline to compare performance with object-based models (my CentreNet-type model)

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOT MY MODEL, RECONSTRUCTED FROM A MANUSCRIPT TO USE AS A BASELINE COMPARISON FOR MY WORK.
# AI RECONSTRUCTION OF EXACT MODEL FROM MANUSCRIPT (Daynac et al, 2026) in preprint.
class BaselineConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = BaselineConvBlock(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels
        )

    @staticmethod
    def _match_size(
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2
                ]
            )

        return x

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor
    ) -> torch.Tensor:

        x = self.up(x)
        x = self._match_size(x, skip)

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)

        return x


class BaselineUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1
    ):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.enc1 = BaselineConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = BaselineConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = BaselineConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = BaselineConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # -------------------------
        # Bottleneck
        # -------------------------
        self.bottleneck = BaselineConvBlock(512, 512)

        # -------------------------
        # Decoder
        # -------------------------
        self.up4 = BaselineUpBlock(
            in_channels=512,
            skip_channels=512,
            out_channels=256
        )

        self.up3 = BaselineUpBlock(
            in_channels=256,
            skip_channels=256,
            out_channels=128
        )

        self.up2 = BaselineUpBlock(
            in_channels=128,
            skip_channels=128,
            out_channels=64
        )

        self.up1 = BaselineUpBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=64
        )

        # Binary segmentation output
        self.output_conv = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)                 # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))    # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool2(e2))    # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool3(e3))    # [B, 512, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(
            self.pool4(e4)
        )                                 # [B, 512, H/16, W/16]

        # Decoder
        d4 = self.up4(b, e4)              # [B, 256, H/8, W/8]
        d3 = self.up3(d4, e3)             # [B, 128, H/4, W/4]
        d2 = self.up2(d3, e2)             # [B, 64, H/2, W/2]
        d1 = self.up1(d2, e1)             # [B, 64, H, W]

        logits = self.output_conv(d1)      # [B, 1, H, W]

        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns pixel-wise probabilities in [0, 1].
        """
        return torch.sigmoid(self.forward(x))


# Loss used in the paper, assumed 50-50 weighting

class BCEDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:

        targets = targets.float()

        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape}, "
                f"targets {targets.shape}"
            )

        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)

        # Dice calculated independently for each image, then averaged.
        dims = (1, 2, 3)

        intersection = (probabilities * targets).sum(dim=dims)
        denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        dice_loss = 1.0 - dice_score.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss