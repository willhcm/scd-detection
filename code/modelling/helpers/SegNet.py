from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, maximum_filter
from datetime import datetime
from scipy.ndimage import label, center_of_mass


# similar to all previous models used, just written up formally to import into colab with ease for cross-validation
# model comparison

class DoubleConv(nn.Module):
    """Back bone UNET. 2 * (Conv2d + Batch + Relu)"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ChannelAttention(nn.Module):

    # learns which channels are the most important. very helpful, but may struggle
    # if trained on one region (where one channel is more dominant), and tested on another
    # may hurt cross-geography generalisation
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.avg(x).squeeze(-1).squeeze(-1))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class Down(nn.Module):
    """MaxPool2x2 followed by DoubleConv, one encoder step."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate skip, DoubleConv. one decoder step"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # in_ch comes from (upsampled features + skip features)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

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
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.project = nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))


# combined
class SegNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        # Bottleneck, pool down then ASPP for *some* scale invariance
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(f * 8, f * 16)
        self.bottleneck_attn = ChannelAttention(f * 16)
        self.bottleneck_drop = nn.Dropout2d(p=0.2)

        # Decoder
        self.dec4 = Up(f * 16 + f * 8, f * 8)
        self.dec3 = Up(f * 8 + f * 4, f * 4)
        self.dec2 = Up(f * 4 + f * 2, f * 2)
        self.dec1 = Up(f * 2 + f, f)

        self.out_conv = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # bottleneck
        b = self.pool(s4)
        b = self.bottleneck_drop(b) # dropout
        b = self.aspp(b) # sees several scales
        b = self.bottleneck_attn(b) # what channels are important?

        x = self.dec4(b, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.out_conv(x)
    

# GenAI assistance in creating this function, whcih takes in the predicted mask and compares the mask object centroids to
# ground truth mask, if the predicted mask centroid is in the ground truth mask, it qualifies as a true positive.
# only one prediction allowed per ground truth object. similar to CentreNet function but is comparing mask agaisnt mask.
# has to derive centroid from mask first
def object_centroid_metrics(pred_mask, gt_mask):
    """
    A prediction is counted as a true positive if the centroid of a predicted object
    falls inside an unmatched ground-truth object.
    """

    pred_labels, n_pred = label(pred_mask > 0)
    gt_labels, n_gt = label(gt_mask > 0)

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_id in range(1, n_pred + 1):

        # gets pred_labels[pred_id]
        pred_obj = pred_labels == pred_id

        min_area = 20 # 20 pixel area prediction likely noise
        if pred_obj.sum() < min_area:
          continue

        # centroid of predicted blob
        cy, cx = center_of_mass(pred_obj)

        if np.isnan(cx) or np.isnan(cy):
            continue

        r = int(round(cy))
        c = int(round(cx))

        # ensure inside image
        r = np.clip(r, 0, gt_mask.shape[0] - 1)
        c = np.clip(c, 0, gt_mask.shape[1] - 1)

        gt_id = gt_labels[r, c]

        if gt_id == 0:
            fp += 1
            continue

        # already matched
        if gt_id in matched_gt:
            fp += 1
            continue

        matched_gt.add(gt_id)
        tp += 1

    fn = n_gt - tp

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, f1


# custom loss for segmentation. 
# TV for total variation (encourages smooth shapes in prediction, doesn't see target.
# Dice for IoU type loss (on sigmoided predictions), BCE for logit-based confidence loss etc.
class TVLoss(nn.Module):
    """
    Total Variation loss penalises pixel roughness in the predicted
    probability map.
    """

    def __init__(self, weight: float = 1e-2):
        super().__init__()
        self.weight = weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        diff_h = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
        diff_w = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
        return self.weight * (diff_h + diff_w)

class DiceLoss(nn.Module):
    """Soft Dice, optimises foreground overlap (IoU-type loss)"""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

class CombinedLoss(nn.Module):
    """
    BCE + Dice + TV loss.

    alpha -> weight on dice
    pos_weight -> weighting on positive class in BCE
    tv_weight -> multiplier for tv loss
    """
    def __init__(
        self,
        alpha = 0.6,
        pos_weight = 5.0,
        tv_weight = 5e-3,
    ):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.tv = TVLoss(weight=tv_weight)

    def forward(self, logits, targets):

        return (self.alpha * self.dice(logits, targets) + (1 - self.alpha) * self.bce(logits, targets)
            + self.tv(logits, targets))


def SN_train(model, criterion, train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model = model.to(device)
    EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
    )

    # unique checkpointing to save best model
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M') 
    CHECKPOINT_DIR = Path(f'/content/drive/MyDrive/IRP/models/checkpoints/{RUN_ID}')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # initialise metrics
    vls, tls, ious, accs, precisions, recalls, f1s = [], [], [], [], [], [], []
    best_iou = 0.0
    obj_precisions, obj_recalls, obj_f1s = [], [], []

    for epoch in range(EPOCHS):

        # Train
        model.train()

        # epoch level loss
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            logits = model(images)

            # BCE + DICE + TV loss
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()

        # init epoch level val metrics
        iou_tp = iou_fp = iou_fn = correct = total = intersection = union = tp = fp = fn = val_loss = 0.0


        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device).unsqueeze(1).float()

                logits = model(images)
                val_loss += criterion(logits, masks).item()

                # threshold at 0.75, from precision-recall curve
                preds = (torch.sigmoid(logits) >= 0.75).float()

                # object metrics
                pred_np = preds.cpu().numpy()
                mask_np = masks.cpu().numpy()

                # is centroid of prediction in ground truth label for that object?
                for i in range(pred_np.shape[0]):
                    iou_tp_i, iou_fp_i, iou_fn_i, _, _, _ = object_centroid_metrics(
                        pred_np[i, 0],
                        mask_np[i, 0]
                    )
                    iou_tp += iou_tp_i
                    iou_fp += iou_fp_i
                    iou_fn += iou_fn_i

                correct += (preds == masks).sum().item()
                total += masks.numel()
                intersection += (preds * masks).sum().item()
                union += (preds + masks - preds * masks).sum().item()
                tp += (preds * masks).sum().item()
                fp += (preds * (1 - masks)).sum().item()
                fn += ((1 - preds) * masks).sum().item()

        scheduler.step()

        # metrics

        iou_precision = (iou_tp + 1e-6) / (iou_tp + iou_fp + 1e-6)
        iou_recall = (iou_tp + 1e-6) / (iou_tp + iou_fn + 1e-6)
        iou_f1 = 2 * iou_precision * iou_recall / (iou_precision + iou_recall + 1e-6)

        acc = 100 * correct / total
        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        vls.append(avg_val)
        tls.append(avg_train)
        ious.append(iou)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        obj_precisions.append(iou_precision)
        obj_recalls.append(iou_recall)
        obj_f1s.append(iou_f1)

        # checkpointing
        if iou > best_iou:
            best_iou = iou

            for old in CHECKPOINT_DIR.glob('best_*.pt'):
                old.unlink()

            torch.save(model.state_dict, CHECKPOINT_DIR / f'best_epoch{epoch+1:03d}_iou{iou:.4f}.pt')
            print(f'Best checkpoint saved (iou {iou:.4f})')

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"train {avg_train:.4f} | "
                f"val {avg_val:.4f} | "
                f"acc {acc:.2f}% | "
                f"iou {iou:.4f} | "
                f"P {precision:.4f} | "
                f"R {recall:.4f} | "
                f"F1 {f1:.4f} | "
                f"Obj_P {iou_precision:.4f} | "
                f"Obj_R {iou_recall:.4f} | "
                f"Obj_F1 {iou_f1:.4f} | ")
        
    return vls, tls, obj_precisions, obj_recalls, obj_f1s, model