from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, maximum_filter
from datetime import datetime
from scipy.ndimage import label, center_of_mass

# recently developed in colab, changed from simple UNET used previously but with two extra prediction heads, different labels.

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


class SCDCentreNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f,f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        # Bottleneck, pool down then ASPP
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(f * 8, f * 16)
        # self.bottleneck_attn = ChannelAttention(f * 16)
        self.bottleneck_drop = nn.Dropout2d(p=0.4)

        # decoder
        self.dec4 = Up(f * 16 + f * 8, f * 8)
        self.dec3 = Up(f * 8  + f * 4, f * 4)
        self.dec2 = Up(f * 4  + f * 2, f * 2)
        self.dec1 = Up(f * 2  + f, f)

        # centroid head, extra conv for task-specific expression
        self.centroid_head = nn.Conv2d(f, 1, 1)

        # radius head
        self.radius_head = nn.Conv2d(f, 1, 1)

        # offset head (prediction is rounded to nearest pixel)
        self.offset_head = nn.Conv2d(f, 2, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # bottleneck
        b = self.pool(s4)
        b = self.bottleneck_drop(b) # dropout
        b = self.aspp(b) # model sees several scales
      #  b = self.bottleneck_attn(b)

        x = self.dec4(b,  s4)
        x = self.dec3(x,  s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)

        centre = self.centroid_head(x)
        radius = self.radius_head(x)
        offset = self.offset_head(x)

        return centre, radius, offset
    

# get derived targets for CentreNet
def build_centernet_targets(mask, min_sigma=4, max_sigma=30, base_weight=1.0):
    labeled, n = label(mask > 0)
    h, w = mask.shape

    heatmap = np.zeros((h, w), dtype=np.float32)
    weights = np.ones((h, w), dtype=np.float32) * base_weight
    offset  = np.zeros((2, h, w), dtype=np.float32)
    radius  = np.zeros((h, w), dtype=np.float32)
    obj_mask = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.indices((h, w))

    for i in range(1, n + 1):
        obj = labeled == i
        cy, cx = center_of_mass(obj)
        cy_r, cx_r = round(cy), round(cx)

        area = obj.sum()
        sigma = np.clip(np.sqrt(area) * 0.15, min_sigma, max_sigma)

        g = np.exp(-((yy - cy_r) ** 2 + (xx - cx_r) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, g)

        blob_weight = 20 
        weights[cy_r, cx_r] = blob_weight # only single centroid pixel weighted up. 
        
        offset[0, cy_r, cx_r] = cy - cy_r
        offset[1, cy_r, cx_r] = cx - cx_r
        radius[cy_r, cx_r] = np.sqrt(area / np.pi)
        obj_mask[cy_r, cx_r] = 1.0

    return heatmap, weights, offset, radius, obj_mask
    


# GenAI assistance in creating this function, whcih takes in the predicted heatmap and compares the centroid predictions tp
# ground truth mask, if the predicted centroid is in the ground truth mask, it qualifies as a true positive.
# only one prediction allowed per ground truth object


def centroid_heatmap_metrics(
    pred_heatmap,
    gt_mask,
    threshold=0.6,
    min_distance=8
):

    gt_labels, n_gt = label(gt_mask > 0)

    # Find local maximu, in the predicted heatmap
    local_max = pred_heatmap == maximum_filter(
        pred_heatmap,
        size=2 * min_distance + 1
    )

    peaks = local_max & (pred_heatmap >= threshold)

    peak_coords = np.argwhere(peaks)

    # Sort strongest peaks first
    peak_scores = pred_heatmap[peaks]
    order = np.argsort(peak_scores)[::-1]
    peak_coords = peak_coords[order]

    matched_gt = set()

    tp = 0
    fp = 0

    # matching using id (using unique x, y coords )
    for r, c in peak_coords:
        gt_id = gt_labels[r, c]

        if gt_id == 0:
            fp += 1
            continue

        if gt_id in matched_gt:
            fp += 1
            continue

        matched_gt.add(gt_id)
        tp += 1

    fn = n_gt - tp

    # calc metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1

# custom loss function (adapted from CentreNet paper (reference to come), with assistance from Claude Sonnet 5)

class CentreNetLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0, eps=1e-6, off_weight=1.0, rad_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.off_weight = off_weight
        self.rad_weight = rad_weight

    def forward(self, pred_hm, pred_rad, pred_off, hm, rad, off, obj_mask, weight=None):
        pred = torch.sigmoid(pred_hm).clamp(self.eps, 1 - self.eps)

        pos_inds = (hm == 1).float()
        neg_inds = (hm < 1).float()
        neg_weights = (1 - hm) ** self.beta

        pos_loss = -torch.log(pred) * (1 - pred) ** self.alpha * pos_inds
        neg_loss = -torch.log(1 - pred) * pred ** self.alpha * neg_weights * neg_inds

        if weight is not None:
            pos_loss = pos_loss * weight

        num_pos = pos_inds.sum(dim=(1, 2, 3))
        num_neg = neg_inds.sum(dim=(1, 2, 3)).clamp(min=1)
        pos_sum = pos_loss.sum(dim=(1, 2, 3))
        neg_sum = neg_loss.sum(dim=(1, 2, 3))

        per_img = torch.where(num_pos > 0, (pos_sum + neg_sum) / num_pos.clamp(min=1), neg_sum / num_neg)
        hm_loss = per_img.mean()

        mask = obj_mask.unsqueeze(1)
        rad = rad.unsqueeze(1)

        off_loss = F.l1_loss(pred_off * mask, off * mask, reduction='sum') / (mask.sum() + self.eps)
        rad_loss = F.l1_loss(pred_rad * mask, rad * mask, reduction='sum') / (mask.sum() + self.eps)

        total = hm_loss + self.off_weight * off_loss + self.rad_weight * rad_loss
        return total, hm_loss, off_loss, rad_loss


# training function for CentreNet

def CN_train(model, criterion, train_loader, val_loader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)
    nn.init.constant_(model.centroid_head.bias, -2.19)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-6)

    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M')
    CHECKPOINT_DIR = Path(f'/content/drive/MyDrive/IRP/models/checkpoints/{RUN_ID}')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vls, tls, precisions, recalls, f1s = [], [], [], [], []

    best_f1 = -1 

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks, centre_map, weights, offset, radius, obj_mask in train_loader:
            images = images.to(device)
            centre_map = centre_map.to(device).unsqueeze(1).float()
            weights = weights.to(device).unsqueeze(1)
            offset = offset.to(device)
            radius = radius.to(device)
            obj_mask = obj_mask.to(device)

            optimizer.zero_grad()
            pred_centres, pred_radius, pred_offset = model(images)

            loss, _, _, _ = criterion(
                pred_centres, pred_radius, pred_offset,
                centre_map, radius, offset, obj_mask,
                weight=weights
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        tp = fp = fn = 0

        with torch.no_grad():
            for images, masks, centre_map, weights, offset, radius, obj_mask in val_loader:
                images = images.to(device)
                centre_map = centre_map.to(device).unsqueeze(1).float()
                weights = weights.to(device).unsqueeze(1)
                offset = offset.to(device)
                radius = radius.to(device)
                obj_mask = obj_mask.to(device)

                pred_centres, pred_radius, pred_offset = model(images)

                vloss, _, _, _ = criterion(
                    pred_centres, pred_radius, pred_offset,
                    centre_map, radius, offset, obj_mask,
                    weight=weights
                )
                val_loss += vloss.item()

                pred_np = torch.sigmoid(pred_centres).cpu().numpy()
                mask_np = masks.cpu().numpy()

                for i in range(pred_np.shape[0]):
                    tp_i, fp_i, fn_i, precision, recall, f1 = centroid_heatmap_metrics(
                        pred_heatmap=pred_np[i, 0],
                        gt_mask=mask_np[i],
                        threshold=0.15,
                        min_distance=8
                    )
                    tp += tp_i
                    fp += fp_i
                    fn += fn_i

        scheduler.step()

        precision = (tp + 1e-6) / (tp + fp + 1e-6) if tp > 0 else 0
        recall = (tp + 1e-6) / (tp + fn + 1e-6) if tp > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        vls.append(avg_val)
        tls.append(avg_train)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # checkpointing
        if f1 > best_f1:
            best_f1 = f1

            for old in CHECKPOINT_DIR.glob('best_*.pt'):
                old.unlink()

            torch.save(model.state_dict, CHECKPOINT_DIR / f'best_epoch{epoch+1:03d}_f1{f1:.4f}.pt')
            print(f'Best checkpoint saved (f1 {f1:.4f})')


        print(f"Epoch {epoch+1:3d}/{EPOCHS} | train {avg_train:.4f} | val {avg_val:.4f} | P {precision:.4f} | R {recall:.4f} | F1 {f1:.4f}")

    return vls, tls, precisions, recalls, f1s, model