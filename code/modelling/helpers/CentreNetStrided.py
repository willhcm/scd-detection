from pathlib import Path
from datetime import datetime
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import label, center_of_mass, maximum_filter, gaussian_filter, zoom

_BANDS_TO_LOAD = ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']

class DoubleConv(nn.Module):
    """UNet block: 2 * (Conv2d + GroupNorm + ReLU).""" # groupnorm vs batch norm: batch norm normalising images independently based on channel, not at Group-level, whcih can contain different regions!
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


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x).squeeze(-1).squeeze(-1))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class Down(nn.Module):
    """MaxPool2d followed by DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Bilinear upsample, concatenate skip, DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.pad(
                x,
                [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8)): # less aggressive to retain finer detail.
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        self.project = nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False)

    def forward(self, x):
        return self.project(torch.cat([b(x) for b in self.branches], dim=1))

def _make_head(in_ch, out_ch, p=0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, in_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p),
        nn.Conv2d(in_ch, out_ch, 1)
    )

class SCDCentreNet(nn.Module):
    """
    Edited for a 128x128 prediction head, same as official CentreNet. 
    """
    def __init__(self, in_channels=4, base_filters=32):
        super().__init__()
        f = base_filters

        self.enc1 = DoubleConv(in_channels, f)       # 512
        self.enc2 = Down(f, f * 2)                   # 256
        self.enc3 = Down(f * 2, f * 4)               # 128
        self.enc4 = Down(f * 4, f * 8)               # 64

        self.pool = nn.MaxPool2d(2)                  # 32
        self.aspp = ASPP(f * 8, f * 16)
        self.bottleneck_attn = ChannelAttention(f * 16)
        self.bottleneck_drop = nn.Dropout2d(p=0.4)

        self.dec4 = Up(f * 16 + f * 8, f * 8)        # 64
        self.dec3 = Up(f * 8 + f * 4, f * 4)         # 128

        # Heads operate at 128x128 for 512x512 inputs.
        self.head_drop = nn.Dropout2d(0.2)
        self.centroid_head = _make_head(f * 4, 1, p=0.2)
        self.radius_head = _make_head(f * 4, 1, p=0.2)
        self.offset_head = _make_head(f * 4, 2, p=0.2)

        # basic FPN test

        self.lat_b  = nn.Conv2d(f * 16, f * 4, 1)
        self.lat_s4 = nn.Conv2d(f * 8,  f * 4, 1)
        self.lat_s3 = nn.Conv2d(f * 4,  f * 4, 1)

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(f * 4, f * 4, 3, padding=1, bias=False),
            nn.GroupNorm(8, f * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.pool(s4)
        b = self.bottleneck_drop(b)
        b = self.aspp(b)

        # FPN 

        x = self.dec4(b, s4)
        x = self.dec3(x, s3)

        x = self.head_drop(x)

        centre = self.centroid_head(x)
        radius = self.radius_head(x)
        offset = self.offset_head(x)
        return centre, radius, offset


# ChatGPT assistanve with this function, edited from previous function but now 
def build_centernet_targets(
    mask,
    stride=4,
    min_sigma=1.0,
    max_sigma=6.0,
    base_weight=1.0,
    centre_weight=5.0,
):
    """
    Build stride-4 CentreNet targets from a full-resolution binary mask.

    Returns low-resolution targets:
      heatmap:  [H/stride, W/stride]
      weights:  [H/stride, W/stride]
      offset:   [2, H/stride, W/stride], fractional offset on low-res grid
      radius:   [H/stride, W/stride], radius in low-res pixels
      obj_mask: [H/stride, W/stride], 1 only at centre cells
    """
    labeled, n = label(mask > 0)
    h, w = mask.shape
    oh, ow = h // stride, w // stride

    heatmap = np.zeros((oh, ow), dtype=np.float32)
    weights = np.ones((oh, ow), dtype=np.float32) * base_weight
    offset = np.zeros((2, oh, ow), dtype=np.float32)
    radius = np.zeros((oh, ow), dtype=np.float32)
    obj_mask = np.zeros((oh, ow), dtype=np.float32)

    yy, xx = np.indices((oh, ow))

    for i in range(1, n + 1):
        obj = labeled == i
        area = float(obj.sum())
        if area <= 0:
            continue

        cy, cx = center_of_mass(obj)
        cy_low = cy / stride
        cx_low = cx / stride
        cy_i = int(np.floor(cy_low))
        cx_i = int(np.floor(cx_low))

        if not (0 <= cy_i < oh and 0 <= cx_i < ow):
            continue

        # Keep the same physical Gaussian size as before, but expressed on the low-res grid.
        sigma_full = np.clip(np.sqrt(area) * 0.15, 4.0, 30.0)
        sigma_low = np.clip(sigma_full / stride, min_sigma, max_sigma)

        g = np.exp(-((yy - cy_i) ** 2 + (xx - cx_i) ** 2) / (2 * sigma_low ** 2))
        heatmap = np.maximum(heatmap, g.astype(np.float32))
        heatmap[cy_i, cx_i] = 1.0

        weights[cy_i, cx_i] = centre_weight
        offset[0, cy_i, cx_i] = cy_low - cy_i
        offset[1, cy_i, cx_i] = cx_low - cx_i
        radius[cy_i, cx_i] = np.sqrt(area / np.pi) / stride
        obj_mask[cy_i, cx_i] = 1.0

    return heatmap, weights, offset, radius, obj_mask


class CenterNetLoss(nn.Module):
    def __init__(
        self,
        alpha=2.0,
        beta=4.0,
        eps=1e-6,
        off_weight=0.2,
        rad_weight=0.1,
        neg_scale=1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.off_weight = off_weight
        self.rad_weight = rad_weight
        self.neg_scale = neg_scale
        self.mse = nn.MSELoss()

    def forward(self, pred_hm, pred_rad, pred_off, hm, rad, off, obj_mask, weight=None):
        pred = torch.sigmoid(pred_hm).clamp(self.eps, 1 - self.eps)

        pos_inds = hm.eq(1.0).float()
        neg_inds = hm.lt(1.0).float()
        neg_weights = (1.0 - hm).pow(self.beta)

        pos_loss = -torch.log(pred) * (1.0 - pred).pow(self.alpha) * pos_inds
        neg_loss = -torch.log(1.0 - pred) * pred.pow(self.alpha) * neg_weights * neg_inds

        # Weight only the true centre positive cells.
        if weight is not None:
            pos_loss = pos_loss * weight

        num_pos = pos_inds.sum(dim=(1, 2, 3))
        num_neg = neg_inds.sum(dim=(1, 2, 3)).clamp(min=1)
        pos_sum = pos_loss.sum(dim=(1, 2, 3))
        neg_sum = neg_loss.sum(dim=(1, 2, 3))

        per_img = torch.where(
            num_pos > 0,
            (pos_sum + self.neg_scale * neg_sum) / num_pos.clamp(min=1),
            self.neg_scale * neg_sum / num_neg,
        )
        hm_loss = per_img.mean()

        mask = obj_mask.unsqueeze(1).float()
        rad = rad.unsqueeze(1)

        off_loss = F.l1_loss(pred_off * mask, off * mask, reduction='sum') / (mask.sum() + self.eps)
        rad_loss = F.l1_loss(pred_rad * mask, rad * mask, reduction='sum') / (mask.sum() + self.eps)

        total = hm_loss + self.off_weight * off_loss + self.rad_weight * rad_loss
        return total, hm_loss, off_loss, rad_loss

# ChatGPT assistance with the function. decodes downsampled 128x128 back to 512x512.
# prediction is done at 128 as otherise negative pixels in loss function dominate.
# this method performs better than downsampling negative pixels in loss directly.
def decode_centernet_predictions(pred_hm, pred_rad=None, pred_off=None, threshold=0.3, min_distance=8, stride=4):
    """
    Decode one low-res prediction into full-resolution centre coordinates.

    pred_hm: [Hlow, Wlow], already sigmoid probabilities
    pred_rad: optional [Hlow, Wlow] or [1, Hlow, Wlow], radius in low-res pixels
    pred_off: optional [2, Hlow, Wlow], offset on low-res grid

    Returns list of dicts: {cy, cx, radius, score, low_r, low_c}
    """
    local_max = pred_hm == maximum_filter(pred_hm, size=2 * min_distance + 1)
    peaks = local_max & (pred_hm >= threshold)
    peak_coords = np.argwhere(peaks)
    if len(peak_coords) == 0:
        return []

    scores = pred_hm[peaks]
    order = np.argsort(scores)[::-1]
    peak_coords = peak_coords[order]
    scores = scores[order]

    out = []
    for (r, c), score in zip(peak_coords, scores):
        if pred_off is not None:
            off_y = float(pred_off[0, r, c])
            off_x = float(pred_off[1, r, c])
        else:
            off_y = off_x = 0.5

        cy_full = (float(r) + off_y) * stride
        cx_full = (float(c) + off_x) * stride

        if pred_rad is not None:
            rr = pred_rad[0] if pred_rad.ndim == 3 else pred_rad
            radius_full = float(rr[r, c]) * stride
        else:
            radius_full = np.nan

        out.append({
            'cy': cy_full,
            'cx': cx_full,
            'radius': radius_full,
            'score': float(score),
            'low_r': int(r),
            'low_c': int(c),
        })
    return out


def centroid_heatmap_metrics(pred_heatmap, gt_mask, pred_offset=None, threshold=0.3, min_distance=8, stride=4):
    """
    Object-level metric for low-res heatmaps. A prediction is TP if its decoded
    full-res centre lies inside an unmatched ground truth object.
    """
    gt_labels, n_gt = label(gt_mask > 0)
    detections = decode_centernet_predictions(
        pred_hm=pred_heatmap,
        pred_off=pred_offset,
        threshold=threshold,
        min_distance=min_distance,
        stride=stride,
    )

    matched_gt = set()
    tp = fp = 0

    h, w = gt_mask.shape
    for det in detections:
        r = int(round(det['cy']))
        c = int(round(det['cx']))
        if r < 0 or r >= h or c < 0 or c >= w:
            fp += 1
            continue

        gt_id = int(gt_labels[r, c])
        if gt_id == 0 or gt_id in matched_gt:
            fp += 1
            continue

        matched_gt.add(gt_id)
        tp += 1

    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return tp, fp, fn, precision, recall, f1

# adapted from CentreNet.py for strided model.
def CN_train(model, criterion, train_loader, val_loader, epochs, stride=4, threshold=0.3, min_distance=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    run_id = datetime.now().strftime('%Y%m%d_%H%M')
    checkpoint_dir = Path(f'/content/drive/MyDrive/IRP/models/checkpoints/{run_id}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vls, tls, precisions, recalls, f1s = [], [], [], [], []
    best_f1 = 0.1
    patience = 15
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks, centre_map, weights, offset, radius, obj_mask in train_loader:
            images = images.to(device).float()
            centre_map = centre_map.to(device).unsqueeze(1).float()
            weights = weights.to(device).unsqueeze(1).float()
            offset = offset.to(device).float()
            radius = radius.to(device).float()
            obj_mask = obj_mask.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            pred_centres, pred_radius, pred_offset = model(images)
            loss, _, _, _ = criterion(
                pred_centres, pred_radius, pred_offset,
                centre_map, radius, offset, obj_mask,
                weight=weights,
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        tp = fp = fn = 0

        with torch.no_grad():
            for images, masks, centre_map, weights, offset, radius, obj_mask in val_loader:
                images = images.to(device).float()
                centre_map = centre_map.to(device).unsqueeze(1).float()
                weights = weights.to(device).unsqueeze(1).float()
                offset = offset.to(device).float()
                radius = radius.to(device).float()
                obj_mask = obj_mask.to(device).float()

                pred_centres, pred_radius, pred_offset = model(images)
                vloss, _, _, _ = criterion(
                    pred_centres, pred_radius, pred_offset,
                    centre_map, radius, offset, obj_mask,
                    weight=weights,
                )
                val_loss += vloss.item()

                pred_hm_np = torch.sigmoid(pred_centres).cpu().numpy()
                pred_off_np = pred_offset.cpu().numpy()
                mask_np = masks.cpu().numpy()

                for i in range(pred_hm_np.shape[0]):
                    tp_i, fp_i, fn_i, _, _, _ = centroid_heatmap_metrics(
                        pred_heatmap=pred_hm_np[i, 0],
                        gt_mask=mask_np[i],
                        pred_offset=pred_off_np[i],
                        threshold=threshold,
                        min_distance=min_distance,
                        stride=stride,
                    )
                    tp += tp_i
                    fp += fp_i
                    fn += fn_i

        scheduler.step()

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        avg_train = train_loss / max(1, len(train_loader))
        avg_val = val_loss / max(1, len(val_loader))

        tls.append(avg_train)
        vls.append(avg_val)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            bad_epochs = 0
            for old in checkpoint_dir.glob('best_*.pt'):
                old.unlink()
            torch.save(model.state_dict(), checkpoint_dir / f'best_epoch{epoch + 1:03d}_f1{f1:.4f}.pt')
            print(f'Best checkpoint saved (f1 {f1:.4f})')
        else:
            bad_epochs += 1

        

        print(
            f'Epoch {epoch + 1:3d}/{epochs} | '
            f'train {avg_train:.4f} | val {avg_val:.4f} | '
            f'P {precision:.4f} | R {recall:.4f} | F1 {f1:.4f}'
        )

        if bad_epochs >= patience:
            print("early stopping")
            break

    return vls, tls, precisions, recalls, f1s, model

# per-tile normalisation introduced, old norm redundant.
class CentreNetDataset(Dataset):
    def __init__(
        self,
        all_paths,
        norm_stats: dict = None,
        tile_size: int = 512,
        skip_partial: bool = True,
        augment: bool = False,
        norm_type='tile',
        stride: int = 4,
    ):
        self.augment = augment
        self.stats = norm_stats
        self.paths = []
        self.norm_type = norm_type
        self.stride = stride

        skipped_shape = 0
        skipped_missing = 0

        for p in sorted(all_paths):
            d = np.load(p, allow_pickle=True)

            if skip_partial:
                _, h, w = d['image'].shape
                if h != tile_size or w != tile_size:
                    skipped_shape += 1
                    continue

            layer_names = list(d['layer_names']) if 'layer_names' in d else []
            if any(b not in layer_names for b in _BANDS_TO_LOAD):
                skipped_missing += 1
                continue

            self.paths.append(p)

        print(f'Found {len(self.paths)} tiles ({skipped_shape} partial, {skipped_missing} missing bands skipped)')

    def __len__(self):
        return len(self.paths)

    def _augment(self, image, mask):
        image, mask = image.copy(), mask.copy()

        if random.random() > 0.5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]
        if random.random() > 0.5:
            image, mask = image[:, ::-1, :], mask[::-1, :]

        k = random.choice([0, 1, 2, 3])
        if k:
            image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, size=image[0].shape).astype(np.float32)
            image[0] = image[0] + noise
        if random.random() > 0.5:
            sigma = random.uniform(0.3, 0.8)
            image[0] = gaussian_filter(image[0], sigma=sigma)
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            image[0] = image[0] * scale

        return image, mask

    def _normalize(self, image, stats):
        for i, name in enumerate(_BANDS_TO_LOAD):
            band = image[i]
            s = stats.get(name) if stats else None

            if name == 'HILLSHADE' and s:
                band = np.log1p(np.clip(band, s['p1'], s['p99']))
                image[i] = (band - s['mean']) / (s['std'] + 1e-6)
            elif name == 'DEM' and s:
                band = np.clip(band, s['p1'], s['p99'])
                image[i] = (band - s['mean']) / (s['std'] + 1e-6)
            elif name == 'DEM_SLOPE' and s:
                band = np.log1p(np.clip(band, s['p1'], s['p99']))
                image[i] = (band - s['mean']) / (s['std'] + 1e-6)
            elif name in ('RR', 'LAPLACE') and s:
                band = np.clip(band, s['p1'], s['p99'])
                image[i] = (band - s['mean']) / (s['std'] + 1e-6)
            else:
                # Fallback to per-tile norm if stats are absent.
                image[i] = (band - band.mean()) / (band.std() + 1e-6)
        return image

    def _resolve_stats(self, path):
        if not self.stats:
            return None
        first_val = next(iter(self.stats.values()))
        if isinstance(first_val, dict) and 'mean' not in first_val:
            region = Path(path).parent.name
            return self.stats.get(region, self.stats.get('Russia'))
        return self.stats

    def __getitem__(self, idx):
        path = self.paths[idx]
        d = np.load(path, allow_pickle=True)

        layer_names = list(d['layer_names'])
        li = {name: i for i, name in enumerate(layer_names)}
        band_indices = [li[b] for b in _BANDS_TO_LOAD]

        image = d['image'][band_indices].astype(np.float32)
        mask = d['labels'].astype(np.float32)


        for i in range(len(_BANDS_TO_LOAD)):
            band = image[i]
            image[i] = (band - band.mean()) / (band.std() + 1e-6)

        # image = self._normalize(image, self._resolve_stats(path))

        if self.augment:
            image, mask = self._augment(image, mask)

        centre_map, weights, offset, radius, obj_mask = build_centernet_targets(mask, stride=self.stride)

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            torch.from_numpy(np.ascontiguousarray(centre_map)),
            torch.from_numpy(np.ascontiguousarray(weights)),
            torch.from_numpy(np.ascontiguousarray(offset)),
            torch.from_numpy(np.ascontiguousarray(radius)),
            torch.from_numpy(np.ascontiguousarray(obj_mask)),
        )
