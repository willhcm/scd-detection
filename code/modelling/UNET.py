# small unet, using pretrained encoder from torchvision.models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import random
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import label, find_objects
from torchvision.ops import box_iou

_BANDS_TO_LOAD = ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']

class ResFiLM(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.film = nn.Linear(1, cout * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, res):
        gamma, beta = self.film(res.view(-1, 1)).chunk(2, dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return (1 + gamma[:, :, None, None]) * x + beta[:, :, None, None]

class Down(nn.Module):
    def __init__(self, body, ch):
        super().__init__()
        self.body = body
        self.film = ResFiLM(ch, ch)

    def forward(self, x, res):
        return self.film(self.body(x), res)

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.film = ResFiLM(cout, cout)

    def forward(self, x, skip, res):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.film(x + skip, res)


def widen_conv1(net, n):
    old = net.conv1.weight.data
    new = old.mean(dim=1, keepdim=True).repeat(1, n, 1, 1)
    k = min(n, 3)
    new[:, :k] = old[:, :k]
    conv = nn.Conv2d(n, 64, 7, stride=2, padding=3, bias=False)
    conv.weight.data = new
    net.conv1 = conv


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        widen_conv1(net, in_channels)
        c = list(net.children())

        self.enc1 = Down(nn.Sequential(*c[:3]), 64)
        self.enc2 = Down(nn.Sequential(*c[3:5]), 64)
        self.enc3 = Down(c[5], 128)
        self.enc4 = Down(c[6], 256)
        self.enc5 = Down(c[7], 512)

        self.dec4 = Up(512, 256)
        self.dec3 = Up(256, 128)
        self.dec2 = Up(128, 64)
        self.dec1 = Up(64, 64)
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, res):
        e1 = self.enc1(x, res)
        e2 = self.enc2(e1, res)
        e3 = self.enc3(e2, res)
        e4 = self.enc4(e3, res)
        e5 = self.enc5(e4, res)

        d = self.dec4(e5, e4, res)
        d = self.dec3(d, e3, res)
        d = self.dec2(d, e2, res)
        d = self.dec1(d, e1, res)

        out = self.head(d)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return torch.sigmoid(out)

class UNETDataset(Dataset):
    
    def __init__(self, all_paths, norm_stats, augment=True, min_instance_area=25,):

        self.augment = augment
        self.paths = all_paths
        self.min_instance_area = min_instance_area
        self.norm_stats = norm_stats
        self.seen_resolutions = []
        self.object_sizes = []
        self.real_object_diameters = []

    def __len__(self):
        return len(self.paths)

    def handle_outliers(self, image):
        for i, band in enumerate(image):
            ok = np.isfinite(band)
            if not ok.any():
                raise ValueError(f"band {i} has no finite pixels")
            band = np.where(ok, band, np.median(band[ok]))
            lo, hi = np.percentile(band, [0.5, 99.5])
            image[i] = np.clip(band, lo, hi)
        return image

    def _crop_zoom(self, image, mask, res):
        h, w = mask.shape
        s = random.uniform(0.7, 0.9)
        ch, cw = int(h * s), int(w * s)
        y = random.randint(0, h - ch)
        x = random.randint(0, w - cw)

        image = image[:, y:y + ch, x:x + cw]
        mask = mask[y:y + ch, x:x + cw]

        image = np.stack([
            cv2.resize(np.ascontiguousarray(c), (w, h),
                       interpolation=cv2.INTER_LINEAR)
            for c in image
        ])
        mask = cv2.resize(np.ascontiguousarray(mask), (w, h),
                          interpolation=cv2.INTER_NEAREST)

        return image, mask, res * s

    def spatial_augment(self, image, mask, res):

        if random.random() > 0.5:
            image, mask = image[:, :, ::-1], mask[:, ::-1]

        if random.random() > 0.5:
            image, mask = image[:, ::-1, :], mask[::-1, :]

        k = random.randint(0, 3)
        if k:
            image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)
        image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)

        if random.random() < 0.5:
            image, mask, res = self._crop_zoom(image, mask, res)

        return image, mask, res

    def _normalise(self, image, li, band_indices):

        # DEM
        dem = image[band_indices.index(li["DEM"])]

        valid = np.isfinite(dem)
        if not valid.any():
            raise ValueError("DEM contains no finite pixels")

        dem_median = np.median(dem[valid])
        dem_rel = np.where(valid, dem, dem_median) - dem_median

        # slope log scaled, then Robust scaling
        slope = image[band_indices.index(li["DEM_SLOPE"])]
        rr = image[band_indices.index(li["RR"])]
        laplace = image[band_indices.index(li["LAPLACE"])]

        slope = np.power(slope, 0.25)

        # Robust scaling all (more efficient):
        for band in ['DEM', 'SLOPE', 'RR', 'LAPLACE']:
            median = self.norm_stats[band]['median']
            iqr = self.norm_stats[band]['iqr']

            if band == 'DEM':
                dem_rel = (dem_rel - median) / iqr
            elif band == 'SLOPE':
                slope = (slope - median) / iqr         
            elif band == 'RR':
                rr = (rr - median) / iqr
            elif band == 'LAPLACE':
                laplace = (laplace - median) / iqr

        return np.stack([dem_rel, slope, rr, laplace], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx], allow_pickle=True)
        res = float(d["resolution"])
        if not np.isfinite(res) or res <= 0:
            raise ValueError(f"bad resolution in {self.paths[idx]}: {res}")

        li = {n: i for i, n in enumerate(d["layer_names"])}
        band_indices = [li[b] for b in _BANDS_TO_LOAD]
        image = d["image"][band_indices].astype(np.float32)
        mask = d["labels"].astype(np.float32)

        image = self.handle_outliers(image)
        if self.augment:
            image, mask, res = self.spatial_augment(image, mask, res)
        image = self._normalise(image, li, band_indices)

        res_value = np.float32(np.log1p(res) / np.log1p(self.norm_stats["max_res"]))
        return image, mask[None], res_value

def evaluate_preds(preds, labels, min_area=25, thr=0.5, iou_thr=0.5):
    preds, labels = preds > thr, labels > thr

    pred_lab, n = label(preds)
    areas = np.bincount(pred_lab.ravel(), minlength=n + 1)
    for i in range(1, n + 1):
        if areas[i] < min_area:
            pred_lab[pred_lab == i] = 0
    preds = pred_lab > 0
    n_pred = int((areas[1:] >= min_area).sum())

    tp = (preds & labels).sum()
    fp = (preds & ~labels).sum()
    fn = (~preds & labels).sum()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    gt_lab, n_gt = label(labels)
    used, obj_tp = set(), 0
    for g in range(1, n_gt + 1):
        gt = gt_lab == g
        best_iou, best_id = 0, None
        for p in np.unique(pred_lab[gt]):
            if p == 0 or p in used:
                continue
            pm = pred_lab == p
            iou = (gt & pm).sum() / (gt | pm).sum()
            if iou > best_iou:
                best_iou, best_id = iou, p
        if best_iou > iou_thr:
            used.add(best_id)
            obj_tp += 1

    obj_fp = n_pred - obj_tp
    obj_fn = n_gt - obj_tp

    return precision, recall, f1, obj_tp, obj_fp, obj_fn