from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from datetime import datetime
import random
from torch.utils.data import Dataset
from depreciated.scripts.misc.blocks import DoubleConv, ASPP, Down, Up
from helpers import object_centroid_metrics, calculate_hillshade

_BANDS_TO_LOAD = ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']

# similar to all previous models used, just written up formally to import into colab with ease for cross-validation
# model comparison

# combined
class SegNet(nn.Module):
    def __init__(self, in_channels=5, base_filters=64):
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
        self.bottleneck_drop = nn.Dropout2d(p=0.2)

       # Decoder
        self.dec4 = Up(f * 16, f * 8, f * 8)
        self.dec3 = Up(f * 8,  f * 4, f * 4)
        self.dec2 = Up(f * 4,  f * 2, f * 2)
        self.dec1 = Up(f * 2,  f, f)

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

        x = self.dec4(b, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.out_conv(x)
    


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


def SN_train(model, criterion, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model = model.to(device)
    EPOCHS = epochs
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
    bad_epochs = 0
    patience = 20

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
            best_state = model.state_dict()
            for old in CHECKPOINT_DIR.glob('best_*.pt'):
                old.unlink()

            torch.save(model.state_dict, CHECKPOINT_DIR / f'best_epoch{epoch+1:03d}_iou{iou:.4f}.pt')
            print(f'Best checkpoint saved (iou {iou:.4f})')
            bad_epochs = 0
        else:
            bad_epochs += 1
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
        
        if bad_epochs >= patience:
            print("early stopping")
            break
    
    model.load_state_dict(best_state)
    
    return vls, tls, obj_precisions, obj_recalls, obj_f1s, model, val_loader


class SegNetDataset(Dataset):

    def __init__(
        self,
        all_paths,
        tile_size: int = 512,
        skip_partial: bool = True,
        augment: bool = False,
    ):
        self.augment = augment
        self.paths = []

        skipped_shape = 0
        skipped_missing = 0

        for p in sorted(all_paths):
            d = np.load(p, allow_pickle=True)

            if skip_partial:
                _, h, w = d["image"].shape
                if h != tile_size or w != tile_size:
                    skipped_shape += 1
                    continue

            layer_names = list(d["layer_names"]) if "layer_names" in d else []
            if any(b not in layer_names for b in _BANDS_TO_LOAD):
                skipped_missing += 1
                continue

            self.paths.append(p)

        print(f"Found {len(self.paths)} tiles "
              f"({skipped_shape} partial, {skipped_missing} missing bands skipped)")

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

    def __getitem__(self, idx):
        path = self.paths[idx]
        d = np.load(path, allow_pickle=True)

        layer_names  = list(d["layer_names"])
        li = {name: i for i, name in enumerate(layer_names)}
        band_indices = [li[b] for b in _BANDS_TO_LOAD]

        image = d["image"][band_indices].astype(np.float32)
        mask = d["labels"].astype(np.float32)

        for i, name in enumerate(_BANDS_TO_LOAD):
          band = image[i].astype(np.float32)

          if name == ["DEM_SLOPE"]:
              band = np.log1p(np.maximum(band, 0))

          elif name == ['LAPLACE']:
              band = np.sign(band) * np.log1p(np.abs(band))

          image[i] = (band - band.mean()) / (band.std() + 1e-6)

        if self.augment:
            image, mask = self._augment(image, mask)

        # hillshade test
        dem = image[0]
        hillshade = calculate_hillshade(
        dem,
    )
        
        image = np.concatenate(
        [image, hillshade[None, :, :]],
        axis=0,
    )

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
        )
