# Using Mask R-CNN pre-trained backbone with a custom CentreNet-style prediction head.

import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn
import torch
from CentreNetStrided import decode_centernet_predictions
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter
import random
from torch.utils.data import Dataset
from pathlib import Path
from datetime import datetime
import copy


# constants
# readded level 3 as dataset now includes some larger SCDs.
LEVEL_STRIDES = {
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32
}
# overlapp bins, gives model choice (implicitly)
LEVEL_RADIUS_BINS = { 
    "0": (0, 24),
    "1": (20, 48),
    "2": (38, 96),
    "3": (80, np.inf),

}

_BANDS_TO_LOAD = ["DEM", "DEM_SLOPE", "RR", "LAPLACE"]

class CenterHead(nn.Module):
    def __init__(self, in_channels=256, hidden=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )

         # added to reduce overfitting
        self.hm_drop = nn.Dropout2d(0.3)
        self.reg_drop = nn.Dropout2d(0.1)
        
       
        self.heatmap = nn.Conv2d(hidden, 1, 1)
        self.offset  = nn.Conv2d(hidden, 2, 1)
        self.radius  = nn.Conv2d(hidden, 1, 1)

        # start heatmap with low confidence (sigmoids to 0.1)
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x):
        feat = self.shared(x)
        
        return {
            "heatmap": self.heatmap(self.hm_drop(feat)),
            "offset": self.offset(self.reg_drop(feat)),
            "radius": self.radius(self.reg_drop(feat)),
        }
            
class FPNCentreNet(nn.Module):
    def __init__(self, in_channels=4, train_all=False):
        super().__init__()

        self.backbone = self._build_backbone()

        self.heads = nn.ModuleDict({
            "0": CenterHead(256),
            "1": CenterHead(256),
            "2": CenterHead(256),
            "3": CenterHead(256),
        })

        # Start with only conv1, bn1, and heads trainable
        self.train_all = train_all
        self.set_training_stage(stage=1)

    def set_training_stage(self, stage):
        if self.train_all:
            for parameter in self.parameters():
                parameter.requires_grad = True
            return

        # Freeze backbone.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Heads always train.
        for parameter in self.heads.parameters():
            parameter.requires_grad = True

        # New input convolution.
        for parameter in self.backbone.body.conv1.parameters():
            parameter.requires_grad = True

        if stage >= 2:
            # Adapt the feature pyramid.
            for parameter in self.backbone.fpn.parameters():
                parameter.requires_grad = True

        if stage >= 3:
            # Adapt highest-level ResNet features.
            for parameter in self.backbone.body.layer4.parameters():
                parameter.requires_grad = True

        if stage >= 4:
            for parameter in self.backbone.body.layer3.parameters():
                parameter.requires_grad = True
                
    def _build_backbone(self, in_channels=4):
        detector = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        )

        backbone = detector.backbone
        old_conv = backbone.body.conv1

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)

            # Initialise every terrain channel from the average RGB filter.
            new_conv.weight.copy_(
                mean_weight.repeat(1, in_channels, 1, 1)
            )

        backbone.body.conv1 = new_conv

        return backbone
    
    def _freeze_backbone(self): # depreciated, remained incase want to manually freeze.

        for name, param in self.backbone.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name == "conv1" or layer_name == "bn1":
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    
    def forward(self, images):
        features = self.backbone(images)

        outputs = {}
        for level, feat in features.items():
            if level in self.heads:
                outputs[level] = self.heads[level](feat)

        return outputs
    

# helpers 
# same function as in CentreNetStrided
def build_centernet_targets_for_objects(
    mask,
    stride=4,
    radius_range_px=(0, np.inf),
    min_sigma=1.0,
    max_sigma=6.0,
    base_weight=1.0,
    centre_weight=5.0,
):

    labeled, n = label(mask > 0)
    h, w = mask.shape
    oh, ow = h // stride, w // stride # reduced size dimensions (for variably strided levels.)

    # make output arrays
    heatmap = np.zeros((oh, ow), dtype=np.float32)
    weights = np.ones((oh, ow), dtype=np.float32) * base_weight
    offset = np.zeros((2, oh, ow), dtype=np.float32)
    radius = np.zeros((oh, ow), dtype=np.float32)
    obj_mask = np.zeros((oh, ow), dtype=np.float32)

    yy, xx = np.indices((oh, ow))

    min_r, max_r = radius_range_px  # upper and lower radius bound from bins defined above.

    # iterates over all objects 
    for obj_id in range(1, n + 1):

        # specific object is where mask has class of obj_id (each target object has a unique id)
        obj = labeled == obj_id
        area = float(obj.sum())

        # clean
        if area <= 0:
            continue

        # best fit circle type of thing
        radius_full = np.sqrt(area / np.pi)

        # object-size assignment to this FPN level
        # does the object fit in the scope of this level?
        if not (min_r <= radius_full < max_r):
            # if not, go to next object 
            # and skipped object will appear in another correct level.
            continue

        cy, cx = center_of_mass(obj)

        # transform into reduced pred space.
        cy_low = cy / stride
        cx_low = cx / stride

        # as integer (offset prediction used in head to correct for this.)
        cy_i = int(np.floor(cy_low))
        cx_i = int(np.floor(cx_low))

        # if int centroid is not in reduced size bounds, skip.
        if not (0 <= cy_i < oh and 0 <= cx_i < ow):
            continue

        # how big should target guassian be?, clips to min and max sigma size.
        # this is in full 512x512 space
        sigma_full = np.clip(np.sqrt(area) * 0.15, 4.0, 30.0)
        # reduce to strided space.
        sigma_low = np.clip(sigma_full / stride, min_sigma, max_sigma)

        # make gaussian
        g = np.exp(
            -((yy - cy_i) ** 2 + (xx - cx_i) ** 2) / (2 * sigma_low ** 2)
        ).astype(np.float32)

        # max of heatmap (all zeros), and gaussian - just fills target with gaussian
        heatmap = np.maximum(heatmap, g)
        # replaces rest with 1.0s
        heatmap[cy_i, cx_i] = 1.0

        # adds centre weight to weight array
        weights[cy_i, cx_i] = centre_weight

        # set offset array (2 channel) at centre coords to be the offset val
        offset[0, cy_i, cx_i] = cy_low - cy_i
        offset[1, cy_i, cx_i] = cx_low - cx_i

        # same with radius
        radius[cy_i, cx_i] = radius_full / stride

        # used to unmask only centroid location for objects.
        obj_mask[cy_i, cx_i] = 1.0

    # returns one target containing all objects at the inputted level.
    return heatmap, weights, offset, radius, obj_mask

# apply centrenet target building function to different levels (based on stride)
def build_multilevel_centernet_targets(
    mask,
    level_strides=LEVEL_STRIDES,
    level_radius_bins=LEVEL_RADIUS_BINS,
):
    """
    Build targets for all FPN levels.
    """

    targets = {}

    # for every level (and associated stride), build relevant targets
    for level, stride in level_strides.items():
        heatmap, weights, offset, radius, obj_mask = build_centernet_targets_for_objects(
            mask=mask,
            stride=stride,
            radius_range_px=level_radius_bins[level],
        )

        # e.g. target[0] = {heatmap: ..., }. (concatenated objects for each level all in one tile).
        targets[level] = {
            "heatmap": heatmap,
            "weights": weights,
            "offset": offset,
            "radius": radius,
            "obj_mask": obj_mask,
        }

    return targets
    
# Dataset

class MultiLevelCentreNetDataset(Dataset):
    def __init__(
        self,
        all_paths,
        tile_size=512,
        skip_partial=True,
        augment=False,
        level_strides=LEVEL_STRIDES,
        level_radius_bins=LEVEL_RADIUS_BINS,
    ):
        self.paths = []
        self.augment = augment
        self.level_strides = level_strides
        self.level_radius_bins = level_radius_bins

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

        print(
            f"Found {len(self.paths)} tiles "
            f"({skipped_shape} partial, {skipped_missing} missing bands skipped)"
        )

    def __len__(self):
        return len(self.paths)

    def _augment(self, image, mask):
        image, mask = image.copy(), mask.copy()

        if random.random() > 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]

        if random.random() > 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]

        k = random.choice([0, 1, 2, 3])
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            mask = np.rot90(mask, k)

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        # DEM-only perturbations, same spirit as your current dataset
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

        layer_names = list(d["layer_names"])
        li = {name: i for i, name in enumerate(layer_names)}

        band_indices = [li[b] for b in _BANDS_TO_LOAD]

        image = d["image"][band_indices].astype(np.float32)
        mask = d["labels"].astype(np.float32)

        # Per-tile z-score
        for i in range(len(_BANDS_TO_LOAD)):
            band = image[i]
            image[i] = (band - band.mean()) / (band.std() + 1e-6)

        if self.augment:
            image, mask = self._augment(image, mask)

        targets_np = build_multilevel_centernet_targets(
            mask=mask,
            level_strides=self.level_strides,
            level_radius_bins=self.level_radius_bins,
        )

        targets = {}
        for level, t in targets_np.items():
            targets[level] = {
                "heatmap": torch.from_numpy(np.ascontiguousarray(t["heatmap"])),
                "weights": torch.from_numpy(np.ascontiguousarray(t["weights"])),
                "offset": torch.from_numpy(np.ascontiguousarray(t["offset"])),
                "radius": torch.from_numpy(np.ascontiguousarray(t["radius"])),
                "obj_mask": torch.from_numpy(np.ascontiguousarray(t["obj_mask"])),
            }

        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask)),
            targets,
        )

# calculates loss at different levels (predictions occur from p0->p2 from fpn)
def multilevel_centernet_loss(
    outputs,
    targets,
    criterion,
    level_weights=None,
    use_softplus_radius=True,
):
    """
    outputs:
        {
            "0": {"heatmap": ..., "offset": ..., "radius": ...},
            "1": ...
        }

    targets:
        {
            "0": {"heatmap": ..., "weights": ..., "offset": ..., "radius": ..., "obj_mask": ...},
            "1": ...
        }
    """
    if level_weights is None:
        level_weights = {level: 1.0 for level in outputs.keys()}

    total_loss = 0.0
    logs = {}

    used_levels = 0

    for level, out in outputs.items():
        if level not in targets:
            continue

        t = targets[level]

        pred_hm = out["heatmap"]
        pred_off = out["offset"]
        pred_rad = out["radius"]

        if use_softplus_radius:
            pred_rad = F.softplus(pred_rad)

        hm = t["heatmap"].to(pred_hm.device).unsqueeze(1).float()
        weights = t["weights"].to(pred_hm.device).unsqueeze(1).float()
        off = t["offset"].to(pred_hm.device).float()
        rad = t["radius"].to(pred_hm.device).float()
        obj_mask = t["obj_mask"].to(pred_hm.device).float()

        loss, hm_loss, off_loss, rad_loss = criterion(
            pred_hm,
            pred_rad,
            pred_off,
            hm,
            rad,
            off,
            obj_mask,
            weight=weights,
        )

        lw = level_weights.get(level, 1.0)
        total_loss = total_loss + lw * loss
        used_levels += 1

        logs[f"{level}_loss"] = float(loss.detach().cpu())
        logs[f"{level}_hm"] = float(hm_loss.detach().cpu())
        logs[f"{level}_off"] = float(off_loss.detach().cpu())
        logs[f"{level}_rad"] = float(rad_loss.detach().cpu())

    total_loss = total_loss / max(1, used_levels)

    return total_loss, logs



def simple_detection_nms(detections, min_dist_px=20, radius_fraction=0.5):
    """
    Remove duplicate centre detections from different FPN levels.

    Keeps highest-score detections first.
    """
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []

    for det in detections:
        cy = det["cy"]
        cx = det["cx"]
        r = det.get("radius", np.nan)

        keep = True

        for old in kept:
            old_r = old.get("radius", np.nan)

            dist = np.sqrt((cy - old["cy"]) ** 2 + (cx - old["cx"]) ** 2)

            if np.isfinite(r) and np.isfinite(old_r):
                threshold = max(min_dist_px, radius_fraction * max(r, old_r))
            else:
                threshold = min_dist_px

            if dist < threshold:
                keep = False
                break

        if keep:
            kept.append(det)

    return kept


def decode_multilevel_predictions(
    outputs,
    image_index,
    level_strides=LEVEL_STRIDES,
    thresholds=None,
    min_distances=None,
    use_softplus_radius=True,
):
    """
    Decode predictions from all FPN levels for one image in a batch.
    """

    # thresholds tuned using score confidence at differnt levels
    if thresholds is None:
        thresholds = {
            "0": 0.5,
            "1": 0.525,
            "2": 0.575,
            "3": 0.75,
        }

    # need to tune likely
    if min_distances is None:
        # In low-resolution feature-map pixels.
        min_distances = {
            "0": 6,
            "1": 4,
            "2": 3,
            "3": 2,
        }

    all_detections = []

    for level, out in outputs.items():
        if level not in level_strides:
            continue

        stride = level_strides[level]

        pred_hm = torch.sigmoid(out["heatmap"][image_index, 0]).detach().cpu().numpy()

        pred_off = out["offset"][image_index].detach().cpu().numpy()

        pred_rad_t = out["radius"][image_index]

        if use_softplus_radius:
            pred_rad_t = F.softplus(pred_rad_t)

        pred_rad = pred_rad_t.detach().cpu().numpy()

        detections = decode_centernet_predictions(
            pred_hm=pred_hm,
            pred_rad=pred_rad,
            pred_off=pred_off,
            threshold=thresholds[level],
            min_distance=min_distances[level],
            stride=stride,
        )

        for d in detections:
            d["level"] = level
            d["stride"] = stride

        all_detections.extend(detections)

    all_detections = simple_detection_nms(all_detections)

    return all_detections

def centroid_metrics_from_detections(detections, gt_mask):
    """
    TP if predicted centre lies inside an unmatched GT object.
    """
    from scipy.ndimage import label

    gt_labels, n_gt = label(gt_mask > 0)

    matched_gt = set()
    tp = 0
    fp = 0

    h, w = gt_mask.shape

    for det in detections:
        r = int(round(det["cy"]))
        c = int(round(det["cx"]))

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

    return tp, fp, fn

def build_staged_optimizer(model, weight_decay=1e-4):
    return torch.optim.AdamW(
        [
            {
                "name": "heads",
                "params": model.heads.parameters(),
                "lr": 1e-3,
            },
            {
                "name": "conv1",
                "params": model.backbone.body.conv1.parameters(),
                "lr": 5e-6,
            },
            {
                "name": "fpn",
                "params": model.backbone.fpn.parameters(),
                "lr": 0.0,
            },
            {
                "name": "layer4",
                "params": model.backbone.body.layer4.parameters(),
                "lr": 0.0,
            },
            {
                "name": "layer3",
                "params": model.backbone.body.layer3.parameters(),
                "lr": 0.0,
            },
        ],
        weight_decay=weight_decay,
    )

STAGE_LRS = {
    1: {
        "heads": 1e-3,
        "conv1": 5e-6,
        "fpn": 0.0,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    2: {
        "heads": 5e-4,
        "conv1": 5e-6,
        "fpn": 1e-5,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    3: {
        "heads": 1e-4,
        "conv1": 1e-6,
        "fpn": 3e-6,
        "layer4": 1e-6,
        "layer3": 0.0,
    },
    4: {
        "heads": 1e-5,
        "conv1": 5e-7,
        "fpn": 1e-6,
        "layer4": 5e-7,
        "layer3": 1e-7,
    },
}


def apply_training_stage(model, optimizer, stage):
    model.set_training_stage(stage)

    lrs = STAGE_LRS[stage]

    for group in optimizer.param_groups:
        group["lr"] = lrs[group["name"]]

    print(f"Applied training stage {stage}")
    for group in optimizer.param_groups:
        n_trainable = sum(
            parameter.numel()
            for parameter in group["params"]
            if parameter.requires_grad
        )
        print(
            f"  {group['name']:8s} "
            f"lr={group['lr']:.2e}, "
            f"trainable={n_trainable:,}"
        )


def FPN_CN_train(
    model,
    criterion,
    train_loader,
    val_loader,
    epochs,
    thresholds=None,
    min_distances=None,
    weight_decay=1e-4,
    checkpoint_root="/content/drive/MyDrive/IRP/models/checkpoints",
):
    

    level_strides = {
        "0": 4,
        "1": 8,
        "2": 16,
        "3": 32,
    }


    level_weights={
        "0": 0.40,
        "1": 0.30,
        "2": 0.20,
        "3": 0.1,
    }


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    checkpoint_dir = Path(checkpoint_root) / f"fpn_centrenet_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tls = []
    vls = []
    precisions = []
    recalls = []
    f1s = []

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
   
    if model.train_all:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.heads.parameters(),
                    "lr": 3e-4,
                },
                {
                    "params": model.backbone.fpn.parameters(),
                    "lr": 1e-5,
                },
                {
                    "params": model.backbone.body.parameters(),
                    "lr": 1e-6,
                },
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = build_staged_optimizer(
            model,
            weight_decay=weight_decay,
        )


    current_stage = 1
    apply_training_stage(model, optimizer, current_stage)

    patience = 20
    for epoch in range(epochs):

        # dont reinit adam, just update lrs.
        if not model.train_all:
            if epoch == 20:
                current_stage = 2
                apply_training_stage(model, optimizer, current_stage)

            elif epoch == 30:
                current_stage = 3
                apply_training_stage(model, optimizer, current_stage)

            elif epoch == 40:
                current_stage = 4
                apply_training_stage(model, optimizer, current_stage)

        # Train
        model.train()
        train_loss = 0.0

        for images, masks, targets in train_loader:
            images = images.to(device).float()

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss, loss_logs = multilevel_centernet_loss(
                outputs=outputs,
                targets=targets,
                criterion=criterion,
                level_weights=level_weights,
                use_softplus_radius=True,
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        avg_train = train_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        tp = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            for images, masks, targets in val_loader:
                images = images.to(device).float()
                masks_np = masks.cpu().numpy()

                outputs = model(images)

                loss, loss_logs = multilevel_centernet_loss(
                    outputs=outputs,
                    targets=targets,
                    criterion=criterion,
                    level_weights=level_weights,
                    use_softplus_radius=True,
                )

                val_loss += loss.item()

                batch_size = images.shape[0]

                for i in range(batch_size):
                    detections = decode_multilevel_predictions(
                        outputs=outputs,
                        image_index=i,
                        level_strides=level_strides,
                        thresholds=thresholds,
                        min_distances=min_distances,
                        use_softplus_radius=True,
                    )

                    tp_i, fp_i, fn_i = centroid_metrics_from_detections(
                        detections=detections,
                        gt_mask=masks_np[i],
                    )

                    tp += tp_i
                    fp += fp_i
                    fn += fn_i

        avg_val = val_loss / max(1, len(val_loader))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        tls.append(avg_train)
        vls.append(avg_val)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())

            for old in checkpoint_dir.glob("best_*.pt"):
                old.unlink()

            torch.save(
                best_state,
                checkpoint_dir / f"best_epoch{epoch + 1:03d}_f1{f1:.4f}.pt",
            )

            print(f"Best checkpoint saved: F1={f1:.4f}")

        else:
            bad_epochs += 1

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train {avg_train:.4f} | val {avg_val:.4f} | "
            f"P {precision:.4f} | R {recall:.4f} | F1 {f1:.4f}"
        )

        if bad_epochs >= patience:
            print("early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return vls, tls, precisions, recalls, f1s, model, val_loader


# add geology prior? elevation in ring vs outside ring? 
class CentreNetLoss(nn.Module):
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
    
