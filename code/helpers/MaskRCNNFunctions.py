import numpy as np
from scipy.ndimage import label, center_of_mass 
import torch
from collections import defaultdict
import math

from torchvision.ops import box_iou

# AI assistance with:

# collate_fn. (didnt know what this was)
# maskRCNN evaluation metrics (IoU, F1 etc and definition of a 'matched object')
# training functionality (specifically fine-tuning and incremental unfreezing in rcnn_train.)

## Displayed in Table 1 of the report.
STAGE_LRS = {
    1: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "film": 1e-3,
        "conv1": 1e-3,
        "fpn": 0.0,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    2: {
        "rpn": 1e-4,
        "roi_heads": 1e-4,
        "film": 1e-4,
        "conv1": 1e-4,
        "fpn": 1e-5,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    3: {
        "rpn": 1e-5,
        "roi_heads": 1e-5,
        "conv1": 1e-5,
        "film": 1e-5,
        "fpn": 1e-6,
        "layer4": 1e-5,
        "layer3": 0.0,
    },
    4: {
        "rpn": 1e-6,
        "roi_heads": 1e-6,
        "conv1": 1e-6,
        "film": 1e-6,
        "fpn": 1e-7,
        "layer4": 1e-6,
        "layer3": 1e-6,
    },
}

def collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader to handle batches of data with varying sizes.
    Mask R-CNN instance segmentation requires a custom collate function to properly handle the variable number of instances per image."""
    return tuple(zip(*batch))

def object_f1_from_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt

    ious = box_iou(pred_boxes, gt_boxes)
    taken = torch.zeros(n_gt, dtype=torch.bool)
    tp = 0

    for i in range(n_pred):
        row = ious[i].masked_fill(taken, -1)
        j = row.argmax()
        if row[j] >= iou_thresh:
            taken[j] = True
            tp += 1

    return tp, n_pred - tp, n_gt - tp


def old_object_f1_from_boxes(pred_boxes, gt_boxes, iou_thresh=0.3):
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt

    ious = box_iou(pred_boxes, gt_boxes)
    taken = torch.zeros(n_gt, dtype=torch.bool, device=ious.device)
    tp = 0

    for i in range(n_pred):
        row = ious[i].masked_fill(taken, -1)
        j = row.argmax()
        if row[j] >= iou_thresh:
            taken[j] = True
            tp += 1
            continue

        # no IoU match, try centroid inside an unmatched gt box
        x1, y1, x2, y2 = pred_boxes[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        inside = (
            (gt_boxes[:, 0] <= cx) & (cx <= gt_boxes[:, 2])
            & (gt_boxes[:, 1] <= cy) & (cy <= gt_boxes[:, 3])
            & ~taken
        )
        if inside.any():
            taken[inside.nonzero()[0, 0]] = True
            tp += 1

    return tp, n_pred - tp, n_gt - tp

def merge_masks(masks, h, w):
    # union of instance masks -> single binary mask
    if masks.shape[0] == 0:
        return torch.zeros((h, w), dtype=torch.bool)
    return masks.any(dim=0)


def evaluate_maskrcnn_metrics(model, val_loader, device, mask_thresh=0.55, score_thresh=0.77):
    model.eval()

    tp = fp = fn = 0           # object level
    ptp = pfp = pfn = 0        # pixel level, global counts
    box_ious = []
    old_tp = old_fp = old_fn = 0   # object level, old definition

    with torch.no_grad():
        for images, targets in val_loader:
            res = torch.stack([t["resolution"] for t in targets]).to(device)
            outputs = model([im.to(device) for im in images], resolutions=res)

            for image, out, tgt in zip(images, outputs, targets):
                h, w = image.shape[-2:]

                keep = out["scores"].cpu() >= score_thresh
                pred_boxes = out["boxes"].cpu()[keep]
                gt_boxes = tgt["boxes"].cpu()

                pred_masks = out["masks"].cpu()[keep, 0] >= mask_thresh
                gt_masks = tgt["masks"].cpu().bool()

                if len(pred_boxes) and len(gt_boxes):
                    # best-matching gt for each kept prediction
                    box_ious.extend(box_iou(pred_boxes, gt_boxes).max(dim=1).values.tolist())

                a, b, c = object_f1_from_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)
                tp += a
                fp += b
                fn += c

                x, y, z = old_object_f1_from_boxes(pred_boxes, gt_boxes, iou_thresh=0.3)
                old_tp += x
                old_fp += y
                old_fn += z

                pred = merge_masks(pred_masks, h, w)
                gt = merge_masks(gt_masks, h, w)

                ptp += (pred & gt).sum().item()
                pfp += (pred & ~gt).sum().item()
                pfn += (~pred & gt).sum().item()

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)

    p_precision = ptp / (ptp + pfp + eps)
    p_recall = ptp / (ptp + pfn + eps)
    p_f1 = 2 * ptp / (2 * ptp + pfp + pfn + eps)
    pixel_iou = ptp / (ptp + pfp + pfn + eps)

    old_f1 = 2 * old_tp / (2 * old_tp + old_fp + old_fn + eps)

    mean_box_iou = float(np.mean(box_ious)) if box_ious else 0.0

    return precision, recall, f1, pixel_iou, p_precision, p_recall, p_f1, mean_box_iou, old_f1

# ai assistance with mask rcnn train functionality and helper functions. original code repo from paper is hard to understand
# ChatGPT and Claude Sonnet 5 used - whenever I have said AI is used its these models.


def build_staged_optimizer(model, weight_decay=1e-4):
    """
    Builds an AdamW optimizer with different learning rates for different parts of the Mask R-CNN model.
    This allows for staged training where different components of the model can be fine-tuned at different learning rates.
    """
    return torch.optim.AdamW(
        [
            {
                "name": "rpn",
                "params": model.model.rpn.parameters(),
                "lr": 1e-3,
            },
            {
                "name": "film",
                "params": model.model.backbone.film.parameters(),
                "lr": 1e-3,
            },
            {
                "name": "roi_heads",
                "params": model.model.roi_heads.parameters(),
                "lr": 1e-3,
            },
            {
                "name": "conv1",
                "params": model.model.backbone.body.conv1.parameters(),
                "lr": 5e-5,
            },
            {
                "name": "fpn",
                "params": model.model.backbone.fpn.parameters(),
                "lr": 0.0,
            },
            {
                "name": "layer4",
                "params": model.model.backbone.body.layer4.parameters(),
                "lr": 0.0,
            },
            {
                "name": "layer3",
                "params": model.model.backbone.body.layer3.parameters(),
                "lr": 0.0,
            },
        ],
        weight_decay=weight_decay,
    )

def apply_training_stage(model, optimizer, stage):
    """
    Applies a specific training stage to the Mask R-CNN model.
    """
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

@torch.no_grad()
def val_loss(model, loader, device):
    """
    Computes the validation loss for a given model and data loader.
    """
    model.train()
    total, n = 0.0, 0
    parts = {}
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        resolutions = torch.stack([t["resolution"] for t in targets]).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(imgs, resolutions, targets)
        total += sum(losses.values()).item() * len(imgs)
        n += len(imgs)
        for k, v in losses.items():
            parts[k] = parts.get(k, 0.0) + v.item() * len(imgs)
    return total / n, {k: v / n for k, v in parts.items()}



def stage_lr(stage, name, frac, final_scale=0.1):
    start = STAGE_LRS[stage][name]
    if start == 0.0:
        return 0.0
    # decay toward whatever the next stage uses; last stage just fades out
    end = STAGE_LRS.get(stage + 1, {}).get(name, start * final_scale)
    return end + (start - end) * 0.5 * (1 + math.cos(math.pi * frac))


def set_stage_progress(optimizer, stage, frac):
    for group in optimizer.param_groups:
        group["lr"] = stage_lr(stage, group["name"], frac)