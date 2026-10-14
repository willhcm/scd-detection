import numpy as np
from scipy.ndimage import label, center_of_mass 
import torch

from collections import defaultdict

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
        "fpn": 5e-6,
        "layer4": 1e-5,
        "layer3": 0.0,
    },
    4: {
        "rpn": 1e-6,
        "roi_heads": 1e-6,
        "conv1": 1e-6,
        "film": 1e-6,
        "fpn": 1e-6,
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


# recomp to make sure IoU and pixel-f1 were calculated at same level (previously image averaged vs global.)
def evaluate_maskrcnn_metrics(
    model,
    val_loader,
    device,
    mask_thresh=0.55,
    score_thresh=0.77):
    """
    Evaluates the Mask R-CNN model on a validation dataset and computes object-level and pixel-level metrics.
    """
    model.eval()

    total_tp = total_fp = total_fn = 0
    ptp = pfp = pfn = 0
    box_iou_final = []

    with torch.no_grad():
        for images, targets in val_loader:
            images_device = [image.to(device) for image in images]

            resolutions = torch.stack([target["resolution"] for target in targets]).to(device)
            outputs = model(images_device, resolutions=resolutions)

            for image, output, target_dict in zip(images, outputs, targets):

                height, width = image.shape[-2:]

                gt_instance_masks = target_dict["masks"].detach().cpu().bool()

                if gt_instance_masks.shape[0] > 0:
                    target_mask = gt_instance_masks.any(dim=0)
                else:
                    target_mask = torch.zeros(
                        (height, width),
                        dtype=torch.bool,
                    )

                scores = output["scores"].detach().cpu()
                raw_pred_masks = output["masks"].detach().cpu()

                # predictions only kept if they meet the score threshold
                keep = scores >= score_thresh

                # threshold masks
                pred_instance_masks = (raw_pred_masks[keep, 0] >= mask_thresh)

                pred_boxes = output["boxes"].detach().cpu()[keep]
                gt_boxes = target_dict["boxes"].detach().cpu()

                box_ious = box_iou(pred_boxes, gt_boxes)
                box_iou_final.extend(box_ious.tolist())

                tp, fp, fn = object_f1_from_boxes(pred_boxes, gt_boxes, iou_thresh=0.5)

                total_tp += tp
                total_fp += fp
                total_fn += fn


                # Merge predicted instances
                if pred_instance_masks.shape[0] > 0:
                    pred_mask = pred_instance_masks.any(dim=0)
                # or return a zero mask if no predictions were made
                else:
                    pred_mask = torch.zeros(
                        (height, width),
                        dtype=torch.bool,
                    )

                # Global pixel counts
                ptp += (pred_mask & target_mask).sum().item()
                pfp += (pred_mask & ~target_mask).sum().item()
                pfn += (~pred_mask & target_mask).sum().item()

    # safe div for precision etc
    eps = 1e-8

    # Object-level metrics
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn + eps)

    # Pixel-level metrics, all calculated from identical global counts
    p_precision = ptp / (ptp + pfp + eps)
    p_recall = ptp / (ptp + pfn + eps)
    p_f1 = 2 * ptp / (2 * ptp + pfp + pfn + eps)
    pixel_iou = ptp / (ptp + pfp + pfn + eps)
    box_iou_final = np.mean(box_iou_final) if box_iou_final else 0.0

    # results for that validation epoch
    return (
        precision,
        recall,
        f1,
        pixel_iou,
        p_precision,
        p_recall,
        p_f1,
        box_iou_final,
    )

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
                "lr": 5e-6,
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