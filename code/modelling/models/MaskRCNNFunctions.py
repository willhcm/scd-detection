import numpy as np
from scipy.ndimage import label, center_of_mass 
import torch


STAGE_LRS = {
    1: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "conv1": 1e-4,
        "fpn": 0.0,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    2: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "conv1": 1e-5,
        "fpn": 1e-5,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    3: {
        "rpn": 5e-4,
        "roi_heads": 5e-4,
        "conv1": 5e-5,
        "fpn": 3e-6,
        "layer4": 1e-6,
        "layer3": 0.0,
    },
    4: {
        "rpn": 1e-4,
        "roi_heads": 1e-4,
        "conv1": 5e-6,
        "fpn": 1e-6,
        "layer4": 5e-7,
        "layer3": 1e-7,
    },
}


def collate_fn(batch):
    return tuple(zip(*batch))

def binary_iou(pred_union, gt_union):
    pred_union = pred_union.astype(bool)
    gt_union = gt_union.astype(bool)

    intersection = np.logical_and(pred_union, gt_union).sum()
    union = np.logical_or(pred_union, gt_union).sum()

    if union == 0:
        return 1.0

    return intersection / union

def maskrcnn_outputs_to_binary_masks(outputs, threshold=0.5, score_thresh=0.5):
    batch_masks = []

    for out in outputs:
        if len(out["scores"]) == 0:
            batch_masks.append(None)
            continue

        keep = out["scores"] >= score_thresh

        if keep.sum() == 0:
            batch_masks.append(None)
            continue

        masks = out["masks"][keep, 0]  # [N, H, W]
        binary = masks >= threshold

        batch_masks.append(binary.cpu().numpy())

    return batch_masks


def object_f1_from_instance_masks(pred_masks, gt_masks):
    """
    pred_masks: [N_pred, H, W] or None
    gt_masks:   [N_gt, H, W]
    """

    n_gt = gt_masks.shape[0]
    n_pred = 0 if pred_masks is None else pred_masks.shape[0]

    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0

    if n_gt == 0:
        return 0, n_pred, 0

    if n_pred == 0:
        return 0, 0, n_gt

    gt_union = gt_masks.sum(axis=0) > 0
    gt_labels, n_gt_cc = label(gt_union)

    matched_gt = set()

    tp = 0
    fp = 0

    for pm in pred_masks:
        cy, cx = center_of_mass(pm)

        if np.isnan(cx) or np.isnan(cy):
            fp += 1
            continue

        r = int(round(cy))
        c = int(round(cx))

        r = np.clip(r, 0, gt_union.shape[0] - 1)
        c = np.clip(c, 0, gt_union.shape[1] - 1)

        gt_id = gt_labels[r, c]

        if gt_id > 0 and gt_id not in matched_gt:
            tp += 1
            matched_gt.add(gt_id)
        else:
            fp += 1

    fn = n_gt_cc - len(matched_gt)

    return tp, fp, fn


# recomp to make sure IoU and pixel-f1 were calculated at same level (previously image averaged vs global.)
def evaluate_maskrcnn_metrics(
    model,
    val_loader,
    device,
    mask_thresh=0.55,
    score_thresh=0.77,
):
    model.eval()

    total_tp = total_fp = total_fn = 0
    ptp = pfp = pfn = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images_device = [image.to(device) for image in images]
            outputs = model(images_device)

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

                keep = scores >= score_thresh


                pred_instance_masks = (
                    raw_pred_masks[keep, 0] >= mask_thresh
                )

                # Object-level counts
                tp, fp, fn = object_f1_from_instance_masks(
                    pred_instance_masks,
                    gt_instance_masks,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn


                # Merge predicted instances
                if pred_instance_masks.shape[0] > 0:
                    pred_mask = pred_instance_masks.any(dim=0)
                else:
                    pred_mask = torch.zeros(
                        (height, width),
                        dtype=torch.bool,
                    )

                # Global pixel counts
                ptp += (pred_mask & target_mask).sum().item()
                pfp += (pred_mask & ~target_mask).sum().item()
                pfn += (~pred_mask & target_mask).sum().item()

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

    return (
        precision,
        recall,
        f1,
        pixel_iou,
        p_precision,
        p_recall,
        p_f1,
    )

# ai assistance with mask rcnn train functionality and helper functions. original code repo from paper is hard to understand
# ChatGPT and Claude Sonnet 5 used - whenever I have said AI is used its these models.


def build_staged_optimizer(model, weight_decay=1e-4):
    return torch.optim.AdamW(
        [
            {
                "name": "rpn",
                "params": model.model.rpn.parameters(),
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