import numpy as np
from scipy.ndimage import label, center_of_mass 
import torch

from collections import defaultdict

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
        "conv1": 1e-4,
        "fpn": 0.0,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    2: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "film": 1e-3,
        "conv1": 1e-5,
        "fpn": 1e-5,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    3: {
        "rpn": 5e-4,
        "roi_heads": 5e-4,
        "conv1": 5e-5,
        "film": 1e-4,
        "fpn": 3e-6,
        "layer4": 1e-6,
        "layer3": 0.0,
    },
    4: {
        "rpn": 1e-4,
        "roi_heads": 1e-4,
        "conv1": 5e-6,
        "film": 1e-5,
        "fpn": 1e-6,
        "layer4": 5e-7,
        "layer3": 1e-7,
    },
}


def collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader to handle batches of data with varying sizes.
    Mask R-CNN instance segmentation requires a custom collate function to properly handle the variable number of instances per image."""
    return tuple(zip(*batch))


def object_f1_from_instance_masks(pred_masks, gt_masks):
    """
    Calculates the F1 score for object-level detection given predicted and ground truth instance masks.

    pred_masks: [N_pred, H, W] or None (i.e. no scd predictions)
    gt_masks:   [N_gt, H, W]
    """

    # Handle cases where there are no ground truth or predicted masks
    n_gt = gt_masks.shape[0]
    n_pred = 0 if pred_masks is None else pred_masks.shape[0]

    # If there are no ground truth and no predicted masks, return 0 for true positives, false positives, and false negatives.
    if n_gt == 0 and n_pred == 0:
        return 0, 0, 0

    if n_gt == 0:
        return 0, n_pred, 0

    if n_pred == 0:
        return 0, 0, n_gt

    # Create a union of all ground truth masks to identify unique ground truth objects
    gt_union = gt_masks.sum(axis=0) > 0
    gt_labels, n_gt_cc = label(gt_union)

    matched_gt = set()

    tp = 0
    fp = 0

    # Iterate through each predicted mask and determine if it matches a ground truth object
    for pm in pred_masks:
        cy, cx = center_of_mass(pm)

        if np.isnan(cx) or np.isnan(cy):
            fp += 1
            continue

        # cx and cy can be non integer ! handles!
        r = int(round(cy))
        c = int(round(cx))

        # Clip the centroid coordinates to ensure they are within valid bounds 
        r = np.clip(r, 0, gt_union.shape[0] - 1)
        c = np.clip(c, 0, gt_union.shape[1] - 1)

        # Get the label of the ground truth object at the centroid of the predicted mask
        # gt_labels are the connected component labels of the ground truth union mask, so gt_id will be 0 if there is no ground truth object at that location.
        gt_id = gt_labels[r, c]

        # If the predicted mask's centroid falls within a ground truth object that hasn't been matched yet, count it as a true positive.
        if gt_id > 0 and gt_id not in matched_gt:
            tp += 1
            matched_gt.add(gt_id)
        # If the predicted mask's centroid does not fall within any ground truth object or falls within an already matched ground truth object, count it as a false positive.
        else:
            fp += 1

    # The number of false negatives is the number of ground truth objects that were not matched by any predicted mask.
    fn = n_gt_cc - len(matched_gt)

    return tp, fp, fn


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

                # Object-level counts
                tp, fp, fn = object_f1_from_instance_masks(
                    pred_instance_masks,
                    gt_instance_masks)

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

    # results for that validation epoch
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




CHANNELS = ["dem", "slope", "laplace", "rr", "hillshade"]


def estimate_training_stats(
    dataset,
    n_pixels_per_tile=2000,
    seed=42,
):
    """
    Estimate normalization statistics from the training dataset.

    Every tile contributes approximately the same number of pixels,
    so large tiles / regions don't dominate the statistics.

    DEM:
        tile-level median is calculated later, so no global DEM stats.

    Other channels:
        global median + IQR estimated from sampled training pixels.
    """

    rng = np.random.default_rng(seed)

    samples = {
        ch: []
        for ch in CHANNELS
        if ch != "dem"
    }

    for tile_idx in range(len(dataset)):

        # ---------------------------------------------------------
        # Load your 5-channel tile
        # ---------------------------------------------------------
        # Adapt this to however your Dataset returns data.
        #
        # e.g.
        # tile = dataset.load_tile(tile_idx)
        #
        # Expected shape:
        #     (5, 512, 512)
        #
        tile = dataset.load_tile(tile_idx)

        # ---------------------------------------------------------
        # Valid pixel mask
        # ---------------------------------------------------------
        #
        # You should adapt this to your NoData/mask convention.
        #
        valid = np.isfinite(tile).all(axis=0)

        valid_indices = np.flatnonzero(valid)

        if len(valid_indices) == 0:
            continue

        n = min(n_pixels_per_tile, len(valid_indices))

        chosen = rng.choice(
            valid_indices,
            size=n,
            replace=False,
        )

        # ---------------------------------------------------------
        # Collect samples
        # ---------------------------------------------------------

        for ch_idx, ch in enumerate(CHANNELS):

            if ch == "dem":
                continue

            values = tile[ch_idx].ravel()[chosen]

            samples[ch].append(values)

    # -------------------------------------------------------------
    # Calculate robust statistics
    # -------------------------------------------------------------

    stats = {}

    for ch in samples:

        values = np.concatenate(samples[ch])

        median = np.median(values)
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25

        stats[ch] = {
            "median": float(median),
            "iqr": float(iqr),
            "n_samples": len(values),
        }

    return stats