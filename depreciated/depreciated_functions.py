## collection of functions which are no longer used in the main code/ section (for training the maskrcnn, classifier or running inference)
## kept here for completeness!!

# AI assistance with learning scipy.ndimage functionality.

# GenAI assistance in creating this function, whcih takes in the predicted mask and compares the mask object centroids to
# ground truth mask, if the predicted mask centroid is in the ground truth mask, it qualifies as a true positive.
# only one prediction allowed per ground truth object.

from scipy.ndimage import label, center_of_mass
import numpy as np

# used for FPN CentreNet initially. not used for maskrcnn, so depreciated.
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



# AI assistance with this plotting function
def pred_centroids_for_plot(pred_mask, gt_mask, min_pred_area=20):
    pred_labels, n_pred = label(pred_mask > 0)
    gt_labels, _ = label(gt_mask > 0)

    points = []

    for pred_id in range(1, n_pred + 1):
        pred_obj = pred_labels == pred_id

        if pred_obj.sum() < min_pred_area:
            continue

        cy, cx = center_of_mass(pred_obj)

        if np.isnan(cx) or np.isnan(cy):
            continue

        r = int(round(cy))
        c = int(round(cx))

        r = np.clip(r, 0, gt_mask.shape[0] - 1)
        c = np.clip(c, 0, gt_mask.shape[1] - 1)

        detected = gt_labels[r, c] > 0
        points.append((cx, cy, detected))

    return points


# was previously used for mask rcnn training / validation. 
def maskrcnn_outputs_to_binary_masks(outputs, threshold=0.5, score_thresh=0.5):
    """
    Converts the raw outputs from a Mask R-CNN model into binary masks based on a specified threshold and score threshold."""
    batch_masks = []
 
    for out in outputs:
        # if the tile contains no predictions, skip
        if len(out["scores"]) == 0:
            batch_masks.append(None)
            continue

        keep = out["scores"] >= score_thresh

        # add output to list
        if keep.sum() == 0:
            batch_masks.append(None)
            continue

        masks = out["masks"][keep, 0]  # [N, H, W]
        binary = masks >= threshold

        batch_masks.append(binary.cpu().numpy())

    return batch_masks
