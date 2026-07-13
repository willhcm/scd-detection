import numpy as np
from scipy.ndimage import label, center_of_mass, maximum_filter



# GenAI assistance in creating this function, whcih takes in the predicted mask and compares the mask object centroids to
# ground truth mask, if the predicted mask centroid is in the ground truth mask, it qualifies as a true positive.
# only one prediction allowed per ground truth object. similar to CentreNet function but is comparing mask agaisnt mask.
# has to derive centroid from mask first
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

# AI assistance with learning scipy.ndimage functionality.
# ChatGPT assistance with thois function. decodes downsampled 128x128 back to 512x512.
# prediction is done at 128 as otherise negative pixels in loss function dominate.
# this method performs better than downsampling negative pixels in loss directly.
def decode_centernet_predictions(pred_hm, pred_rad=None, pred_off=None, threshold=0.3, min_distance=8, stride=4):

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

# changed to multidirectional hillshade
# multi-directional hillshade. returns a function for registry
def calculate_hillshade(dem, cell_size=1.0, altitude_deg=45.0, z_factor=1.0):

    alt = np.radians(altitude_deg)
    dz_dx = np.gradient(dem * z_factor, cell_size, axis=1)
    dz_dy = np.gradient(dem * z_factor, cell_size, axis=0)
    slope = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    aspect = np.arctan2(-dz_dy, dz_dx)
    hs = np.zeros_like(dem, dtype=np.float64)
    for az_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        az = np.radians(360 - az_deg + 90)
        hs += np.cos(alt) * np.cos(slope) + np.sin(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs / 8, 0, 1).astype(np.float32)
