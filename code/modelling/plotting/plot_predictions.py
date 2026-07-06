import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass
import numpy as np
import torch
from skimage.feature import peak_local_max
import matplotlib.patches as patches

CHANNELS = ['DEM',
            'Slope',
            'Residual Relief',
            'LAPLACE']

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

def get_seg_preds(val_loader, model, device):


    images, masks = next(iter(val_loader))
    images = images.to(device)
    masks = masks.unsqueeze(1).float().to(device)

    with torch.no_grad():
        logits = model(images)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.75).float()

    return images, masks, preds

def plot_segmentation_predictions(val_loader, model, device):
    
    to_plot = CHANNELS.extend(['Ground Truth', 'Labels'])
    n_cols = len(CHANNELS)

    fig, axes = plt.subplots(4, n_cols, figsize=(n_cols * 3, 14))

    for ax, col in zip(axes[0], to_plot):
        ax.set_title(col, fontsize=9)

    images, masks, preds = get_seg_preds(val_loader, model, device)

    for i in range(len(preds)):
        img = images[i].cpu().numpy()
        gt = masks[i, 0].cpu().numpy()
        pred = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img[0], cmap='terrain') # DEM
        axes[i, 1].imshow(img[0], cmap='terrain') # Slope
        axes[i, 2].imshow(img[1], cmap='terrain') # RR 
        axes[i, 3].imshow(img[2], cmap='coolwarm') # Laplace
        axes[i, 4].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 5].imshow(pred, cmap='gray', vmin=0, vmax=1)

        # annotations for interpretation

        # GT outline
        axes[i, 5].contour(gt, levels=[0.5], colors='lime', linewidths=1)

        # Predicted object outline
        axes[i, 5].contour(pred, levels=[0.5], colors='cyan', linewidths=1)

        points = pred_centroids_for_plot(pred, gt, min_pred_area=20)

        # green or red for predictrion success (TP, FP)
        for cx, cy, detected in points:
            colour = 'lime' if detected else 'red'
            axes[i, 4].scatter(cx, cy, c=colour, s=40, marker='x', linewidths=2)

        for j in range(n_cols):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def get_centre_net_preds(model, val_loader, device):

    model.eval()
    images, masks, centre_map, weights, offset, radius, obj_mask = next(iter(val_loader))
    images = images.to(device)
    masks_gt = masks.to(device).unsqueeze(1).float()
    centroids_gt = centre_map.to(device).unsqueeze(1).float()

    with torch.no_grad():
        pred_centres, pred_radius, pred_offset = model(images)

    pred_centres_cpu = torch.sigmoid(pred_centres).cpu().numpy()
    pred_radius_cpu = pred_radius.cpu().numpy()
    images_cpu = images.cpu().numpy()
    masks_gt_cpu = masks_gt.cpu().numpy()
    centroids_gt_cpu = centroids_gt.cpu().numpy()

    return pred_centres_cpu, pred_radius_cpu, images_cpu, masks_gt_cpu, centroids_gt_cpu


def plot_centrenet_predictions(model, val_loader, device):

    fig, axes = plt.subplots(len(val_loader), 5, figsize=(20, 3 * len(val_loader)))

    # get preds and inputs
    pred_centres_cpu, pred_radius_cpu, images_cpu, masks_gt_cpu, centroids_gt_cpu = get_centre_net_preds(model, val_loader, device)

    # plot
    for i, ax_row in enumerate(axes):
        ax_row[0].imshow(images_cpu[i, 1], cmap='terrain')
        ax_row[1].imshow(masks_gt_cpu[i, 0], cmap='gray', vmin=0, vmax=1)
        ax_row[2].imshow(centroids_gt_cpu[i, 0], cmap='viridis', vmin=0, vmax=1)

        im3 = ax_row[3].imshow(pred_centres_cpu[i, 0], cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im3, ax=ax_row[3], fraction=0.046)

        ax_row[4].imshow(masks_gt_cpu[i, 0], cmap='gray', vmin=0, vmax=1)
        peaks = peak_local_max(pred_centres_cpu[i, 0], min_distance=8, threshold_abs=0.3)
        for y, x in peaks:
            r = pred_radius_cpu[i, 0, y, x]
            ax_row[4].add_patch(patches.Circle((x, y), radius=r, edgecolor='red', facecolor='none', linewidth=1.5))

        for j, ax in enumerate(ax_row):
            ax.set_title(['Input', 'GT Mask', 'GT Centroid', 'Pred Heatmap', 'Pred Radius vs GT'][j], fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualise(model_type, model, val_loader, device):

    if model_type == 'segnet':
        plot_segmentation_predictions(val_loader, model, device)
    elif model_type == 'centrenet':
        plot_centrenet_predictions(model, val_loader, device)
    else:
        print('please enter a valid model type (segnet or centrenet)')


