import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, maximum_filter
import numpy as np
import torch
from skimage.feature import peak_local_max
import matplotlib.patches as patches
from mask_rcnn_plots import plot_preds as plot_maskrcnn_preds
from helpers import pred_centroids_for_plot, decode_centernet_predictions
from SegNet import SegNet
from MaskRCNN import MaskRCNN
from FPNCentreNet import FPNCentreNet

CHANNELS = ['DEM',
            'Slope',
            'Residual Relief',
            'LAPLACE']

# need to completely revamp this plotting file, it is very poorly implemented and not model-agnostic!
# make class based!


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

    fig, axes = plt.subplots(8, 6, figsize=(4 * 3, 14))

    images, masks, preds = get_seg_preds(val_loader, model, device)
    images, masks, preds = images[:8], masks[:8], preds[:8]
    for i in range(len(preds)):
        img = images[i].cpu().numpy()
        gt = masks[i, 0].cpu().numpy()
        pred = preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img[0], cmap='terrain') # DEM
        axes[i, 1].imshow(img[2], cmap='terrain') # Slope
        axes[i, 2].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].imshow(pred, cmap='gray', vmin=0, vmax=1)

        # annotations for interpretation

        # GT outline
        axes[i, 3].contour(gt, levels=[0.5], colors='lime', linewidths=1)

        # Predicted object outline
        axes[i, 3].contour(pred, levels=[0.5], colors='cyan', linewidths=1)

        points = pred_centroids_for_plot(pred, gt, min_pred_area=20)

        # green or red for predictrion success (TP, FP)
        for cx, cy, detected in points:
            colour = 'lime' if detected else 'red'
            axes[i, 2].scatter(cx, cy, c=colour, s=40, marker='x', linewidths=2)

        for j in range(6):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_all(segnet, centrenet, seg_loader, centre_loader, device):

    seg_images, seg_masks, seg_preds = get_seg_preds(seg_loader, segnet, device)
    pred_centres_cpu, pred_radius_cpu, pred_offset_cpu, _, masks_gt_cpu, centroids_gt_cpu = get_centre_net_preds(centrenet, 
                                                                                                        centre_loader,
                                                                                                        device=device)
    
    fig, axes = plt.subplots(4, 7, figsize=(15, 12))

    for i in range(len(seg_preds)):
        img = seg_images[i].cpu().numpy()
        gt = seg_masks[i, 0].cpu().numpy()
        pred = seg_preds[i, 0].cpu().numpy()

        axes[i, 0].imshow(img[0], cmap='terrain') # DEM
        axes[i, 1].imshow(img[2], cmap='terrain') # Slope
        axes[i, 2].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].imshow(pred, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title('Segmentation Predictions')

        # annotations for interpretation

        # GT outline
        axes[i, 3].contour(gt, levels=[0.5], colors='lime', linewidths=0.5)

        # Predicted object outline
        axes[i, 3].contour(pred, levels=[0.5], colors='cyan', linewidths=0.2)

        points = pred_centroids_for_plot(pred, gt, min_pred_area=20)

        # green or red for predictrion success (TP, FP)
        for cx, cy, detected in points:
            colour = 'lime' if detected else 'red'
            axes[i, 2].scatter(cx, cy, c=colour, s=40, marker='x', linewidths=0.5)

        for j in range(6):
            axes[i, j].axis('off')

        # centrenet preds

        detections = decode_centernet_predictions(pred_centres_cpu[i, 0], pred_radius_cpu[i, 0], pred_offset_cpu[i])

        axes[i, 4].imshow(centroids_gt_cpu[i, 0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 4].set_title('Centroid GT')

        axes[i, 5].imshow(pred_centres_cpu[i, 0], cmap='viridis', vmin=0, vmax=1)
        axes[i, 5].set_title('Centroid Predictions')


        axes[i, 6].imshow(masks_gt_cpu[i, 0], cmap='gray', vmin=0, vmax=1)
        for det in detections:
            axes[i, 6].add_patch(patches.Circle((det['cx'], det['cy']), radius=det['radius'],
                                                  edgecolor='red', facecolor='none', linewidth=1.5))
        axes[i, 6].set_title('Centroid Pred vs GT')


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
    pred_offset_cpu = pred_offset.cpu().numpy()

    return pred_centres_cpu, pred_radius_cpu, pred_offset_cpu, images_cpu, masks_gt_cpu, centroids_gt_cpu

# need to adjust for strided (128x128 centrenet predictions!)
def plot_centrenet_predictions(model, val_loader, device):

    fig, axes = plt.subplots(len(val_loader), 5, figsize=(20, 3 * len(val_loader)))

    # get preds and inputs
    pred_centres_cpu, pred_radius_cpu, pred_offset_cpu, images_cpu, masks_gt_cpu, centroids_gt_cpu = get_centre_net_preds(model, val_loader, device)

    # plot
    for i, ax_row in enumerate(axes):
        detections = decode_centernet_predictions(pred_centres_cpu, pred_radius_cpu, pred_offset_cpu)
        ax_row[0].imshow(images_cpu[i, 1], cmap='terrain')
        ax_row[1].imshow(masks_gt_cpu[i, 0], cmap='gray', vmin=0, vmax=1)
        ax_row[2].imshow(centroids_gt_cpu[i, 0], cmap='viridis', vmin=0, vmax=1)

        im3 = ax_row[3].imshow(pred_centres_cpu[i, 0], cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(im3, ax=ax_row[3], fraction=0.046)

        ax_row[4].imshow(masks_gt_cpu[i, 0], cmap='gray', vmin=0, vmax=1)
        for det in detections:
            axes[i, 4].add_patch(patches.Circle((det['cx'], det['cy']), radius=det['radius'],
                                                  edgecolor='red', facecolor='none', linewidth=1.5))

        for j, ax in enumerate(ax_row):
            ax.set_title(['Input', 'GT Mask', 'GT Centroid', 'Pred Heatmap', 'Pred Radius vs GT'][j], fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualise(model_type, model, val_loader, device):

    if model_type == 'Segmentation' or model_type == 'DaynacModel':
        plot_segmentation_predictions(val_loader, model, device)
    elif model_type == 'CentreNet':
        plot_centrenet_predictions(model, val_loader, device)
    elif model_type == 'MaskRCNN':
        plot_maskrcnn_preds(model, val_loader, device)
    else:
        print('please enter a valid model type (Segmentation or CentreNet)')

# making above implementation to be all Object oriented and clean (its really annoying me)


class SegNetPlotter():

    def __init__(self, model):

        self.model = model

    def _plot_preds(self, ax):
        ...

    def _get_preds(self, images):
        ...

class MaskRCNNPlotter():

    def __init__(self, model):

        self.model = model

    def _plot_preds(self, ax):
        ...

    def _get_preds(self, images):
        ...


class FPNCentreNetPlotter():

    def __init__(self, model):

        self.model = model

    def _plot_preds(self, ax):
        ...

    def _get_detections(self, images):
        ...


class Plotter():

    def __init__(self, models: list, sample_paths = None):

        self.models = models
        self._build_models()

        if sample_paths:
            images = []
            for p in sample_paths:
                d = np.load(p)
                images.append(d)
            
            self.tiles = images

    def _build_models(self):

        for model in self.models:

            if isinstance(model, SegNet):
                self.SNPlotter = SegNetPlotter(model)
            elif isinstance(model, MaskRCNN):
                self.MRCNNPlotter = MaskRCNNPlotter(model)
            elif isinstance(model, FPNCentreNet):
                self.FPNCNPlotter = FPNCentreNetPlotter(model)
            else:
                print('please enter a valid model type!')

