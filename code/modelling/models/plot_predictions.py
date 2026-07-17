import matplotlib.pyplot as plt
from scipy.ndimage import label, center_of_mass, maximum_filter
import numpy as np
import torch
from skimage.feature import peak_local_max
import matplotlib.patches as patches
from mask_rcnn_plots import plot_preds as plot_maskrcnn_preds
from SegNet import SegNet
from MaskRCNN import MaskRCNN
from FPNCentreNet import FPNCentreNet, decode_multilevel_predictions
from matplotlib.patches import Circle

LEVEL_STRIDES = {
    "0": 4,
    "1": 8,
    "2": 16,
    "3": 32
}

class SegNetPlotter():

    def __init__(self, model, loader):

        self.model = model
        self.name = 'SegNet'
        self.loader = loader
        self.preds = self._get_preds(loader)

    def plot_preds(self, ax, idx, dem):
        ax.imshow(self.preds[idx])
        
    def _get_preds(self):
        images, _ = next(iter(self.loader))
        logits = self.model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.75).float()

        return preds

class MaskRCNNPlotter():

    def __init__(self, model, loader):

        self.model = model
        self.name = 'Mask-RCNN'
        self.loader = loader
        self.preds = self._get_preds(loader)

    def plot_preds(self, ax, idx, dem):
        ax.imshow(self.preds[idx])

    def _get_preds(self):

        images, _ = next(iter(self.loader))
        height, width = images.shape[-2:]
        output = self.model(images)
        scores = output["scores"].detach().cpu()
        raw_pred_masks = output["masks"].detach().cpu()
        keep = scores >= self.pred_args['score_threshold']
        pred_instance_masks = (
                    raw_pred_masks[keep, 0] >= self.pred_args[['mask_threshold']]
                )

        # convert from instance masks to combined masks.s
        if pred_instance_masks.shape[0] > 0:
            pred_masks = pred_instance_masks.any(dim=0)
        else:
            pred_masks = torch.zeros(
                (height, width),
                dtype=torch.bool,
            )
        
        return pred_masks


class FPNCentreNetPlotter():

    def __init__(self, model, loader):

        self.model = model
        self.name = 'FPN CentreNet'
        self.loader = loader
        self.preds = self._get_preds(loader)

    def plot_preds(self, ax, idx, dem):

        detections = self.preds[idx]

        for detection in detections:
            cy = detection["cy"]
            cx = detection["cx"]
            score = detection["score"]
            radius = detection.get("radius", np.nan)
            level = detection.get("level", "?")

            ax.scatter(
                cx,
                cy,
                marker="x",
                s=80,
                linewidths=2,
            )

            ax.text(
                cx + 4,
                cy - 4,
                f"{score:.2f}\nP{level}",
                fontsize=8,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )

            if np.isfinite(radius) and radius > 0:
                circle = Circle(
                    (cx, cy),
                    radius=radius,
                    fill=False,
                    linewidth=1.5,
                )
                ax.add_patch(circle)

        ax.set_title(f"Predictions: {len(detections)}")
        ax.axis("off")


    def _get_detections(self, images):
        preds = []
        images, _, _ = next(iter(self.loader))

        outputs = self.model(images)

        for i in len(self.loader):
            detections = decode_multilevel_predictions(
                        outputs=outputs,
                        image_index=i,
                        use_softplus_radius=True,
                    )
            preds.append(detections)

        return preds
            
class Plotter():

    def __init__(self, models: dict, loaders: dict, device):

        self.models = models
        print(models)
        self.loaders = loaders
        self.plotters = self._build_plotters()
        self.device = device

    def _build_plotters(self):

        plotters = []
        for model, loader in zip(self.models, self.loaders):

            if isinstance(model, SegNet):
                plotters.append(SegNetPlotter(model, loader))
                self.base_loader = loader
                self.has_base_loader = True
                self.base_loader_type = 'SegNet'
            elif isinstance(model, FPNCentreNet):
                plotters.append(FPNCentreNetPlotter(model, loader))
                if not self.has_base_loader:
                    self.base_loader = loader
                    self.has_base_loader = True
                    self.base_loader_type = 'FPNCN'
            elif isinstance(model, MaskRCNN):
                plotters.append(MaskRCNNPlotter(model, loader))
                if not self.has_base_loader:
                    self.base_loader = loader
                    self.base_loader_type = 'MRCNN'
            else:
                print('please enter a valid model type!')

        return plotters
    
    def get_base_images(self, idx):

        if self.base_loader_type == 'SegNet':
            images, masks = next(iter(self.base_loader))
            dem = images[idx][0]
            gt = masks[idx]

            return dem, gt
    
        elif self.base_loader_type == 'FPNCN':
            images, masks, targets = next(iter(self.base_loader))
            dem = images[idx][0]
            gt = masks[idx]

            return dem, gt
        
        elif self.base_loader_type == 'MRCNN':
            images, targets = next(iter(self.base_loader_type))
            dem = images[idx][0]
            gt = targets['masks'][idx]

            return dem, gt


    def plot(self):
        
        # build figure
        fig, axes = plt.subplots(len(self.loaders['MaskRCNN']), 2 + len(self.models))

        for i, rows in enumerate(axes):
            dem, gt = self.get_base_images(i)

            axes[i, 0].imshow(dem)
            
            axes[i, 1].imshow(gt)

            for j, plotter in enumerate(self.plotters):
                # j should start at 2
                plotter.plot_preds(axes[i, j + 2].imshow(), i, dem)
                axes[i, j + 2].set_title(f'{plotter.name} Preds')
        

        axes[i, 0].set_title('DEM')
        axes[i, 1].set_title('Target')

        for k, row in enumerate(axes):
            row[k].axis('off')

        plt.tight_layout()
        plt.show()
