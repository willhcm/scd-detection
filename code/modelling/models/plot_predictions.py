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

class SegNetPlotter():

    def __init__(self, model, loader):

        self.model = model
        self.name = 'SegNet'
        self.loader = loader
        self.preds = self._get_preds(loader)

    def plot_preds(self, ax, idx):
        ax.imshow(self.preds[idx])
        
    def _get_preds(self, images):
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

    def plot_preds(self, ax, idx):
        ...

    def _get_preds(self, i):
        ...


class FPNCentreNetPlotter():

    def __init__(self, model, loader):

        self.model = model
        self.name = 'FPN CentreNet'
        self.loader = loader
        self.preds = self._get_preds(loader)

    def plot_preds(self, ax, idx):
        ...

    def _get_detections(self, images):
        ...


class Plotter():

    def __init__(self, models: list, loaders: list, device):

        self.models = models
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


    def plot(self, images=None):
        
        # build figure

        fig, axes = plt.subplots(len(images), 2 + len(self.models))

        for i, rows in enumerate(axes):
            # need to figure out how to plot dem and gt before getting plotter-specific preds.
            # cant specify segnet loader as it might not always be being used.
            dem, gt = self.get_base_images(i)

            axes[i, 0].imshow(dem)
            
            axes[i, 1].imshow(gt)

            for j, (plotter, loader) in enumerate(self.plotters):
                # j should start at 2
                plotter.plot_preds(axes[i, j + 2].imshow(), i)
                axes[i, j + 2].set_title(f'{plotter.name} Preds')
        

        axes[i, 0].set_title('DEM')
        axes[i, 1].set_title('Target')

        for k, row in enumerate(axes):
            row[k].axis('off')

        plt.tight_layout()
        plt.show()

            


        

        