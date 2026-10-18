import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
import random
import cv2
import numpy as np
from modelling.helpers import calculate_hillshade
import torch

import torch.nn.functional as F



_BANDS_TO_LOAD = ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=5,
    classes=1,
)

loss_fn = smp.losses.DiceLoss("binary")

class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.loss = loss_fn

    def forward(self, pred, target):
        return self.loss(pred, target)

# AI assistance with several refactors of dataset when changing augmentation and channel logic.
class UNetNDataset(Dataset):

    # remove path checking for time optimisation
    def __init__(self, all_paths, norm_stats, augment=True, min_instance_area=10,):

        self.augment = augment
        self.paths = all_paths
        self.min_instance_area = min_instance_area
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def pad_to(x, size=512):
        h, w = x.shape[-2:]
        return F.pad(x, (0, size - w, 0, size - h))

    def _crop_zoom(self, image, mask):
        h, w = mask.shape
        s = random.uniform(0.7, 0.9)
        ch, cw = int(h * s), int(w * s)
        y = random.randint(0, h - ch)
        x = random.randint(0, w - cw)

        image = image[:, y:y + ch, x:x + cw]
        mask = mask[y:y + ch, x:x + cw]

        image = np.stack([
            cv2.resize(np.ascontiguousarray(c), (w, h),
                       interpolation=cv2.INTER_LINEAR)
            for c in image
        ])
        mask = cv2.resize(np.ascontiguousarray(mask), (w, h),
                          interpolation=cv2.INTER_NEAREST)
        return image, mask
    
    # same augmentation changes as made in FPNCN
    def _spatial_augment(self, image, mask):
        """
        Apply identical spatial transformations to every channel and the mask.
        """
        image = image.copy()
        mask = mask.copy()

        if random.random() > 0.5:
            image = image[:, :, ::-1]
            mask = mask[:, ::-1]

        if random.random() > 0.5:
            image = image[:, ::-1, :]
            mask = mask[::-1, :]

        k = random.choice([0, 1, 2, 3])
        if k:
            image = np.rot90(image, k, axes=(1, 2))
            mask = np.rot90(mask, k)

        if random.random() < 0.5:
            image, mask = self._crop_zoom(image, mask)

        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
        )

    def handle_outliers(self, image, li, band_indices):
        """
        Handle outliers in the DEM, Slope, RR, and Laplace bands by clipping to the 1st and 99th percentiles.
        """
        for band in ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']:
            band_index = band_indices.index(li[band])
            band_data = image[band_index]

            lower_bound = np.percentile(band_data, 0.5)
            upper_bound = np.percentile(band_data, 99.5)

            band_data = np.clip(band_data, lower_bound, upper_bound)
            image[band_index] = band_data

        return image

    def _normalise(self, image, li, band_indices):

        # DEM
        dem = image[band_indices.index(li["DEM"])]

        valid = np.isfinite(dem)

        dem_median = np.median(dem[valid])
        dem_rel = dem - dem_median

        # slope log scaled, then Robust scaling
        slope = image[band_indices.index(li["DEM_SLOPE"])]
        rr = image[band_indices.index(li["RR"])]
        laplace = image[band_indices.index(li["LAPLACE"])]

        slope = np.power(slope, 0.25)
        hills = np.cos(np.radians(45)) - image[-1]
        hills = np.power(hills, 0.25)

        # Robust scaling all (more efficient):
        for band in ['DEM', 'SLOPE', 'RR', 'LAPLACE']:
            median = self.norm_stats[band]['median']
            iqr = self.norm_stats[band]['iqr']

            if band == 'DEM':
                dem_rel = (dem_rel - median) / iqr
            elif band == 'SLOPE':
                slope = (slope - median) / iqr         
            elif band == 'RR':
                rr = (rr - median) / iqr
            elif band == 'LAPLACE':
                laplace = (laplace - median) / iqr
            elif band == 'HILLSHADE':
                hills = (hills - median) / iqr

        return np.stack([dem_rel, slope, rr, laplace, hills], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        path = self.paths[idx]
        d = np.load(path, allow_pickle=True)
 
        layer_names = list(d["layer_names"])
        li = {name: i for i, name in enumerate(layer_names)}
        band_indices = [li[b] for b in _BANDS_TO_LOAD]

        image = d["image"][band_indices].astype(np.float32)
        mask = d["labels"].astype(np.float32)

        if self.augment:
            # rotate / flip
            image, mask = self._spatial_augment(image, mask)

        image = self.handle_outliers(image, li, band_indices)

        # Calculate hillshade
        hillshade = calculate_hillshade(image[band_indices.index(li["DEM"])])
        image = np.concatenate([image, hillshade[np.newaxis, ...]], axis=0)
        
        image = self._normalise(image, li, band_indices)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(1) 

        image = self.pad_to(image, size=512)
        mask = self.pad_to(mask, size=512)

        return image, mask