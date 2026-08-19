from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from scipy.ndimage import label, find_objects, gaussian_filter
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from helpers import calculate_hillshade
from code.helpers.MaskRCNNFunctions import evaluate_maskrcnn_metrics, build_staged_optimizer, apply_training_stage
import copy
from tqdm.auto import tqdm

_BANDS_TO_LOAD = ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']

# ChatGPT assistance with FiLM encodings for resolution
# specifically with incorporation into pre-existing MaskRCNN wrapper and condition-setting for forward()
class ResolutionFiLM(nn.Module):
    """
    Generates per-channel gamma/beta from a scalar resolution value.
    """

    def __init__(self, out_channels=256, hidden=64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels * 2),
        )
        self.out_channels = out_channels

        nn.init.zeros_(self.gen[-1].weight)
        nn.init.zeros_(self.gen[-1].bias)

    def forward(self, cond):
        gamma, beta = self.gen(cond).chunk(2, dim=1)
        return gamma, beta


class FiLMBackbone(nn.Module):
    """
    Wraps torchvision's BackboneWithFPN so every FPN level gets modulated
    by a resolution conditioning vector before being handed to the RPN/ROI heads.
    FiLM.
    """

    def __init__(self, backbone_with_fpn):
        super().__init__()
        self.body = backbone_with_fpn.body
        self.fpn = backbone_with_fpn.fpn
        self.out_channels = backbone_with_fpn.out_channels

        self.film = ResolutionFiLM(out_channels=self.out_channels)
        self._cond = None

    def set_condition(self, cond):
        # cond: (B, 1) tensor, same device + batch order as the images
        self._cond = cond

    def forward(self, x):
        feats = self.fpn(self.body(x))

        if self._cond is None:
            return feats

        gamma, beta = self.film(self._cond)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        return {k: v * (1 + gamma) + beta for k, v in feats.items()}

class MaskRCNN(nn.Module):
    """
    Wrapper around torchvision's Mask R-CNN
    """
 
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        anchor_sizes= ((20,), (40,), (62,), (120,), (220,)), # decreased to match SCD size population.
        aspect_ratios=((0.75, 1.0, 1.35),) * 5, # default, elongated doesn't match subcircular appearance (square bounding boxes)
        loss_weights = None, # depreciated but kept for future.
    ):
        super().__init__()

        # changed proposal thresholds 
        weights = "DEFAULT" if pretrained else None
        self.model = maskrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone="DEFAULT" if pretrained else None,
            rpn_fg_iou_thresh=0.70, # rpn proposal thresholds
            rpn_bg_iou_thresh=0.30,
            box_fg_iou_thresh=0.55, # roi classification threshold
            box_bg_iou_thresh=0.45,
            trainable_backbone_layers=3
        )

        in_channels = 5
 
        self._replace_input_conv(in_channels)
        self.model.transform.image_mean = [0.0] * in_channels
        self.model.transform.image_std = [1.0] * in_channels
        self._set_anchor_generator(anchor_sizes, aspect_ratios)
        self._replace_heads(num_classes)

        # add FiLM conditioning to backbone to encode resolution
        self.model.backbone = FiLMBackbone(self.model.backbone)
 
        # multiplies each named loss the model returns
        self.loss_weights = loss_weights or {}

        self.set_training_stage(stage=1)

    def set_training_stage(self, stage):

        # Freeze backbone.
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad = False

        for parameter in self.model.backbone.film.parameters():
            parameter.requires_grad = True

        # Heads always train.
        for parameter in self.model.rpn.parameters():
            parameter.requires_grad = True

        for parameter in self.model.roi_heads.parameters():
            parameter.requires_grad = True

        # New input convolution.
        for parameter in self.model.backbone.body.conv1.parameters():
            parameter.requires_grad = True

        if stage >= 2:
            # Adapt the feature pyramid.
            for parameter in self.model.backbone.fpn.parameters():
                parameter.requires_grad = True

        if stage >= 3:
            # Adapt highest-level ResNet features.
            for parameter in self.model.backbone.body.layer4.parameters():
                parameter.requires_grad = True

        if stage >= 4:
            for parameter in self.model.backbone.body.layer3.parameters():
                parameter.requires_grad = True

    def _replace_input_conv(self, in_channels):
        old_conv = self.model.backbone.body.conv1

        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            if in_channels == 3:
                new_conv.weight.copy_(old_conv.weight)
            else:
                # Average RGB filters, then repeat for all input channels.
                mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))

        self.model.backbone.body.conv1 = new_conv
 

    def _set_anchor_generator(self, sizes, aspect_ratios):
        anchor_gen = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
        self.model.rpn.anchor_generator = anchor_gen
        num_anchors = anchor_gen.num_anchors_per_location()[0]
        in_channels = self.model.backbone.out_channels

        self.model.rpn.head = RPNHead(in_channels, num_anchors)
 
    def _replace_heads(self, num_classes):
        box_in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)
 
        mask_in_features = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        mask_hidden_layers = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            mask_in_features, mask_hidden_layers, num_classes
        )
 
    def set_loss_weights(self, weights: dict):
        self.loss_weights.update(weights)
 
    def forward(self, images, resolutions= None , targets=None):
        self.model.backbone.set_condition(resolutions)

        if self.training:
            loss_dict = self.model(images, targets)
 
            weighted = {
                k: v * self.loss_weights.get(k, 1.0) for k, v in loss_dict.items()
            }
            return weighted
 
        return self.model(images, targets)

# AI assistance with several refactors of dataset when changing augmentation and channel logic.
class MaskRCNNDataset(Dataset):

    # remove path checking for time optimisation
    def __init__(self, all_paths, augment=False, min_instance_area=20):

        self.augment = augment
        self.paths = all_paths
        self.min_instance_area = min_instance_area

    def __len__(self):
        return len(self.paths)

    def _mask_to_instances(self, mask):
        binary = mask > 0

        labelled, n = label(binary)
        objects = find_objects(labelled)

        instance_masks = []
        boxes = []
        areas = []

        for instance_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue

            inst_mask = labelled == instance_id
            area = inst_mask.sum()

            if area < self.min_instance_area:
                continue

            y_slice, x_slice = slc
            y_min, y_max = y_slice.start, y_slice.stop
            x_min, x_max = x_slice.start, x_slice.stop

            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            instance_masks.append(inst_mask.astype(np.uint8))
            areas.append(area)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(np.stack(instance_masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        return boxes, labels, masks, areas, iscrowd
    
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

        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
        )
    
    def _augment_dem(self, dem):
        """
        Apply perturbations while the DEM is still in elevation units.
        """
        dem = dem.copy().astype(np.float32)


        if random.random() > 0.5:
            sigma = random.uniform(0.3, 0.8)
            dem = gaussian_filter(dem, sigma=sigma).astype(np.float32)

        if random.random() > 0.5:
            # 2% of the DEM's local standard deviation, rather than 0.02 metres
            noise_std = 0.02 * (dem.std() + 1e-6)
            noise = np.random.normal(
                0.0,
                noise_std,
                size=dem.shape,
            ).astype(np.float32)

            dem += noise

        return dem.astype(np.float32)

    @staticmethod
    def _normalise_band(band, name):
        band = band.astype(np.float32)

        if name in {"DEM_SLOPE",  "DEM"}:
            transformed = band
        else:
            transformed = np.sign(band) * np.log1p(np.abs(band))

        median = np.median(transformed)
        q75, q25 = np.percentile(transformed, [75, 25])
        iqr = q75 - q25
        scaled = (transformed - median) / (iqr + 1e-6)

        return scaled

    def __getitem__(self, idx):
        path = self.paths[idx]
        d = np.load(path, allow_pickle=True)
        res = d['res']
        res_value = np.log(res) / np.log(15.0)

        layer_names = list(d["layer_names"])
        li = {name: i for i, name in enumerate(layer_names)}
        band_indices = [li[b] for b in _BANDS_TO_LOAD]

        image = d["image"][band_indices].astype(np.float32)
        mask = d["labels"].astype(np.float32)

        # safe for any adjustments
        dem_idx = _BANDS_TO_LOAD.index("DEM")

        # dem in metres
        raw_dem = image[dem_idx].copy()

        if self.augment:
            # removed physical pertubations for now 
            # rotate / flip
            image, mask = self._spatial_augment(image, mask)

            # Retrieve the spatially transformed raw DEM.
            raw_dem = image[dem_idx]

        # Hillshade is calculated from the unnormalised DEM.
        hillshade = calculate_hillshade(raw_dem).astype(np.float32)

        # Normalise bands.
        for i, name in enumerate(_BANDS_TO_LOAD):
            image[i] = self._normalise_band(image[i], name)

        # normalise hillshade seperately.
        hillshade = (
            hillshade - hillshade.mean()
        ) / (
            hillshade.std() + 1e-6
        )

        image = np.concatenate(
            [image, hillshade[None, :, :]],
            axis=0,
        ).astype(np.float32)
        
        boxes, labels, masks, areas, iscrowd = self._mask_to_instances(mask)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd,
            "resolution": torch.tensor([res_value], dtype=torch.float32)
        }

        image = torch.from_numpy(image)

        return image, target

def rcnn_train(model, _, train_loader, val_loader, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-4,
        weight_decay=1e-4
    )

    EPOCHS = epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
        )

    best_f1 = -1.0
    best_state = None
    best_epoch = -1

    train_losses = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    val_ious = []
    patience = 20
    current_stage = 1
    bad_epochs = 0
    optimizer = build_staged_optimizer(
    model,
    weight_decay=1e-4,
)
    apply_training_stage(model, optimizer, current_stage)

    for epoch in range(EPOCHS):
            
        if epoch == 25:
            current_stage = 2
            apply_training_stage(model, optimizer, current_stage)

        elif epoch == 30:
            current_stage = 3
            apply_training_stage(model, optimizer, current_stage)

        elif epoch == 40:
            current_stage = 4
            apply_training_stage(model, optimizer, current_stage)


        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, targets in pbar:
            images = [img.to(device) for img in images]

            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            resolutions = torch.stack([t["resolution"] for t in targets]).to(device)
            
            loss_dict = model(images, resolutions, targets)

            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

            pbar.set_postfix({
                "loss": float(losses.detach().cpu())
            })

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        precision, recall, val_f1, val_iou, p_precision, p_recall, p_f1 = evaluate_maskrcnn_metrics(
            model,
            val_loader,
            device,
            mask_thresh=0.55,
            score_thresh=0.77
        )

        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(val_f1)
        val_ious.append(val_iou)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"Val P: {precision:.4f} | "
            f"Val R: {recall:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Pixel P: {p_precision:.4f} | "
            f"Pixel R: {p_recall:.4f} | "
            f"Pixel F1: {p_f1:.4f} | ")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

            torch.save({
                "epoch": best_epoch,
                "model_state_dict": best_state,
                "best_f1": best_f1,
                "train_losses": train_losses,
                "val_precisions": val_precisions,
                "val_recalls": val_recalls,
                "val_f1s": val_f1s,
            }, "/content/drive/MyDrive/IRP/models/best_maskrcnn_scd.pt")

            print(f"Saved new best model at epoch {best_epoch} with F1={best_f1:.4f}")
            bad_epochs = 0
        else:
            bad_epochs += 1

        scheduler.step()

        #if bad_epochs >= patience:
            #print("early stopping")
            #break

    model.load_state_dict(best_state)

    # val losses done internally, so return None instead
    return None, train_losses, val_precisions, val_recalls, val_f1s, model, val_loader 
