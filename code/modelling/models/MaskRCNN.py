import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import glob
import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from scipy.ndimage import label, find_objects, gaussian_filter
from torchvision.models.detection.rpn import RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
 
class MaskRCNN(nn.Module):
    """
    Wrapper around torchvision's Mask R-CNN
    """
 
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        anchor_sizes=((16,), (32,), (64,), (128,), (256,)), # default
        aspect_ratios=((0.5, 1.0, 2.0),) * 5, # default
        trainable_backbone_layers=3,
        loss_weights=None,
        freeze=True,
        train_schedule = False
    ):
        super().__init__()
 
        weights = "DEFAULT" if pretrained else None
        self.model = maskrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone="DEFAULT" if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        in_channels = 4
 
        self._replace_input_conv(in_channels)
        self.model.transform.image_mean = [0.0] * in_channels
        self.model.transform.image_std = [1.0] * in_channels
        self._set_anchor_generator(anchor_sizes, aspect_ratios)
        self._replace_heads(num_classes)
        self.train_schedule = train_schedule
 
        # multiplies each named loss the model returns, e.g. {"loss_box_reg": 2.0}
        self.loss_weights = loss_weights or {}

        if freeze:
            self.freeze_backbone(True)

        self.set_training_stage(stage=1)

    def set_training_stage(self, stage):
        # Freeze backbone.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Heads always train.
        for parameter in self.rpn.parameters():
            parameter.requires_grad = True

        for parameter in self.roi_heads.parameters():
            parameter.requires_grad = True

        # New input convolution.
        for parameter in self.backbone.body.conv1.parameters():
            parameter.requires_grad = True

        if stage >= 2:
            # Adapt the feature pyramid.
            for parameter in self.backbone.fpn.parameters():
                parameter.requires_grad = True

        if stage >= 3:
            # Adapt highest-level ResNet features.
            for parameter in self.backbone.body.layer4.parameters():
                parameter.requires_grad = True

        if stage >= 4:
            for parameter in self.backbone.body.layer3.parameters():
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

 
    def freeze_backbone(self, freeze=True):

        backbone = self.model.backbone

        for param in backbone.parameters():
            param.requires_grad = not freeze
 
    def set_loss_weights(self, weights: dict):
        self.loss_weights.update(weights)
 
    def forward(self, images, targets=None):
        if self.training:
            loss_dict = self.model(images, targets)
 
            weighted = {
                k: v * self.loss_weights.get(k, 1.0) for k, v in loss_dict.items()
            }
            return weighted
 
        return self.model(images, targets)

_BANDS_TO_LOAD = ['DEM', 'DEM_SLOPE', 'RR', 'LAPLACE']

class MaskRCNNDataset(Dataset):

    def __init__(
        self,
        all_paths,
        tile_size=512,
        skip_partial=True,
        augment=False,
        min_instance_area=20,
    ):
        self.augment = augment
        self.paths = []
        self.min_instance_area = min_instance_area

        skipped_shape = 0
        skipped_missing = 0

        for p in sorted(all_paths):
            d = np.load(p, allow_pickle=True)

            if skip_partial:
                _, h, w = d["image"].shape
                if h != tile_size or w != tile_size:
                    skipped_shape += 1
                    continue

            layer_names = list(d["layer_names"]) if "layer_names" in d else []
            if any(b not in layer_names for b in _BANDS_TO_LOAD):
                skipped_missing += 1
                continue

            self.paths.append(p)

        print(
            f"Found {len(self.paths)} tiles "
            f"({skipped_shape} partial, {skipped_missing} missing bands skipped)"
        )

    def __len__(self):
        return len(self.paths)

    def _augment(self, image, mask):
        image, mask = image.copy(), mask.copy()

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

        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, size=image[0].shape).astype(np.float32)
            image[0] = image[0] + noise

        if random.random() > 0.5:
            sigma = random.uniform(0.3, 0.8)
            image[0] = gaussian_filter(image[0], sigma=sigma)

        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            image[0] = image[0] * scale

        return image, mask

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

    def __getitem__(self, idx):
        path = self.paths[idx]
        d = np.load(path, allow_pickle=True)

        layer_names = list(d["layer_names"])
        li = {name: i for i, name in enumerate(layer_names)}
        band_indices = [li[b] for b in _BANDS_TO_LOAD]

        image = d["image"][band_indices].astype(np.float32)
        mask = d["labels"].astype(np.float32)

        for i, name in enumerate(_BANDS_TO_LOAD):
          band = image[i].astype(np.float32)

          if name == ["DEM_SLOPE"]:
              band = np.log1p(np.maximum(band, 0))

          elif name == ['LAPLCE']:
              band = np.sign(band) * np.log1p(np.abs(band))

          image[i] = (band - band.mean()) / (band.std() + 1e-6)

        if self.augment:
            image, mask = self._augment(image, mask)

        boxes, labels, masks, areas, iscrowd = self._mask_to_instances(mask)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        return image, target
    
def collate_fn(batch):
    return tuple(zip(*batch))

def get_rcnn_loaders(train_paths, val_paths):

    train_dataset = MaskRCNNDataset(train_paths, augment=True)
    val_dataset = MaskRCNNDataset(val_paths, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    return val_loader, train_loader

import numpy as np
from scipy.ndimage import label, center_of_mass



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


def evaluate_maskrcnn_metrics(model, val_loader, device, mask_thresh=0.5, score_thresh=0.5):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    ptp = pfp = pfn = 0

    ious = []

    with torch.no_grad():
        for images, targets in val_loader:
            images_device = [image.to(device) for image in images]
            outputs = model(images_device)

            for image, output, target_dict in zip(images, outputs, targets):

                height, width = image.shape[-2:]


                # Ground-truth combined binary mask
                gt_instance_masks = target_dict["masks"].detach().cpu()

                if gt_instance_masks.shape[0] > 0:
                    target_mask = gt_instance_masks.bool().any(dim=0)
                else:
                    target_mask = torch.zeros(
                        (height, width),
                        dtype=torch.bool,
                    )

                scores = output["scores"].detach().cpu()
                raw_pred_masks = output["masks"].detach().cpu()

                keep = scores >= score_thresh

                # AI assistance with figuring out channels / dimensions for metric calculation 
                # [N_pred, 1, H, W] -> [N_pred, H, W]
                pred_instance_masks = (
                    raw_pred_masks[keep, 0] >= mask_thresh
                )

                # [N_gt, H, W]
                gt_instance_masks = (
                    target_dict["masks"]
                    .detach()
                    .cpu()
                    .bool()
                )

                tp, fp, fn = object_f1_from_instance_masks(
                    pred_instance_masks,
                    gt_instance_masks,
                )

                total_fn += fn
                total_fp += fp
                total_tp += tp

                # Combined predicted mask: [H, W]
                if pred_instance_masks.shape[0] > 0:
                    pred_mask = pred_instance_masks.any(dim=0)
                else:
                    pred_mask = torch.zeros(
                        (height, width),
                        dtype=torch.bool,
                    )

                # Pixel-level metrics using Boolean logic. AI assistance (ChatGPT) with boolean logic 
                ptp += torch.logical_and(
                    pred_mask,
                    target_mask,
                ).sum().item()

                pfp += torch.logical_and(
                    pred_mask,
                    ~target_mask,
                ).sum().item()

                pfn += torch.logical_and(
                    ~pred_mask,
                    target_mask,
                ).sum().item()

                # IoU
                pred_union = pred_mask.cpu().numpy()
                gt_union = target_mask.cpu().numpy()

                ious.append(binary_iou(pred_union, gt_union))

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)


    p_precision = (ptp + 1e-6) / (ptp + pfp + 1e-6)
    p_recall = (ptp + 1e-6) / (ptp + pfn + 1e-6)
    p_f1 = 2 * p_precision * p_recall / (p_precision + p_recall + 1e-8) 

    mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0

    return precision, recall, f1, mean_iou, p_precision, p_recall, p_f1

# ai assistance with mask rcnn train functionality and helper functions. original code repo from paper is hard to understand
# ChatGPT and Claude Sonnet 5 used - whenever I have said AI is used its these models.


def build_staged_optimizer(model, weight_decay=1e-4):
    return torch.optim.AdamW(
        [
            {
                "name": "rpn",
                "params": model.rpn.parameters(),
                "lr": 1e-3,
            },
            {
                "name": "roi_heads",
                "params": model.roi_heads.parameters(),
                "lr": 1e-3,
            },
            {
                "name": "conv1",
                "params": model.backbone.body.conv1.parameters(),
                "lr": 5e-6,
            },
            {
                "name": "fpn",
                "params": model.backbone.fpn.parameters(),
                "lr": 0.0,
            },
            {
                "name": "layer4",
                "params": model.backbone.body.layer4.parameters(),
                "lr": 0.0,
            },
            {
                "name": "layer3",
                "params": model.backbone.body.layer3.parameters(),
                "lr": 0.0,
            },
        ],
        weight_decay=weight_decay,
    )

STAGE_LRS = {
    1: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "conv1": 1e-5,
        "fpn": 0.0,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    2: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "conv1": 5e-6,
        "fpn": 1e-5,
        "layer4": 0.0,
        "layer3": 0.0,
    },
    3: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "conv1": 1e-6,
        "fpn": 3e-6,
        "layer4": 1e-6,
        "layer3": 0.0,
    },
    4: {
        "rpn": 1e-3,
        "roi_heads": 1e-3,
        "conv1": 5e-7,
        "fpn": 1e-6,
        "layer4": 5e-7,
        "layer3": 1e-7,
    },
}


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

def rcnn_train(model, _, train_loader, val_loader, epochs):

    import copy
    from tqdm.auto import tqdm

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
            
        if epoch == 40:
            current_stage = 2
            apply_training_stage(model, optimizer, current_stage)

        elif epoch == 55:
            current_stage = 3
            apply_training_stage(model, optimizer, current_stage)

        elif epoch == 70:
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

            loss_dict = model(images, targets)

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
            mask_thresh=0.625,
            score_thresh=0.6
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
            f"Pixel F1: {p_f1:.4f} | "

        )

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

        if bad_epochs >= patience:
            print("early stopping")
            break

    model.load_state_dict(best_state)

    # val losses done internally, so return None instead
    return None, train_losses, val_precisions, val_recalls, val_f1s, model, val_loader 
