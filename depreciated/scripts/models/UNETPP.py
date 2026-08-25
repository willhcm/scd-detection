from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, maximum_filter, gaussian_filter
from datetime import datetime
from scipy.ndimage import label, center_of_mass
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from depreciated.scripts.misc.blocks import DoubleConv, ASPP
from helpers import object_centroid_metrics


# AI assistance with UNET++ code generation. used as a comparison to UNET architecture to see benefits of a slighlty more expressive
# and complex model. not used for any deployment-level prediction tasks. 

class UNetPP(nn.Module):

    def __init__(
        self,
        in_channels=4,
        base_filters=32,
        deep_supervision=False,
        use_aspp=True,
        dropout=0.2,
    ):
        super().__init__()
        f = base_filters
        self.deep_supervision = deep_supervision
        self.use_aspp = use_aspp

        # encoder column x_i0
        self.conv00 = DoubleConv(in_channels, f)
        self.conv10 = DoubleConv(f, f * 2)
        self.conv20 = DoubleConv(f * 2, f * 4)
        self.conv30 = DoubleConv(f * 4, f * 8)
        self.conv40 = DoubleConv(f * 8, f * 16)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        if use_aspp:
            self.aspp40 = ASPP(f * 16, f * 16)
            self.drop40 = nn.Dropout2d(p=dropout)
        else:
            self.aspp40 = nn.Identity()
            self.drop40 = nn.Identity()

        # nested decoder nodes
        self.conv01 = DoubleConv(f + f * 2, f)
        self.conv11 = DoubleConv(f * 2 + f * 4, f * 2)
        self.conv21 = DoubleConv(f * 4 + f * 8, f * 4)
        self.conv31 = DoubleConv(f * 8 + f * 16, f * 8)

        self.conv02 = DoubleConv(f * 2 + f * 2, f)  
        self.conv12 = DoubleConv(f * 4 + f * 4, f * 2) 
        self.conv22 = DoubleConv(f * 8 + f * 8, f * 4)

        self.conv03 = DoubleConv(f * 3 + f * 2, f) 
        self.conv13 = DoubleConv(f * 6 + f * 4, f * 2) 

        self.conv04 = DoubleConv(f * 4 + f * 2, f) 

        if deep_supervision:
            self.out1 = nn.Conv2d(f, 1, kernel_size=1)
            self.out2 = nn.Conv2d(f, 1, kernel_size=1)
            self.out3 = nn.Conv2d(f, 1, kernel_size=1)
            self.out4 = nn.Conv2d(f, 1, kernel_size=1)
        else:
            self.out = nn.Conv2d(f, 1, kernel_size=1)

    @staticmethod
    def _cat(*xs):
        """Upsampling can be off by one pixel for odd shapes; crop/pad to first tensor."""
        ref = xs[0]
        H, W = ref.shape[-2:]
        out = []
        for x in xs:
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
            out.append(x)
        return torch.cat(out, dim=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool(x00))
        x20 = self.conv20(self.pool(x10))
        x30 = self.conv30(self.pool(x20))
        x40 = self.conv40(self.pool(x30))
        x40 = self.drop40(x40)
        x40 = self.aspp40(x40)

        x01 = self.conv01(self._cat(x00, self.up(x10)))
        x11 = self.conv11(self._cat(x10, self.up(x20)))
        x21 = self.conv21(self._cat(x20, self.up(x30)))
        x31 = self.conv31(self._cat(x30, self.up(x40)))

        x02 = self.conv02(self._cat(x00, x01, self.up(x11)))
        x12 = self.conv12(self._cat(x10, x11, self.up(x21)))
        x22 = self.conv22(self._cat(x20, x21, self.up(x31)))

        x03 = self.conv03(self._cat(x00, x01, x02, self.up(x12)))
        x13 = self.conv13(self._cat(x10, x11, x12, self.up(x22)))

        x04 = self.conv04(self._cat(x00, x01, x02, x03, self.up(x13)))

        if self.deep_supervision:
            return [self.out1(x01), self.out2(x02), self.out3(x03), self.out4(x04)]

        return self.out(x04)


def main_logits(out):
    """Return final logits whether model returns Tensor or deep-supervision list."""
    if isinstance(out, (list, tuple)):
        return out[-1]
    return out


def deep_supervision_loss(criterion, out, target):
    """Average loss across deep-supervision outputs, weighting later outputs slightly more."""
    if not isinstance(out, (list, tuple)):
        return criterion(out, target)
    weights = torch.linspace(0.5, 1.0, steps=len(out), device=target.device)
    weights = weights / weights.sum()
    return sum(w * criterion(o, target) for w, o in zip(weights, out))


def UPP_train(model, criterion, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model = model.to(device)
    EPOCHS = epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
    )

    # unique checkpointing to save best model
    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M') 
    CHECKPOINT_DIR = Path(f'/content/drive/MyDrive/IRP/models/checkpoints/{RUN_ID}')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # initialise metrics
    vls, tls, ious, accs, precisions, recalls, f1s = [], [], [], [], [], [], []
    best_iou = -1.0
    best_state = copy.deepcopy(model.state_dict())
    obj_precisions, obj_recalls, obj_f1s = [], [], []

    for epoch in range(EPOCHS):

        # Train
        model.train()

        # epoch level loss
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            out = model(images)
            logits = main_logits(out)

            # BCE + DICE + TV loss
            loss = deep_supervision_loss(criterion, out, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()

        # init epoch level val metrics
        iou_tp = iou_fp = iou_fn = correct = total = intersection = union = tp = fp = fn = val_loss = 0.0


        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device).unsqueeze(1).float()

                out = model(images)
                logits = main_logits(out)
                val_loss += deep_supervision_loss(criterion, out, masks).item()

                # threshold at 0.75, from precision-recall curve
                preds = (torch.sigmoid(logits) >= 0.75).float()

                # object metrics
                pred_np = preds.cpu().numpy()
                mask_np = masks.cpu().numpy()

                # is centroid of prediction in ground truth label for that object?
                for i in range(pred_np.shape[0]):
                    iou_tp_i, iou_fp_i, iou_fn_i, _, _, _ = object_centroid_metrics(
                        pred_np[i, 0],
                        mask_np[i, 0]
                    )
                    iou_tp += iou_tp_i
                    iou_fp += iou_fp_i
                    iou_fn += iou_fn_i

                correct += (preds == masks).sum().item()
                total += masks.numel()
                intersection += (preds * masks).sum().item()
                union += (preds + masks - preds * masks).sum().item()
                tp += (preds * masks).sum().item()
                fp += (preds * (1 - masks)).sum().item()
                fn += ((1 - preds) * masks).sum().item()

        scheduler.step()

        # metrics

        iou_precision = (iou_tp + 1e-6) / (iou_tp + iou_fp + 1e-6)
        iou_recall = (iou_tp + 1e-6) / (iou_tp + iou_fn + 1e-6)
        iou_f1 = 2 * iou_precision * iou_recall / (iou_precision + iou_recall + 1e-6)

        acc = 100 * correct / total
        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        vls.append(avg_val)
        tls.append(avg_train)
        ious.append(iou)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        obj_precisions.append(iou_precision)
        obj_recalls.append(iou_recall)
        obj_f1s.append(iou_f1)

        # checkpointing
        if iou > best_iou:
            best_iou = iou
            best_state = copy.deepcopy(model.state_dict())
            for old in CHECKPOINT_DIR.glob('best_*.pt'):
                old.unlink()

            torch.save(model.state_dict(), CHECKPOINT_DIR / f'best_epoch{epoch+1:03d}_iou{iou:.4f}.pt')
            print(f'Best checkpoint saved (iou {iou:.4f})')

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"train {avg_train:.4f} | "
                f"val {avg_val:.4f} | "
                f"acc {acc:.2f}% | "
                f"iou {iou:.4f} | "
                f"P {precision:.4f} | "
                f"R {recall:.4f} | "
                f"F1 {f1:.4f} | "
                f"Obj_P {iou_precision:.4f} | "
                f"Obj_R {iou_recall:.4f} | "
                f"Obj_F1 {iou_f1:.4f} | ")
    
    model.load_state_dict(best_state)
    
    return vls, tls, obj_precisions, obj_recalls, obj_f1s, model, val_loader
