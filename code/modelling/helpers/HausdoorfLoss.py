import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import HausdorffERLoss
from SegNet import Up, Down, ASPP, DoubleConv, object_centroid_metrics
from datetime import datetime
from pathlib import Path

# UNET adjusted for HD Loss (2 output channels, one for foreground, one for background.)
class HDSegNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = Down(f, f * 2)
        self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8)

        # Bottleneck 
        self.pool = nn.MaxPool2d(2)
        self.aspp = ASPP(f * 8, f * 16)
        self.bottleneck_drop = nn.Dropout2d(p=0.2)

        # Decoder
        self.dec4 = Up(f * 16 + f * 8, f * 8)
        self.dec3 = Up(f * 8 + f * 4, f * 4)
        self.dec2 = Up(f * 4 + f * 2, f * 2)
        self.dec1 = Up(f * 2 + f, f)

        self.out_conv = nn.Conv2d(f, 2, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # bottleneck
        b = self.pool(s4)
        b = self.bottleneck_drop(b) # dropout
        b = self.aspp(b) # several scales

        x = self.dec4(b,  s4)
        x = self.dec3(x,  s3)
        x = self.dec2(x,  s2)
        x = self.dec1(x,  s1)

        return self.out_conv(x)
    

# custom loss including haussdoorf loss
class HDLoss(nn.Module):
    def __init__(
        self,
        ce_weight=0.5,
        dice_weight=0.5,
        tv_weight=0.1,
        hd_weight=0.5,
        hd_alpha=2.0,
        hd_k=10,
        smooth=1e-6,
    ):
        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.tv_weight = tv_weight
        self.hd_weight = hd_weight
        self.smooth = smooth

        self.ce = nn.CrossEntropyLoss()
        self.hd = HausdorffERLoss(alpha=hd_alpha, k=hd_k)

    def dice_loss(self, fg_prob, fg_mask):
        # fg_prob vs fg_mask
        dims = (1, 2, 3)

        intersection = (fg_prob * fg_mask).sum(dim=dims)
        union = fg_prob.sum(dim=dims) + fg_mask.sum(dim=dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

    def tv_loss(self, fg_prob):
        # Smooths probability map spatially
        dy = torch.abs(fg_prob[:, :, 1:, :] - fg_prob[:, :, :-1, :]).mean()
        dx = torch.abs(fg_prob[:, :, :, 1:] - fg_prob[:, :, :, :-1]).mean()
        return dx + dy

    def forward(self, logits, masks):

        # ensures format
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks_long = masks.long() 
        masks_float = masks_long.unsqueeze(1).float()

        probs = torch.softmax(logits, dim=1)
        fg_prob = probs[:, 1:2]

        loss_ce = self.ce(logits, masks_long)
        loss_dice = self.dice_loss(fg_prob, masks_float)
        loss_tv = self.tv_loss(fg_prob)
        loss_hd = self.hd(logits, masks_long.unsqueeze(1))

        return (
            self.ce_weight * loss_ce
            + self.dice_weight * loss_dice
            + self.tv_weight * loss_tv
            + self.hd_weight * loss_hd
        )

def HD_train(model, criterion, train_loader, val_loader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )

    RUN_ID = datetime.now().strftime('%Y%m%d_%H%M')
    CHECKPOINT_DIR = Path(f'/content/drive/MyDrive/IRP/models/checkpoints/{RUN_ID}')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vls, tls, ious, accs, precisions, recalls, f1s = [], [], [], [], [], [], []
    obj_precisions, obj_recalls, obj_f1s = [], [], []
    best_iou = 0.0

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device).float()
            masks = masks.to(device)

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            masks = masks.long()

            optimizer.zero_grad()

            logits = model(images)          # [B, 2, H, W]
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()

        val_loss = 0.0
        correct = total = intersection = union = tp = fp = fn = 0
        obj_tp = obj_fp = obj_fn = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device).float()
                masks = masks.to(device)

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                masks_long = masks.long()              # [B, H, W]
                masks_float = masks_long.unsqueeze(1).float()

                logits = model(images)
                val_loss += criterion(logits, masks_long).item()

                probs = torch.softmax(logits, dim=1)
                fg_prob = probs[:, 1:2]                # [B, 1, H, W]

                preds = (fg_prob >= 0.75).float()      # [B, 1, H, W]

                pred_np = preds.cpu().numpy()
                mask_np = masks_float.cpu().numpy()

                for i in range(pred_np.shape[0]):
                    tp_i, fp_i, fn_i, _, _, _ = object_centroid_metrics(
                        pred_np[i, 0],
                        mask_np[i, 0]
                    )
                    obj_tp += tp_i
                    obj_fp += fp_i
                    obj_fn += fn_i

                correct += (preds == masks_float).sum().item()
                total += masks_float.numel()
                intersection += (preds * masks_float).sum().item()
                union += (preds + masks_float - preds * masks_float).sum().item()
                tp += (preds * masks_float).sum().item()
                fp += (preds * (1 - masks_float)).sum().item()
                fn += ((1 - preds) * masks_float).sum().item()

        scheduler.step()

        obj_precision = (obj_tp + 1e-6) / (obj_tp + obj_fp + 1e-6)
        obj_recall = (obj_tp + 1e-6) / (obj_tp + obj_fn + 1e-6)
        obj_f1 = 2 * obj_precision * obj_recall / (obj_precision + obj_recall + 1e-6)

        acc = 100 * correct / total
        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)
        f1  = 2 * precision * recall / (precision + recall + 1e-6)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        vls.append(avg_val)
        tls.append(avg_train)
        ious.append(iou)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        obj_precisions.append(obj_precision)
        obj_recalls.append(obj_recall)
        obj_f1s.append(obj_f1)

        if iou > best_iou:
            best_iou = iou
            print(f'Best checkpoint: (iou {iou:.4f})')

        print(f"Epoch {epoch+1:3d}/{epochs} | "
            f"train {avg_train:.4f} | "
            f"val {avg_val:.4f} | "
            f"acc {acc:.2f}% | "
            f"iou {iou:.4f} | "
            f"P {precision:.4f} | "
            f"R {recall:.4f} | "
            f"F1 {f1:.4f} | "
            f"Obj_P {obj_precision:.4f} | "
            f"Obj_R {obj_recall:.4f} | "
            f"Obj_F1 {obj_f1:.4f}")
        
    return vls, tls, precisions, recalls, f1s, model