# maskRCNN Trainer class to cleanly enable script-based training and LORO-CV for HPC cluster use.

from pathlib import Path
import sys
from xml.parsers.expat import model
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from modelling.MaskRCNN import MaskRCNN
from loaders.dataset import DatasetWrapper
from helpers.MaskRCNNFunctions import evaluate_maskrcnn_metrics, build_staged_optimizer, apply_training_stage, collate_fn, set_stage_progress, val_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import copy
from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parent.parent
from scipy.ndimage import label

from modelling.UNET import UNet, DiceLoss
    
class Trainer():

    def __init__(self, root, device):
        self.root = root
        self.dset = DatasetWrapper(root)
        self.device = device
        self.set_seed(42)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _get_MaskRCNN_loaders(self, folded_out_region, batch_size, random_train=False):

        if random_train:
            train_set, val_set = self.dset.generate_random_split(train_ratio=0.8, exclude_region=folded_out_region)
        else:
            train_set, val_set = self.dset.generate_dataset(folded_out_region)

        train_loader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=0,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

        val_loader = DataLoader(val_set,
                        shuffle=False,
                        num_workers=0,
                        batch_size=batch_size,
                        collate_fn=collate_fn)
        
        return train_loader, val_loader

    def train_fold(self, folded_out_region, epochs=50, batch_size=8, out_path=None, random_train=False, exclude_region=None):
        train_loader, val_loader = self._get_MaskRCNN_loaders(folded_out_region, batch_size, random_train=random_train)
        model = MaskRCNN()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        best_f1 = -1.0
        best_state = None
        best_epoch = -1

        train_losses = []
        val_losses = []
        val_precisions = []
        val_recalls = []
        val_f1s = []
        val_ious = []
        bbox_ious = []
        current_stage = 1
        stage_start = {1: 0, 2: 20, 3: 40, 4: 60}
        stage_end = {1: 20, 2: 40, 3: 60, 4: epochs}

        bad_epochs = 0
        optimizer = build_staged_optimizer(model, weight_decay=1e-4)

        apply_training_stage(model, optimizer, current_stage)

        for epoch in range(epochs):
                
            if epoch == 20:
                current_stage = 2
                apply_training_stage(model, optimizer, current_stage)

            elif epoch == 40:
                current_stage = 3
                apply_training_stage(model, optimizer, current_stage)

            elif epoch == 60:
                current_stage = 4
                apply_training_stage(model, optimizer, current_stage)


            s, e = stage_start[current_stage], stage_end[current_stage]

            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for i, (images, targets) in enumerate(pbar):
                frac = (epoch - s + i / len(train_loader)) / (e - s)
                set_stage_progress(optimizer, current_stage, frac)
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

            _val_loss, val_loss_parts = val_loss(model, val_loader, device)
            val_losses.append(_val_loss)

            precision, recall, val_f1, val_iou, p_precision, p_recall, p_f1, bbox_iou, old_f1 = evaluate_maskrcnn_metrics(
                model,
                val_loader,
                device,
                mask_thresh=0.50,
                score_thresh=0.50
            )

            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1s.append(val_f1)
            val_ious.append(val_iou)
            bbox_ious.append(bbox_iou)

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {_val_loss:.4f} | "
                f"Box IoU: {bbox_iou:.4f} | "
                f"IoU: {val_iou:.4f} | "
                f"Val P: {precision:.4f} | "
                f"Val R: {recall:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Pixel P: {p_precision:.4f} | "
                f"Pixel R: {p_recall:.4f} | "
                f"Pixel F1: {p_f1:.4f} | "
                f"Old F1: {old_f1:.4f} | ")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())

                print(f"Saved new best model at epoch {best_epoch} with F1={best_f1:.4f}")
                bad_epochs = 0
            else:
                bad_epochs += 1

        results = {'train_losses': train_losses,
                   'val_losses': val_losses,
                   'val_recalls': val_recalls,
                   'val_precisions': val_precisions,
                   'val_ious': val_ious,
                   'val_f1s': val_f1s,
                   'bbox_ious': bbox_ious}

        torch.save(best_state, out_path / f"best_model_fold_{folded_out_region}.pt")
        print(f"Best model saved at epoch {best_epoch} with F1={best_f1:.4f}")

        # save dict of results to npz file
        np.savez_compressed(out_path / f"training_metrics_fold_{folded_out_region}.npz", **results)

        return results

    def LORO_CV(self, regions, out_path, epochs=50, batch_size=8):
        results = {}
        for region in regions: # where region is the region being folded out.
            fold_out_path = out_path / f"fold_{region}"
            fold_out_path.mkdir(parents=True, exist_ok=True)
            print(f"\n holding out {region}")
            results[region] = self.train_fold(region, out_path=fold_out_path, epochs=epochs, batch_size=batch_size)

        np.savez_compressed(out_path / "LORO_CV_results.npz", **results)

    def random_split(self, out_path, epochs=50, batch_size=8, exclude_region=None):
        fold_out_path = out_path / "random_split"
        fold_out_path.mkdir(parents=True, exist_ok=True)

        results = self.train_fold(folded_out_region=exclude_region, epochs=epochs, batch_size=batch_size, out_path=fold_out_path, random_train=True)
        np.savez_compressed(out_path / "random_split_results.npz", **results)

    def parameter_sweep(self):
        params = {'score_thresh': [0.5, 0.6, 0.7, 0.8],
                  'mask_thresh': [0.5, 0.6, 0.7, 0.8],
                  'veto_thresh': [0.5, 0.6, 0.7, 0.8],
                  'detect_thresh': [0.5, 0.6, 0.7, 0.8]}

        ...

    def train_unet(self, epochs=50, batch_size=8, out_path=None, folded_out_region=None):
        train_set, test_set = self.dset.generate_unet_dataset()

        train_loader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=0,
                            batch_size=batch_size)

        val_loader = DataLoader(test_set,
                        shuffle=False,
                        num_workers=0,
                        batch_size=batch_size)

        model = UNet()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = DiceLoss()

        best_f1 = -1.0
        best_state = None
        best_epoch = -1

        train_losses = []
        val_losses = []
        val_ious = []

        bad_epochs = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        for epoch in range(epochs):

            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pbar.set_postfix({
                    "loss": float(loss.detach().cpu())
                })

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)

                    logits = model(images)
                    val_loss += criterion(logits, masks).item()

                    preds = torch.sigmoid(logits) > 0.5
                    preds = preds.cpu().numpy()
                    masks = masks.cpu().numpy()
                    tp = np.sum((preds == 1) & (masks == 1))
                    fp = np.sum((preds == 1) & (masks == 0))
                    fn = np.sum((preds == 0) & (masks == 1))

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step(loss)
                
            p_precision = tp / (tp + fp + 1e-8)
            p_recall = tp / (tp + fn + 1e-8)
            p_f1 = 2 * p_precision * p_recall / (p_precision + p_recall + 1e-8)
            val_iou = tp / (tp + fp + fn + 1e-8)
                    
            val_ious.append(val_iou)

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"IoU: {val_iou:.4f} | "
                f"Pixel P: {p_precision:.4f} | "
                f"Pixel R: {p_recall:.4f} | "
                f"Pixel F1: {p_f1:.4f} | ")

        results = {'train_losses': train_losses,
                'val_losses': val_losses,
                'val_ious': val_ious}

        torch.save(best_state, out_path / f"best_UNET_model_fold_{folded_out_region}.pt")
        print(f"Best model saved at epoch {best_epoch} with F1={best_f1:.4f}")

        # save dict of results to npz file
        np.savez_compressed(out_path / f"training_metrics_fold_{folded_out_region}.npz", **results)

        return results
        