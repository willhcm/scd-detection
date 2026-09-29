# maskRCNN Trainer class to cleanly enable script-based training and LORO-CV for HPC cluster use.

from pathlib import Path
import sys
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))

from modelling.MaskRCNN import MaskRCNN, MaskRCNNDataset, collate_fn
from loaders.dataset import DatasetWrapper
from helpers.MaskRCNNFunctions import build_staged_optimizer, apply_training_stage

class Trainer():

    def __init__(self, root, device):
        self.root = root
        self.dset = DatasetWrapper(root)
        self.device = device

    def _get_loaders(self, folded_out_region):
        train_set, val_set = dset.generate_dataset(folded_out_region)

        train_loader = DataLoader(train_set,
                            shuffle=True,
                            num_workers=0,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

        val_loader = DataLoader(val_set,
                        shuffle=True,
                        num_workers=0,
                        batch_size=batch_size,
                        collate_fn=collate_fn)
    
    def train_fold(self, folded_out_region):
        train_loader, val_loader = _get_loaders(folded_out_region)
        model = MaskRCNN()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        best_f1 = -1.0
        best_state = None
        best_epoch = -1

        train_losses = []
        val_precisions = []
        val_recalls = []
        val_f1s = []
        val_ious = []
        current_stage = 1
        bad_epochs = 0
        optimizer = build_staged_optimizer(model, weight_decay=1e-4)

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

        results = {'train_losses': train_losses,
                   'val_precisions': val_precisions,
                   'val_recalls': val_recalls,
                   'val_f1s': val_f1s}

        return results

    def LORO_CV(self, regions):
        results = {}
        for region in regions: # where region is the region being folded out.
            train_loader, val_loader = _get_loaders(region)
            results[region] = train_fold(region)

        return results

