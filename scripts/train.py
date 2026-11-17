
from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import torch
from modelling.Trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent

def main(
    root,
    folded_out_regions=None,
    epochs=50,
    batch_size=8,
    run=None,
    LORO_CV=False,
    random_split=False,
    device="cuda" if torch.cuda.is_available() else "cpu", 
    test_size=0.2,):

    """Train an SCD detection MaskRCNN using the selected training strategy.

    Parameters
    ----------
    root : str
        Root directory of the training dataset.
    folded_out_regions : list of str, optional
        Regions to hold out during training.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Number of samples per training batch.
    run : str, optional
        Name of the training run.
    LORO_CV : bool, optional
        Run leave-one-region-out cross-validation.
    random_split : bool, optional
        Use a random train/validation split.
    device : str, optional
        Device to run the model on (e.g., 'cuda' or 'cpu').
    model : str, optional
        Model type to train, such as ``"MaskRCNN"`` or ``"UNET"``.

    Returns
    -------
    None
        Training artifacts are saved under
        ``results/training/<run>``. Mask R-CNN training writes the checkpoint
        ``MaskRCNN.pt`` and compressed NumPy metric or resolution archives.
        Leave-one-region-out training additionally writes
        ``LORO_CV_results.npz`` and per-region subdirectories; random-split
        training writes its artifacts under ``random_split``. The UNet
        training branch writes its artifacts using the selected trainer.
    """

    folded_out_regions = [] if folded_out_regions is None else folded_out_regions
    trainer = Trainer(root, device=device)
    out_path = REPO_ROOT / "results" / "training" / run
    out_path.mkdir(parents=True, exist_ok=True)

    if LORO_CV:
        regions = folded_out_regions
        trainer.LORO_CV(regions, out_path, epochs=epochs, batch_size=batch_size)

    elif random_split:
        trainer.random_split(
            out_path=out_path,
            epochs=epochs,
            batch_size=batch_size,
            exclude_region=folded_out_regions[0] if folded_out_regions else None,
            test_size=test_size,)

    else:
        trainer.train_fold(
            folded_out_region=folded_out_regions[0],
            epochs=epochs,
            batch_size=batch_size,
            out_path=out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model for SCD detection.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--folded_out_regions", nargs="+", default=[])
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--run", type=str, default=None, help="Name of the run to save the best model and training metrics.")
    parser.add_argument("--LORO_CV", action="store_true", help="Perform Leave-One-Region-Out Cross-Validation.")
    parser.add_argument("--random_split", action="store_true", help="Perform random split training instead of LORO-CV.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the validation split when using random split training.")
    args = parser.parse_args()

    main(**vars(args))
