
from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import torch
from modelling.Trainer import VetoTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent

def main(root, folded_out_regions=None, epochs=50, batch_size=8, run=None):
    """Train the veto classifier using parsed command-line arguments.

    Parameters
    ----------
    root : str
        Root directory of the veto training dataset.
    folded_out_regions : list of str, optional
        Regions to hold out during training.
    epochs : int, optional
        Number of training epochs.
    batch_size : int, optional
        Number of samples per training batch.
    run : str, optional
        Name of the training run.

    Returns
    -------
    None
        Training artifacts are saved under ``results/training/<run>``:
        ``Vetoer.pt`` contains the best model checkpoint and normalization
        statistics, while ``veto_training_metrics_fold_<region>.npz`` and
        ``veto_embeddings.npz`` contain compressed NumPy training metrics,
        validation embeddings, and labels.
    """

    folded_out_regions = [] if folded_out_regions is None else folded_out_regions
    trainer = VetoTrainer(root, device="cuda" if torch.cuda.is_available() else "cpu")
    out_path = REPO_ROOT / "results" / "training" / run
    out_path.mkdir(parents=True, exist_ok=True)

    trainer.train(
        folded_out_region=folded_out_regions[0] if folded_out_regions else None,
        epochs=epochs,
        batch_size=batch_size,
        out_path=out_path,) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model for SCD detection.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--folded_out_regions", nargs="+", default=None, help="Regions to fold out for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--run", type=str, default=None, help="Name of the run to save the best model and training metrics.")
    args = parser.parse_args()

    main(**vars(args))
