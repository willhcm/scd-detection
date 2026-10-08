
from pathlib import Path
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import torch
from modelling.Trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent

def main(args):

    trainer = Trainer(args.root, device="cuda" if torch.cuda.is_available() else "cpu")

    out_path = REPO_ROOT / "results" / "training" / args.run
    
    trainer.train_fold(
        folded_out_region=args.folded_out_region,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run=args.run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mask R-CNN model for SCD detection.")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset.")
    parser.add_argument("--folded_out_region", type=str, required=True, help=" Region to fold out for training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--run", type=str, default=None, help="Name of the run to save the best model and training metrics.")
    args = parser.parse_args()

    main(args)
