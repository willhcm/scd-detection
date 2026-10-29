
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
    out_path.mkdir(parents=True, exist_ok=True)

    if args.model.upper() != "MASKRCNN":
        trainer.train_unet_fold(
            folded_out_region=args.folded_out_regions[0] if args.folded_out_regions else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_path=out_path,
            random_train=args.random_split)
        return

    if args.LORO_CV:
        regions = args.folded_out_regions # Replace with actual region names
        trainer.LORO_CV(regions, out_path, epochs = args.epochs, batch_size=args.batch_size)

    elif args.random_split:
        trainer.random_split(
            out_path=out_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            exclude_region=args.folded_out_regions[0] if args.folded_out_regions else None
        )

    else:
        trainer.train_fold(
            folded_out_region=args.folded_out_regions[0],
            epochs=args.epochs,
            batch_size=args.batch_size,
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
    parser.add_argument("--model", type=str, default="MaskRCNN", help="Model to train.")
    args = parser.parse_args()

    main(args)
