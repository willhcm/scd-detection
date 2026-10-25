import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "code"))
import config
from deployment.Inference import SCDModel

# overrides for any config.DeployConfig field, e.g. {"veto_threshold": 0.7}
HYPERPARAMS = {}

def main(args):

    out_path = RESULTS_DIR / args.run_name
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists, pick a different run name.")

    dem_path = args.dem_path.resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM path not found: {dem_path}")

    model_state = torch.load(REPO_ROOT / args.model_state_path, map_location=args.device)

    veto_state = None
    if args.veto_model_state_path:
        veto_state = torch.load(REPO_ROOT / args.veto_model_state_path, map_location=args.device)

    out_path.mkdir(parents=True)

    scd_model = SCDModel(
        MaskRCNN_model_state=model_state,
        veto_model_state=veto_state,
        dem_path=dem_path,
        device=args.device,
        resolutions=args.resolutions,
        out_path=out_path,
        hyperparams=HYPERPARAMS,
        run_name=args.run_name,
        to_veto=not args.no_veto)

    print("effective config:")
    for k, v in asdict(config.get()).items():
        print(f"  {k} = {v}")

    predictions, transform, crs, coverage, support = scd_model.predict(
        return_coverage=True,
        return_support=True)

    if args.return_all:
        return predictions, transform, crs, coverage, support


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SCD model on DEM data.")
    parser.add_argument("--dem_path", type=Path, default=REPO_ROOT / "tiles")
    parser.add_argument("--model_state_path", type=Path, default=REPO_ROOT / "models" / "maskrcnn.pt")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--veto_model_state_path", type=Path, required=False, help="Path to the veto model state dictionary.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--return_all", action="store_true", help="Return coverage and support as well (only useful when calling main() from a notebook).")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")

    main(parser.parse_args())