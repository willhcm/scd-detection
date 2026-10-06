import argparse
import os
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from deployment.Inference import SCDModel

DEPLOYER_PARAMS = {
    "tile_size": 512,
    "veto_context_tile_size": 224,
    "veto_batch_size": 16,
    "veto_threshold": 0.74, # tune on larger train set
    "score_threshold": 0.60, # to tune
    "mask_threshold": 0.55, # to tune
    "context_scale": 5.0,
    "min_context_width_m": 750.0,
    "max_context_width_m": 4000.0,
    "native_res": 1,
    "stride_frac": 0.75,
}

def main(dem_path, model_state_path, veto_model_state_path, rgb_path, device, resolutions, rgb_veto, run_name, return_all='false'):
    # Convert string to boolean
    rgb_veto = rgb_veto.lower() == 'true'
    return_all = return_all.lower() == 'true'

    out_path = RESULTS_DIR / run_name
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists, pick a different run name.")
    out_path.mkdir(parents=True)

    dem_path = Path(dem_path).resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM path not found: {dem_path}")
    
    model_path = REPO_ROOT / model_state_path
    model_state = torch.load(model_path, map_location=device)

    veto_model_path = REPO_ROOT / veto_model_state_path if veto_model_state_path else None
    veto_state = torch.load(veto_model_path, map_location=device) if veto_model_state_path else None
    rgb_path = Path(rgb_path) if rgb_path else None

    # Create an instance of the SCDModel
    scd_model = SCDModel(
        MaskRCNN_model_state=model_state,
        veto_model_state=veto_state,
        rgb_veto=rgb_veto,
        dem_path=dem_path,
        device=device,
        resolutions=resolutions,
        out_path=out_path,
        rgb_path=rgb_path,
        hyperparams=DEPLOYER_PARAMS,
        run_name=run_name
    )

    # Run prediction
    predictions, transform, crs, coverage, support = scd_model.predict(
        return_coverage=True,
        return_support=True,
        return_centroids=True
    )
    if return_all:
        return predictions, transform, crs, coverage, support
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SCD model on DEM data.")
    parser.add_argument("--dem_path", type=Path, default=REPO_ROOT / "tiles")
    parser.add_argument("--model_state_path", type=Path, default=REPO_ROOT / "models" / "maskrcnn.pt")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--veto_model_state_path", type=str, required=False, help="Path to the veto model state dictionary.")
    parser.add_argument("--rgb_path", type=str, required=False, help="Path to the RGB file for vetoing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--rgb_veto", type=str, required=False, help="Boolean flag to indicate whether to use RGB vetoing (True/False).")
    parser.add_argument("--return_all", type=str, required=False, default='false', help="Flag to indicate whether to return coverage, support, and centroids.")
    args = parser.parse_args()
    main(args.dem_path, args.model_state_path, args.veto_model_state_path, args.rgb_path, args.device, args.resolutions, args.rgb_veto, args.run_name, return_all=args.return_all)