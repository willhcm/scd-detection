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
    "veto_threshold": 0.60, # tune on larger train set
    "score_threshold": 0.60, # to tune
    "mask_threshold": 0.55, # to tune
    "context_scale": 5.0,
    "min_context_width_m": 750.0,
    "max_context_width_m": 4000.0,
    "native_res": 1,
    "stride_frac": 0.75,
}

def main(dem_path, model_state_path, veto_model_state_path, rgb_path, device, resolutions, rgb_veto, run_name, return_all='false', no_veto=True, hyperparams=None):
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

    params = {**DEPLOYER_PARAMS, **(hyperparams or {})}
    print("effective hyperparams:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    print(f"  no_veto = {no_veto}, rgb_veto = {rgb_veto}")

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
        hyperparams=params,
        run_name=run_name,
        to_veto=not no_veto
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
    parser.add_argument("--rgb_veto", type=str, default='false', help="Use RGB vetoing (true/false).")  
    parser.add_argument("--return_all", type=str, required=False, default='false', help="Flag to indicate whether to return coverage, support, and centroids.")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")

    hp = parser.add_argument_group("deployer hyperparameters")
    for name, default in DEPLOYER_PARAMS.items():
        hp.add_argument(f"--{name}", type=type(default), default=default)

    args = parser.parse_args()
    hyperparams = {k: getattr(args, k) for k in DEPLOYER_PARAMS}

    main(args.dem_path, args.model_state_path, args.veto_model_state_path,
         args.rgb_path, args.device, args.resolutions, args.rgb_veto,
         args.run_name, return_all=args.return_all, no_veto=args.no_veto,
         hyperparams=hyperparams)
    