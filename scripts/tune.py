import argparse
import os
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from modelling import Tuner

PARAM_GRID = {
    'score_thresh': [0.5, 0.6, 0.7, 0.8],
    'mask_thresh': [0.5, 0.6, 0.7, 0.8],
    'veto_thresh': [0.5, 0.6, 0.7, 0.8],
    'detect_thresh': [0.5, 0.6, 0.7, 0.8]
}

DEPLOYER_PARAMS = {
    "tile_size": 512,
    "veto_context_tile_size": 224,
    "veto_batch_size": 16,
    "veto_threshold": 0.60, # tune on larger train set
    "score_threshold": 0.60, # to tune
    "mask_threshold": 0.55, # to tune
    "detect_threshold": 0.6, # to tune
    "context_scale": 5.0,
    "min_context_width_m": 750.0,
    "max_context_width_m": 4000.0,
    "native_res": 1,
    "stride_frac": 0.75,
}

def main(dem_path, model_state_path, veto_model_state_path, rgb_path, device, resolutions, rgb_veto, run_name, return_all='false', no_veto=True, hyperparams=None, label_path=None):
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

    params = {**PARAM_GRID, **(hyperparams or {})}
    print("effective hyperparams:")
    for k, v in params.items():
        print(f"  {k} = {v}")
    print(f"  no_veto = {no_veto}, rgb_veto = {rgb_veto}")

    tuner = Tuner(model=model_state,
                  dem_path=dem_path,
                  device=device,
                  resolutions=resolutions,
                  out_path=out_path,
                  run_name=run_name,
                  veto_model_state=veto_state,
                  hyperparams=params,
                  to_veto=not no_veto,
                  label_path=label_path)

    best_params, best_results = tuner.tune()

    print('Best hyperparameters found:')
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print('Achieved the following metrics:')
    for k, v in best_results.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SCD model on DEM data.")
    parser.add_argument("--dem_path", type=Path, default=REPO_ROOT / "tiles")
    parser.add_argument("--model_state_path", type=Path, default=REPO_ROOT / "models" / "maskrcnn.pt")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--veto_model_state_path", type=str, required=False, help="Path to the veto model state dictionary.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--return_all", type=str, required=False, default='false', help="Flag to indicate whether to return coverage, support, and centroids.")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")
    parser.add_argument("--label_path", type=str, required=True, help="Path to the label file for evaluation.")

    hp = parser.add_argument_group("deployer hyperparameters")
    for name, default in DEPLOYER_PARAMS.items():
        hp.add_argument(f"--{name}", type=type(default), default=default)

    args = parser.parse_args()
    hyperparams = {k: getattr(args, k) for k in DEPLOYER_PARAMS}

    main(args.dem_path, args.model_state_path, args.veto_model_state_path,
         args.label_path, args.device, args.resolutions,
         args.run_name, return_all=args.return_all, no_veto=args.no_veto,
         hyperparams=hyperparams, label_path=args.label_path)
    