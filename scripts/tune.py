import argparse
from dataclasses import fields
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "code"))
from config import DeployConfig
from modelling.Tuner import Tuner

# keys must be DeployConfig field names
PARAM_GRID = {
    'score_threshold': [0.5, 0.6, 0.7, 0.8],
    'mask_threshold': [0.5, 0.6, 0.7, 0.8],
    'veto_threshold': [0.5, 0.6, 0.7, 0.8],
    'detect_threshold': [0.5, 0.6, 0.7, 0.8]
}

# only used by the veto stage, pointless to sweep with --no_veto
VETO_ONLY = {'veto_threshold', 'detect_threshold'}

# values for anything not being swept (also CLI-overridable)
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
    "native_res": 1.0,
    "stride_frac": 0.75,
}


def main(args):

    valid = {f.name for f in fields(DeployConfig)}
    unknown = (set(PARAM_GRID) | set(DEPLOYER_PARAMS)) - valid
    if unknown:
        raise KeyError(f"not DeployConfig fields: {sorted(unknown)}")

    if not args.no_veto and not args.veto_model_state_path:
        raise ValueError("pass --veto_model_state_path, or use --no_veto.")

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

    grid = {k: v for k, v in PARAM_GRID.items() if not (args.no_veto and k in VETO_ONLY)}
    fixed = {k: getattr(args, k) for k in DEPLOYER_PARAMS if k not in grid}

    print("fixed hyperparams:")
    for k, v in fixed.items():
        print(f"  {k} = {v}")
    print("grid:")
    for k, v in grid.items():
        print(f"  {k} = {v}")
    print(f"  no_veto = {args.no_veto}")

    tuner = Tuner(model=model_state,
                  dem_path=dem_path,
                  device=args.device,
                  resolutions=args.resolutions,
                  out_path=out_path,
                  run_name=args.run_name,
                  veto_model_state=veto_state,
                  hyperparams=fixed,
                  to_veto=not args.no_veto,
                  label_path=args.label_path)

    best_params, best_results = tuner.tune(grid)

    print('Best hyperparameters found:')
    for k, v in best_params.items():
        print(f"  {k} = {v}")
    print('Achieved the following metrics:')
    for k, v in best_results.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune SCD model thresholds on DEM data.")
    parser.add_argument("--dem_path", type=Path, default=REPO_ROOT / "tiles")
    parser.add_argument("--model_state_path", type=Path, default=REPO_ROOT / "models" / "maskrcnn.pt")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--veto_model_state_path", type=Path, required=False, help="Path to the veto model state dictionary.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")
    parser.add_argument("--label_path", type=Path, required=True, help="Path to the label file for evaluation.")

    hp = parser.add_argument_group("deployer hyperparameters")
    for name, default in DEPLOYER_PARAMS.items():
        hp.add_argument(f"--{name}", type=type(default), default=default)

    main(parser.parse_args())