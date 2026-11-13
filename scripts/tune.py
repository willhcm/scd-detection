import argparse
from dataclasses import MISSING, fields
from pathlib import Path
import sys
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "code"))
from deployment.config import DeployConfig
from modelling.Tuner import Tuner

# keys must be DeployConfig field names
PARAM_GRID = {
    'score_threshold': np.arange(0.4, 0.8, 0.025).tolist(),
    'mask_threshold': np.arange(0.3, 0.6, 0.025).tolist(),
    'veto_threshold': np.arange(0.5, 0.975, 0.025).tolist(),
    'size_sigma': np.arange(0.2, 0.8, 0.05).tolist(),
    'object_threshold': np.arange(0.3, 0.7, 0.05).tolist(),
    'keep_threshold': np.arange(0.3, 0.7, 0.05).tolist(),
}

# Continuous ranges used by Bayesian search. Bounds match the minimum and
# maximum values represented in PARAM_GRID.
PARAM_SPACE = {
    'score_threshold': {'type': 'float', 'low': 0.4, 'high': 0.775},
    'mask_threshold': {'type': 'float', 'low': 0.3, 'high': 0.575},
    'veto_threshold': {'type': 'float', 'low': 0.5, 'high': 0.95},
    'size_sigma': {'type': 'float', 'low': 0.2, 'high': 0.75},
    'object_threshold': {'type': 'float', 'low': 0.3, 'high': 0.65},
    'keep_threshold': {'type': 'float', 'low': 0.3, 'high': 0.65},
}

# Fixed deployment parameters passed to each tuning run. Their values are
# taken from DeployConfig so this script cannot silently drift from runtime
# defaults.
_CONFIG_FIELDS = {field.name: field for field in fields(DeployConfig)}
_FIXED_PARAM_NAMES = (
    "tile_size",
    "veto_context_tile_size",
    "veto_batch_size",
    "context_scale",
    "min_context_width_m",
    "max_context_width_m",
    "native_res",
    "stride_frac",
)
DEPLOYER_PARAMS = {
    name: _CONFIG_FIELDS[name].default
    for name in _FIXED_PARAM_NAMES
    if _CONFIG_FIELDS[name].default is not MISSING
}

# only used by the veto stage, pointless to sweep with --no_veto
VETO_ONLY = {'veto_threshold'}

def main(args):

    valid = {f.name for f in fields(DeployConfig)}
    unknown = (set(PARAM_SPACE) | set(DEPLOYER_PARAMS)) - valid

    params = DEPLOYER_PARAMS.copy()
    overrides = {f.name: getattr(args, f.name) for f in fields(DeployConfig) if hasattr(args, f.name)}
    params.update(overrides)

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
    search_space = {
        k: v for k, v in PARAM_SPACE.items()
        if not (args.no_veto and k in VETO_ONLY)
    }

    fixed = {k: getattr(args, k) for k in params if k not in grid}

    print("fixed hyperparams:")
    for k, v in fixed.items():
        print(f"  {k} = {v}")
    print("search space:")
    values_to_print = grid if args.method == "grid" else search_space
    for k, v in values_to_print.items():
        print(f"  {k} = {v}")
    print(f"  no_veto = {args.no_veto}")
    print(f"  method = {args.method}")

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

    if args.method == "grid":
        best_params, best_results = tuner.tune(grid)
    else:
        best_params, best_results = tuner.tune_bayes(
            search_space,
            n_trials=args.n_trials,
            seed=args.seed,
            study_name=args.study_name,
            storage=args.storage,
        )

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
    parser.add_argument(
        "--device",
        type=str,
        default=_CONFIG_FIELDS["device"].default,
        help="Device to run the model on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")
    parser.add_argument("--label_path", type=Path, required=True, help="Path to the label file for evaluation.")
    parser.add_argument(
        "--method",
        choices=("grid", "bayes"),
        default="bayes",
        help="Hyperparameter search method.",
    )
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL, e.g. sqlite:///tuning.db.",
    )

    hp = parser.add_argument_group("deployer hyperparameters")
    for name, default in DEPLOYER_PARAMS.items():
        hp.add_argument(f"--{name}", type=type(default), default=default)

    main(parser.parse_args())