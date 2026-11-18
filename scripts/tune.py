import argparse
from dataclasses import MISSING, fields
import json
from pathlib import Path
import sys
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "code"))
from deployment.config import DeployConfig
from modelling.Tuner import Tuner
import signal
import sys

# keys must be DeployConfig field names
# Used for discrete grid search.
PARAM_GRID = {
    'score_threshold': np.arange(0.4, 0.8, 0.025).tolist(),
    'veto_threshold': np.arange(0.5, 0.975, 0.025).tolist(),
    'size_sigma': np.arange(0.2, 0.8, 0.05).tolist(),
    'object_threshold': np.arange(0.3, 0.7, 0.05).tolist(),
    'keep_threshold': np.arange(0.3, 0.7, 0.05).tolist(),
}

# Continuous ranges used by Bayesian search. Bounds match the minimum and
# maximum values represented in PARAM_GRID.
PARAM_SPACE = {
    'score_threshold': {'type': 'float', 'low': 0.1, 'high': 0.9},
    'veto_threshold': {'type': 'float', 'low': 0.1, 'high': 0.975},
    'size_sigma': {'type': 'float', 'low': 0.1, 'high': 0.9},
    'object_threshold': {'type': 'float', 'low': 0.1, 'high': 0.9},
    'keep_threshold': {'type': 'float', 'low': 0.1, 'high': 0.9},
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
    if _CONFIG_FIELDS[name].default is not MISSING # MISSING being a sentinal value for fields which dont have defaults.
}

# only used by the veto stage, pointless to sweep with --no_veto
VETO_ONLY = {'veto_threshold'}

def main(args):

    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

    valid = {f.name for f in fields(DeployConfig)}
    unknown = (set(PARAM_SPACE) | set(DEPLOYER_PARAMS)) - valid

    params = {
        name: getattr(args, name)
        for name in DEPLOYER_PARAMS
    }

    if unknown:
        raise KeyError(f"not DeployConfig fields: {sorted(unknown)}")

    if not args.areas_config and not args.no_veto and not args.veto_model_state_path:
        raise ValueError("pass --veto_model_state_path, or use --no_veto.")

    out_path = RESULTS_DIR / args.run_name
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists, pick a different run name.")

    dem_path = args.dem_path.resolve()
    if not args.areas_config and not dem_path.exists():
        raise FileNotFoundError(f"DEM path not found: {dem_path}")

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

    if args.areas_config:
        config_path = (REPO_ROOT / args.areas_config).resolve()
        with config_path.open() as config_file:
            area_config = json.load(config_file)
        area_specs = (
            area_config.get("areas", area_config)
            if isinstance(area_config, dict)
            else area_config
        )
        if not isinstance(area_specs, list) or not area_specs:
            raise ValueError("areas_config must contain a non-empty 'areas' list.")

        areas = []
        for index, spec in enumerate(area_specs, start=1):
            required = {"dem_path", "model_state_path", "label_path"}
            missing = required - set(spec)
            if missing:
                raise ValueError(f"Area {index} is missing: {sorted(missing)}")

            def area_path(value, field_name):
                path = Path(value)
                if path.is_absolute():
                    candidates = [path]
                else:
                    candidates = [
                        config_path.parent / path,
                        REPO_ROOT / path,
                        Path.cwd() / path,
                    ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate.resolve()
                raise FileNotFoundError(
                    f"Area {index} {field_name} not found. Tried: "
                    + ", ".join(str(candidate.resolve()) for candidate in candidates)
                )

            area_device = spec.get("device", args.device)
            veto_path = spec.get("veto_model_state_path")
            if not args.no_veto and not veto_path:
                raise ValueError(
                    f"Area {index} needs veto_model_state_path, or use --no_veto."
                )
            area_hyperparams = dict(fixed)
            if "native_res" in spec:
                area_hyperparams["native_res"] = spec["native_res"]
            areas.append({
                "name": spec.get("name", f"area-{index}"),
                "model": torch.load(
                    area_path(spec["model_state_path"], "model_state_path"),
                    map_location=area_device,
                ),
                "dem_path": area_path(spec["dem_path"], "dem_path"),
                "device": area_device,
                "resolutions": spec.get("resolutions", args.resolutions),
                "out_path": out_path,
                "run_name": args.run_name,
                "veto_model_state": (
                    torch.load(
                        area_path(veto_path, "veto_model_state_path"),
                        map_location=area_device,
                    )
                    if veto_path else None
                ),
                "hyperparams": area_hyperparams,
                "to_veto": not args.no_veto,
                "label_path": area_path(spec["label_path"], "label_path"),
            })

        tuner = Tuner(
            model=None,
            dem_path=None,
            device=args.device,
            resolutions=args.resolutions,
            out_path=out_path,
            areas=areas,
        )
    else:
        model_state = torch.load(
            REPO_ROOT / args.model_state_path,
            map_location=args.device,
        )
        veto_state = None
        if args.veto_model_state_path:
            veto_state = torch.load(
                REPO_ROOT / args.veto_model_state_path,
                map_location=args.device,
            )
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
    parser.add_argument("--device", type=str, default=_CONFIG_FIELDS["device"].default, help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")
    parser.add_argument("--label_path", type=Path, help="Path to the label file for evaluation.")
    parser.add_argument("--areas_config", type=Path, help=("JSON file defining multiple test areas. Each area needs dem_path, model_state_path and label_path; optional fields are "
            "name, veto_model_state_path, device, resolutions and native_res."))
    
    parser.add_argument("--method", choices=("grid", "bayes"), default="bayes", help="Hyperparameter search method.",)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///tuning.db.",)

    hp = parser.add_argument_group("deployer hyperparameters")
    for name, default in DEPLOYER_PARAMS.items():
        hp.add_argument(f"--{name}", type=type(default), default=default)

    main(parser.parse_args())