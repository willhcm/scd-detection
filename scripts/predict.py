import argparse
from dataclasses import asdict, fields
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "code"))
from deployment.config import DeployConfig
import deployment.config as config
from deployment.Inference import SCDModel

# set by their own arguments (or read from the DEM), so no override flag
NOT_OVERRIDABLE = {"dem_path", "resolutions", "device", "nodata", "to_veto"}


def main(args):

    names = {f.name for f in fields(DeployConfig)} - NOT_OVERRIDABLE
    overrides = {k: v for k, v in vars(args).items() if k in names}
    requested_returns = {k.split("_")[1]: True for k in vars(args) if k.startswith("return_") and getattr(args, k)}

    requested_returns["predictions"] = True
    requested_returns['transform'] = True
    requested_returns['crs'] = True
    requested_returns = set(k for k, v in requested_returns.items() if v)

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
        hyperparams=overrides,
        run_name=args.run_name,
        to_veto=not args.no_veto,
        requested_returns=requested_returns)

    print("effective config:")
    for k, v in asdict(config.get()).items():
        print(f"  {k} = {v}")

    results = scd_model.predict()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SCD model on DEM data.")
    parser.add_argument("--dem_path", type=Path, default=REPO_ROOT / "tiles")
    parser.add_argument("--model_state_path", type=Path, default=REPO_ROOT / "models" / "maskrcnn.pt")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")
    parser.add_argument("--veto_model_state_path", type=Path, required=False, help="Path to the veto model state dictionary.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--no_veto", action="store_true", help="dont veto")

    # one flag per DeployConfig field; only flags actually passed end up in args
    cfg_args = parser.add_argument_group("config overrides")
    for f in fields(DeployConfig):
        if f.name in NOT_OVERRIDABLE:
            continue
        flag = f"--{f.name}"
        help_text = f"default: {f.default}"
        if f.type is bool:
            cfg_args.add_argument(flag, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help=help_text)
        else:
            cfg_args.add_argument(flag, type=f.type, default=argparse.SUPPRESS, help=help_text)

    request_arggs = parser.add_argument_group("requested outputs")
    request_arggs.add_argument("--return_all", action="store_true")
    request_arggs.add_argument("--return_rejected", action="store_true")
    request_arggs.add_argument("--return_veto_probability_map", action="store_true")
    request_arggs.add_argument("--return_raw", action="store_true")
    request_arggs.add_argument("--return_provisional", action="store_true")
    request_arggs.add_argument("--return_provisional_objects", action="store_true")

    main(parser.parse_args())