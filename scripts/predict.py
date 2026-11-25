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
import signal
import sys
import cProfile
import pstats

# set by their own arguments (or read from the DEM), so no override flag
NOT_OVERRIDABLE = {"dem_path", "resolutions", "device", "nodata", "to_veto"}


def main(
    dem_path=REPO_ROOT / "tiles",
    model_state_path=REPO_ROOT / "models" / "maskrcnn.pt",
    run_name=None,
    veto_model_state_path=None,
    device="cuda",
    resolutions=None,
    no_veto=False,
    return_all=False,
    return_raw=False,
    return_provisional=False,
    return_unvetoed=False,
    **config_overrides):

    """Run SCD inference from parsed command-line arguments.

    Parameters
    ----------
    dem_path : pathlib.Path, optional
        Path to the DEM raster or directory of DEM tiles.
    model_state_path : pathlib.Path, optional
        Path to the Mask R-CNN checkpoint.
    run_name : str
        Name of the output directory under ``results``.
    veto_model_state_path : pathlib.Path, optional
        Path to the veto classifier checkpoint.
    device : str, optional
        Inference device, such as ``"cuda"`` or ``"cpu"``.
    resolutions : list of float, optional
        Resolutions at which to run inference.
    no_veto : bool, optional
        Disable veto classification.
    return_all, return_raw, return_provisional,
    return_unvetoed : bool, optional
        Select additional intermediate outputs.
    **config_overrides
        Overrides for fields in :class:`deployment.config.DeployConfig`.

    Returns
    -------
    dict
        In-memory prediction results selected by the ``--return_*`` flags.
        The final predictions and requested intermediate arrays are also
        saved under ``results/<run_name>``: predictions are written as the
        ESRI Shapefile ``predictions.shp``, array outputs as NumPy ``.npy``
        files, and rejected predictions as the GeoTIFF ``rejected.tif``.
        Run metadata is appended to ``covered_bounds.txt`` in the same
        directory.

    Raises
    ------
    FileExistsError
        If ``results/<run_name>`` already exists.
    FileNotFoundError
        If the requested DEM path does not exist.
    """
    if run_name is None:
        raise ValueError("run_name is required.")

    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))

    profiler = cProfile.Profile()
    profiler.enable()

    resolutions = [1.0] if resolutions is None else resolutions
    names = {f.name for f in fields(DeployConfig)} - NOT_OVERRIDABLE
    overrides = {k: v for k, v in config_overrides.items() if k in names}
    requested_flags = {
        "return_all": return_all,
        "return_raw": return_raw,
        "return_provisional": return_provisional,
        "return_unvetoed": return_unvetoed}

    requested_returns = {
        k.split("_")[1]: True
        for k, value in requested_flags.items()
        if value}

    requested_returns["predictions"] = True
    requested_returns['transform'] = True
    requested_returns['crs'] = True
    requested_returns = set(k for k, v in requested_returns.items() if v)

    out_path = RESULTS_DIR / run_name
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists, pick a different run name.")

    dem_path = Path(dem_path).resolve()
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM path not found: {dem_path}")

    model_state = torch.load(REPO_ROOT / model_state_path, map_location=device)

    veto_state = None
    if veto_model_state_path:
        veto_state = torch.load(REPO_ROOT / veto_model_state_path, map_location=device)

    out_path.mkdir(parents=True)

    scd_model = SCDModel(
        MaskRCNN_model_state=model_state,
        veto_model_state=veto_state,
        dem_path=dem_path,
        device=device,
        resolutions=resolutions,
        out_path=out_path,
        hyperparams=overrides,
        to_veto=not no_veto,
        requested_returns=requested_returns)

    print("effective config:")
    for k, v in asdict(config.get()).items():
        print(f"  {k} = {v}")

    results = scd_model.predict()

    profiler.disable()

    # Print profiling results
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(50)
    profiler.dump_stats(out_path / "deployment.prof")

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
    request_arggs.add_argument("--return_raw", action="store_true")
    request_arggs.add_argument("--return_provisional", action="store_true")
    request_arggs.add_argument("--return_provisional_objects", action="store_true")
    request_arggs.add_argument("--return_unvetoed", action="store_true")

    main(**vars(parser.parse_args()))