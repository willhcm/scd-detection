import argparse
from pathlib import Path
import sys

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

def main(dem_path, model_state_path, veto_model_state_path, rgb_path, output_path, device, resolutions, rgb_veto):
    # Convert string to boolean
    rgb_veto = rgb_veto.lower() == 'true'

    # Create an instance of the SCDModel
    scd_model = SCDModel(
        MaskRCNN_model_state=model_state_path,
        veto_model_state=veto_model_state_path,
        rgb_veto=rgb_veto,
        dem_path=dem_path,
        device=device,
        resolutions=resolutions,
        out_path=output_path,
        rgb_path=rgb_path,
        hyperparams=DEPLOYER_PARAMS
    )

    # Run prediction
    predictions, transform, crs, coverage, support, centroids = scd_model.predict(
        return_coverage=True,
        return_support=True,
        return_centroids=True
    )

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy SCD model on DEM data.")
    parser.add_argument("--dem_path", type=str, required=True, help="Path to the DEM file.")
    parser.add_argument("--model_state_path", type=str, required=True, help="Path to the Mask R-CNN model state dictionary.")
    parser.add_argument("--veto_model_state_path", type=str, required=False, help="Path to the veto model state dictionary.")
    parser.add_argument("--rgb_path", type=str, required=False, help="Path to the RGB file for vetoing.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output predictions.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--resolutions", type=float, nargs='+', default=[1.0], help="List of resolutions to process.")
    parser.add_argument("--rgb_veto", type=str, required=False, help="Boolean flag to indicate whether to use RGB vetoing (True/False).")
    args = parser.parse_args()
    main(args.dem_path, args.model_state_path, args.veto_model_state_path, args.rgb_path, args.output_path, args.device, args.resolutions, args.rgb_veto)