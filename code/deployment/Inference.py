# loads DEM tiles (and merges into chunks) from directory of small DEM scenes.
# feeds into model deployer and handles result internally.
# at the end, combines all into one final .shp file. 

# imports
import os
from pathlib import Path
import sys
CODE_DIR = Path("../../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
import config
from deployment.Deployer import Deployer
import numpy as np
import rasterio as rio
import subprocess

# main class for whole model. calls deployer internally (which itself calls ResPredictor (which itself uses MaskRCNN wrapper class) and PostProcessor classes)
#
#              SCD Model
#                   |
#                   v
#                Deployer
#                   |
#            -----------------
#            |               |
#            v               v
#        ResPredictor   PostProcessor
#            |               |
#            v               v
#     MaskRCNN Wrapper   Veto Model
#           | 
#           v 
#      MaskRCNN Model

class SCDModel():

    def __init__(self, 
                MaskRCNN_model_state, 
                dem_path:Path, 
                device:str,
                resolutions:list, 
                out_path:Path,
                veto_model_state=None, 
                hyperparams=None,
                run_name=None,
                to_veto=True):

        if to_veto and veto_model_state is None:
            raise ValueError("to_veto is True but no veto_model_state was given.")
        
        self.MaskRCNN_model_state = MaskRCNN_model_state
        self.veto_model_state = veto_model_state
        self.device = device
        self.resolutions = resolutions
        self.out_path = out_path
        self.run_name = run_name
        self.to_veto = to_veto

        # config needs the resolved DEM path and its nodata value, so it is
        # initialised after the DEM has been found/merged.
        self.dem_path = self._resolve_data_source(dem_path)
        self.deployer = self._create_deployer(hyperparams)

    # PUBLIC METHODS
    def set_params(self, **params):
        # Deployer snapshots the config on init, so it has to be rebuilt.
        config.update(**params)
        self.deployer = Deployer(self.MaskRCNN_model_state, self.veto_model_state)

    def predict(self,
                return_coverage=False,
                return_support=False,
                return_centroids=False):

        print('Running SCD model prediction...')

        results = self.deployer.sweep()

        out_path = self.out_path / "predictions.shp"

        requested = {
            "predictions": True,
            "transform": True,
            "crs": True,
            "coverage": return_coverage,
            "support": return_support}

        self.deployer.merge_predictions(results['predictions'], results['transform'], results['crs'], out_path)

        # write bounds of DEM to a text file for reference
        with open(self.out_path / "covered_bounds.txt", "a") as f:
            f.write(f"Run: {self.run_name}\n")
            f.write(f"Bounds of DEM: {self.DEM_bounds}\n")
            f.write(f"CRS of DEM: {self.DEM_crs}\n")

        return tuple(results[key] for key, include in requested.items() if include)

    # Private
    def _create_deployer(self, hyperparams):

        config.init(
            dem_path=self.dem_path,
            resolutions=tuple(self.resolutions),
            device=self.device,
            nodata=self.NODATA if self.NODATA is not None else -9999,
            to_veto=self.to_veto,
            **(hyperparams or {}),
        )

        return Deployer(self.MaskRCNN_model_state, self.veto_model_state)

    def _resolve_data_source(self, dem_path):
        if os.path.isfile(dem_path):
            with rio.open(dem_path) as src:
                self.NODATA = src.nodata
                self.DEM_bounds = src.bounds
                self.DEM_crs = src.crs
            return dem_path
        elif os.path.isdir(dem_path):
            return self._build_dem(dem_path)
        else:
            raise TypeError('Please input a valid file or directory as a PosixPath')

    def _build_dem(self, dem_dir):
        dem_dir = Path(dem_dir)

        paths = list(dem_dir.glob("*.tif"))
        if not paths:
            raise FileNotFoundError(f"No .tif files found in {dem_dir}.")
        if 'merged.tif' in [p.name for p in paths]:
            raise FileExistsError(f"merged.tif already exists in {dem_dir}. Please remove it or choose a different directory.")

        print(f"Building DEM from directory: {dem_dir}")

        with rio.open(paths[0]) as src:
            src_nodata = src.nodata

        cmd = ["gdalwarp"]
        if src_nodata is not None:
            cmd += ["-srcnodata", str(src_nodata)]
        cmd += [
            "-dstnodata", "-9999",
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES",
            "-of", "GTiff",
            "-co", "BIGTIFF=YES",
            *[str(p) for p in paths],
            str(dem_dir / "merged.tif")
        ]
        subprocess.run(cmd, check=True)

        # merged raster is written with -9999 as nodata
        self.NODATA = -9999

        with rio.open(dem_dir / "merged.tif") as src:
            self.DEM_bounds = src.bounds
            self.DEM_crs = src.crs
        output_path = dem_dir / "merged.tif"


        return output_path