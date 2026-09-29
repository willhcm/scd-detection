# loads DEM tiles (and merges into chunks) from directory of small DEM scenes.
# feeds into model deployer and handles result internally.
# at the end, combines all into one final .shp file. 
# need to set up kwargs here for sure.

# imports
import os
import pathlib
import glob
from pathlib import Path
import sys
CODE_DIR = Path("../../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
from deployment.Deployer import Deployer

# main class for whole model. calls deployer internally (which itself calls ResPredictor (which itself uses MaskRCNN wrapper class) and PostProcessor classes)
#              SCD Model
#                   |
#                   v
#                Deployer
#                   |
#            -----------------
#            |               |
#            v               v
#        ResPredictor   PostProcessor
#            |                
#            v               
#       MaskRCNN Wrapper

class SCDModel():

    def __init__(self, 
                MaskRCNN_model_state, 
                rgb_veto:bool,
                dem_path:Path, 
                device:str,
                resolutions:list, 
                veto_model_state=None, 
                rgb_path=None, 
                hyperparams=None):
        
        self.MaskRCNN_model_state = MaskRCNN_model_state
        self.veto_model_state = veto_model_state
        self.rgb_veto = rgb_veto
        self.rgb_path = rgb_path
        self.device = device
        self.resolutions=resolutions

        self.dem_path = self._resolve_data_source(self, dem_path)
        self.deployer = self._create_deployer(hyperparams)

    # PUBLIC METHOD
    def predict(self,
                return_coverage=False,
                return_support=False,
                return_centroids=False):

        results = self.deployer.sweep()

        requested = {
            "predictions": True,
            "transform": True,
            "crs": True,
            "coverage": return_coverage,
            "support": return_support,
            "centroids": return_centroids}

        return tuple(results[key] for key, include in requested.items() if include)

    # Private
    def _create_deployer(self, hyperparams):

        deployer_kwargs = {
            "dem_path": self.dem_path,
            "model_state_dict": self.MaskRCNN_model_state,
            "veto_model_dict": self.veto_model_state,
            "device": self.device,
            "resolutions": self.resolutions,
            "rgb_path": self.rgb_path,
            "rgb_veto": self.rgb_veto,
            **hyperparams
        }

        return Deployer(**deployer_kwargs)

    def _resolve_data_source(self, dem_path):
        if os.path.isfile(dem_path):
            return dem_path
        elif os.path.isdir(dem_path):
            return self._build_dem(dem_path)
        else:
            raise TypeError('Please input a valid file or directory as a PosixPath')

    def _build_dem(self, dem_dir):
        paths = glob.glob(dem_dir)
        
        # merge DEMs in same way 
