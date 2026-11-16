# loads DEM tiles (and merges into chunks) from directory of small DEM scenes.
# feeds into model deployer and handles result internally.
# at the end, combines all into one final .shp file. 

# imports
import os
from pathlib import Path
import sys
CODE_DIR = Path("../../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
import deployment.config as config
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
    """Run the complete SCD inference workflow.

    ``SCDModel`` is the main user-facing inference class. It prepares the
    DEM input, configures a :class:`Deployer`, runs predictions at the
    requested resolutions, applies the optional veto model, and saves the
    resulting outputs.

    Parameters
    ----------
    MaskRCNN_model_state : dict
        Checkpoint for the Mask R-CNN model. It must contain the keys
        ``"model_state_dict"`` and ``"norm_stats"``.
    dem_path : pathlib.Path
        Path to a DEM raster or to a directory of ``.tif`` DEM tiles. A
        directory is mosaicked into ``merged.tif`` before inference.
    device : str
        Device used for model inference, such as ``"cpu"`` or ``"cuda"``.
    resolutions : list
        DEM resolutions at which to run inference.
    out_path : pathlib.Path
        Directory where prediction outputs and ``covered_bounds.txt`` are
        saved.
    veto_model_state : dict, optional
        Checkpoint for the veto classifier. Required when ``to_veto`` is
        ``True``.
    hyperparams : dict, optional
        Overrides for fields in :class:`deployment.config.DeployConfig`.
    run_name : str, optional
        Name recorded in ``covered_bounds.txt`` for this inference run.
    to_veto : bool, default=True
        Whether to apply the veto classifier to the merged predictions.
    requested_returns : set, optional
        Names of the intermediate and final results to return from the
        deployment pipeline, such as ``"predictions"``, ``"transform"``,
        ``"crs"``, or ``"raw"``.

    Raises
    ------
    ValueError
        If vetoing is enabled without a veto model checkpoint.
    TypeError
        If ``dem_path`` is neither a file nor a directory.
    FileNotFoundError
        If a DEM directory contains no ``.tif`` files.
    FileExistsError
        If a DEM directory already contains ``merged.tif``.
    """

    def __init__(self, 
                MaskRCNN_model_state, 
                dem_path:Path, 
                device:str,
                resolutions:list, 
                out_path:Path,
                veto_model_state=None, 
                hyperparams=None,
                run_name=None,
                to_veto=True, 
                requested_returns=None):

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
        self.deployer = self._create_deployer(hyperparams, requested_returns)

    # PUBLIC METHODS
    def set_params(self, **params):
        """Update deployment parameters and rebuild the deployer.

        Parameters
        ----------
        **params
            Values for fields in :class:`deployment.config.DeployConfig`.
            The existing configuration is replaced with these overrides,
            and the deployer is rebuilt so it uses the updated snapshot.
        """
        # Deployer snapshots the config on init, so it has to be rebuilt.
        config.update(**params)
        self.deployer = Deployer(self.MaskRCNN_model_state, self.veto_model_state)

    def predict(self):
        """Run inference, save predictions, and record DEM coverage metadata.

        Returns
        -------
        dict
            Results selected by ``requested_returns``. The available results
            include merged ``"predictions"``, ``"transform"``, ``"crs"``,
            ``"raw"``, ``"unvetoed"``, ``"obj_score"``, ``"provisional"``,
            and veto outputs when vetoing is enabled. The returned values are
            in-memory arrays or geospatial metadata; they are also written to
            ``self.out_path`` when their names are requested:

            * ``predictions`` is saved as ``predictions.shp`` in ESRI
              Shapefile format.
            * ``raw``, ``provisional``, ``obj_score``, and ``unvetoed`` are
              saved as NumPy ``.npy`` arrays.
            * ``rejected`` is saved as a GeoTIFF ``rejected.tif``.
            * Run metadata is appended to ``covered_bounds.txt`` as text,
              including the run name, DEM bounds, CRS, and source tile names.

            Files are saved under ``self.out_path``. Only outputs included in
            ``requested_returns`` are written, except
            ``covered_bounds.txt``, which is always updated.
        """

        print('Running SCD model prediction...')

        results = self.deployer.sweep()

        print(f'Saving results to {self.out_path}...')
        self.deployer.save_predictions(results, self.out_path)

        # write bounds of DEM to a text file for reference
        with open(self.out_path / "covered_bounds.txt", "a") as f:
            f.write(f"Run: {self.run_name}\n")
            f.write(f"Bounds of DEM: {self.DEM_bounds}\n")
            f.write(f"CRS of DEM: {self.DEM_crs}\n")
            f.write(f"Tiles used: {', '.join(self.tiles_names)}\n")

        return results

    # Private
    def _create_deployer(self, hyperparams, requested_returns):
        """Initialize deployment configuration and construct a deployer."""

        config.init(
            dem_path=self.dem_path,
            resolutions=tuple(self.resolutions),
            device=self.device,
            nodata=self.NODATA if self.NODATA is not None else -9999,
            to_veto=self.to_veto,
            **(hyperparams or {}),
            requested_returns=requested_returns)

        return Deployer(self.MaskRCNN_model_state, self.veto_model_state)

    def _resolve_data_source(self, dem_path):
        """Resolve a DEM file or build a raster mosaic from DEM tiles."""

        if os.path.isfile(dem_path):
            with rio.open(dem_path) as src:
                self.NODATA = src.nodata
                self.DEM_bounds = src.bounds
                self.DEM_crs = src.crs
                self.tiles_names = [Path(dem_path).name]
            return dem_path
        elif os.path.isdir(dem_path):
            return self._build_dem(dem_path)
        else:
            raise TypeError('Please input a valid file or directory as a PosixPath')

    def _build_dem(self, dem_dir):
        """Mosaic all DEM GeoTIFFs in a directory into ``merged.tif``.

        Parameters
        ----------
        dem_dir : pathlib.Path
            Directory containing the source DEM tiles.

        Returns
        -------
        pathlib.Path
            Path to the generated ``merged.tif`` raster.
        """

        dem_dir = Path(dem_dir)

        paths = list(dem_dir.glob("*.tif"))
        if not paths:
            raise FileNotFoundError(f"No .tif files found in {dem_dir}.")
        if 'merged.tif' in [p.name for p in paths]:
            raise FileExistsError(f"merged.tif already exists in {dem_dir}. Please remove it or choose a different directory.")

        print(f"Building DEM from directory: {dem_dir}")
        self.tiles_names = [p.name for p in paths]

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