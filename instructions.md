# Setting up the model and HPC!

All commands are to be run on the command line.

## Cloning the GitHub repository.

Navigate to your desired destination directory, e.g.:

```cd documents/MSci ```

```git clone tbw```

### Python Environment

First, create the python environment. NOTE: THIS ONLY NEEDS TO BE DONE ONCE PER MACHINE.

Navigate into the scd-detection repository directory:

```cd documents/scd-detection``` (equivalent command to navigate through your files)

Check conda is installed:

```conda --version```

If not, try and install Miniconda through this [link](https://www.anaconda.com/docs/getting-started/installation). Let me know if it doesn't work, it can be annoying.


Once done, navigate to the scd-detection directory:

```cd documents/scd-detection``` (equivalent command to navigate through your files)

and run:

```conda env create -n scd_env -f enviroment.yml```

After this has been done the first time, simply activate the environment each time you log-on:

```conda activate scd_env```

## Model Download. (Only needs to be done once)

Download the two models (MaskRCNN, DEMVeto) to your local machine from this [link](https://drive.google.com/drive/folders/1SP--9SlvA7zn0S0yoilJTJe9A-x-umr_?usp=sharing):

Copy them to the HPC virtual machine using:

```scp -r downloads/models USER@borg-login.ese.ic.ac.uk:/scratch_root/USER/scd-detection```

# Running the Model!

## Local Machine

Run urls.py to generate a new .txt files of valid download URLs. Need to input a manually collated .txt list of tile names, or .gpkg file from ArcGIS/QGIS which outlines the region wanted:

```python3 scripts/urls.py --request NEW_LIST_OF_DEFRA_TILE_NAMES.txt --request_name 'CUSTOMISE_THIS_REQUEST_NAME'```

or [s-]

```python3 scripts/urls.py --request WANTED_REGION.gpkg --request_name 'CUSTOMISE_THIS_REQUEST_NAME'```

Second, download the files using curl. The files will be stored in a directory called tiles/

```xargs -n 1 curl -L -J -O --output-dir ./tiles < ./urls/test_urls.txt```

Third, unzip all the files (inplace), keeping only the .tif files (the actual data):

```cd tiles```

```for f in *.zip; do unzip -j -o "$f" '*.tif' -d . && rm "$f"; done```

```cd ..```

Next, move these files from your local machine to the virtual HPC, by first delete previous files stored in the directory (don't want to overload the memory on the HPC):

```cmd tbw```

Then, copy the DEM files from your local machine to the HPC:

```cmd tbw```

## Virtual Machine

We need to clone the repository, create and activate an environment, like in the first few steps above:

```git clone ```

```cd scd-detection```

```conda env create -n scd_env -f ./envs/enviroment.yml```

```conda activate scd_env```

To run the prediction model, inputting the appropriate inputs. Default command is below. MAKE SURE TO CHANGE NAME OF THE RUN OR IT WILL OVERWRITE THE PREVIOUS RESULTS FILE!!

```python3 scripts/predict.py --dem_path tiles --model_state_path models/MaskRCNN.pt --veto_model_state_path models/DEMVeto.pt --resolutions 1 3 5 --rgb_veto False --device cpu --run_name test```

This will output the results into ./results/RUN_NAME/predictions.shp

an example of a good run_name would be: 'SW_Cornwall_Zone1'. (could be a good idea to systematically split up England into small subregions in a named grid type thing so its clear through the file names.)






