# Setting up your local machine

All commands are to be run on the command line.

## Cloning the GitHub repository locally.

Navigate to your desired destination directory, e.g.:

```cd documents/ ```

```git clone https://github.com/willhcm/scd-detection.git```

## Python Environment

First, create the python environment. Install 'miniconda' from the internet.

Then, Navigate into the scd-detection repository directory:

```cd documents/scd-detection``` (equivalent command to navigate through your files)

and run:

```conda env create -f enviroment.yml```

After this has been done the first time, simply activate the environment each time you log-on:

```conda activate scd_env```

## Model Download. (Only needs to be done once)

Download the two models (MaskRCNN, DEMVeto) to your local machine from this [link](https://drive.google.com/drive/folders/1SP--9SlvA7zn0S0yoilJTJe9A-x-umr_?usp=sharing). Keep these on your laptop for now, until the HPC set-up stage.

# Setting up the HPC and running the model

On the HPC, we need to clone the repository, create and activate an environment, just like in the first few steps above (only need to be done once):

```ssh IMPERIAL_SHORTCODE@borg-login.ese.ic.ac.uk``` (connects to the HPC)

```cd /scratch_root/IMPERIAL_SHORTCODE/``` (Navigate to your file store on the cluster)

```git clone https://github.com/willhcm/scd-detection.git``` (clone the underlying code for the prediction system)

```cd scd-detection``` (Navigate into the codebase)

```conda env create -n scd_env -f enviroment.yml``` (Create the environment)

```conda activate scd_env``` (Activate the environment)

Now, we need to move the models over to the HPC, so exit (by typing ```exit``` in the terminal/command line when connected to the HPC). Once you are 'back on' the laptop, move the models over:

```scp -r downloads/models IMPERIAL_SHORTCODE@borg-login.ese.ic.ac.uk:/scratch_root/IMPERIAL_SHORTCODE/scd-detection```


 Create a .txt file on your local machine as a line-by-line list of tiles you want the model to merge and predict over (from DEFRA website is the best way). Move the file over to the HPC by doing the following command on your laptop:

```scp -r path/to/example_request.txt IMPERIAL_SHORTCODE@borg-login.ese.ic.ac.uk:/scratch_root/IMPERIAL_SHORTCODE/scd-detection``` 

Now, connect back on to the HPC to generate URLs, download from URLs and run the model.


Then, in the HPC, generate the urls by doing the below command. Customise the example_request and example_request_name inputs to a suitable name to describe the region they represent, for example.

```python3 scripts/urls.py --request example_request.txt --request_name example_request_name```

Second, download the files using curl. The files will be stored in a directory on the HPC called tiles/

```xargs -n 1 curl -L -J -O --output-dir ./tiles < ./urls/example_request_name_urls.txt```

Third, unzip all the files (inplace), keeping only the .tif files (the actual data):

```cd tiles```

```for f in *.zip; do unzip -j -o "$f" '*.tif' -d . && rm "$f"; done```

```cd ..```

To run the prediction model, inputting the appropriate inputs. Default command is below.

```python3 scripts/predict.py --dem_path tiles --model_state_path models/MaskRCNN.pt --veto_model_state_path models/DEMVeto.pt --resolutions 1 3 5 --rgb_veto False --device cpu --run_name test```

This will output the results into ./results/RUN_NAME/predictions.shp

an example of a good run_name would be: 'SW_Cornwall_Zone1'. (could be a good idea to systematically split up England into small subregions in a named grid type thing so its clear through the file names.)






