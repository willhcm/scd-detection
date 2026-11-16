# Logging into the HPC

```ssh ph122@borg-login.ese.ic.ac.uk``` (connects to the HPC)

```cd /scratch_root/ph122/``` (Navigate to your file store on the cluster)

```conda activate scd_env``` (Activate the environment)

Now, we need to move the models over to the HPC, so exit (by typing ```exit``` in the terminal/command line when connected to the HPC). Once you are 'back on' the laptop, move the models over:

```scp -r downloads/models IMPERIAL_SHORTCODE@borg-login.ese.ic.ac.uk:/scratch_root/IMPERIAL_SHORTCODE/scd-detection```

 Create a .txt file on your local machine as a line-by-line list of tiles you want the model to merge and predict over (from DEFRA website is the best way). Move the file over to the HPC by doing the following command on your laptop:

```scp -r path/to/example_request.txt IMPERIAL_SHORTCODE@borg-login.ese.ic.ac.uk:/scratch_root/IMPERIAL_SHORTCODE/scd-detection``` 

Now, connect back on to the HPC to generate URLs, download from URLs and run the model.

Then, in the HPC, generate the urls by doing the below command. Customise the example_request and example_request_name inputs to a suitable name to describe the region they represent, for example.

```python3 scripts/urls.py --request requests/example_reques_list.txt --request_name example_request_name```

Second, download the files using curl. The files will be stored in a directory on the HPC called tiles/

while read -r u; do
  curl -fL -J -O --output-dir ./tiles \
    --retry 5 --retry-delay 10 --retry-all-errors "$u" || echo "$u" >> failed.txt
  sleep 3
done < urls/CHANGE_THIS.txt

Third, unzip all the files (inplace), keeping only the .tif files (the actual data):

```cd tiles```

```for f in *.zip; do unzip -j -o "$f" '*.tif' -d . && rm "$f"; done```

```cd ..```

Edit the job submission file (run.sh) (from scd-detection):

```nano runs/run.sh```

This will output the results into ./results/RUN_NAME/predictions.shp

an example of a good run_name would be: 'SW_Cornwall_Zone1'. (could be a good idea to systematically split up England into small subregions in a named grid type thing so its clear through the file names.)
