# Logging into the HPC

```ssh ph122@borg-login.ese.ic.ac.uk``` (connects to the HPC)

```cd /scratch_root/ph122/``` (Navigate to your file store on the cluster)

```conda activate scd_env``` (Activate the environment)

 Create a .txt file on your local machine (!!) as a line-by-line list of tiles you want the model to merge and predict over (from DEFRA website is the best way). Move the file over to the HPC by doing the following command on your laptop:

```scp -r documents/SCDs/Requests/example_request_list.txt ph122@borg-login.ese.ic.ac.uk:/scratch_root/ph122/scd-detection/requests``` 

Then, in the HPC, generate the urls by doing the below command. Customise the example_request and example_request_name inputs to a suitable name to describe the region they represent, for example.

```python3 scripts/urls.py --request requests/example_reques_list.txt --request_name example_request_name```

Second, download the files using curl. The files will be stored in a directory on the HPC called tiles/

```xargs -n 1 curl -L -J -O --output-dir ./tiles < ./urls/example_request_name_urls.txt```

Third, unzip all the files (inplace), keeping only the .tif files (the actual data):

```cd tiles```

```for f in *.zip; do unzip -j -o "$f" '*.tif' -d . && rm "$f"; done```

```cd ..```

Edit the job submission file (run.sh) (from scd-detection):

```nano runs/run.sh```

Change the job name (and tile directory if you are using another tile folder for some reason but probs not).

Save using CTRL+O, and exit using CTRL+X. 

Submit the job (from scd-detection folder) using:

```sbatch runs/run.sh```

This will output the results into ./results/RUN_NAME/predictions.shp. You can view the queue / job status using:

```squeue -u ph122```

Copy the results (on your Windows command line) using:

```scp -r p122@borg-login.ese.ic.ac.uk:/scratch_root/ph122/scd-detection/results/EXAMPLE_RUN_NAME documents/SCDs/results```

This will move the predictions.shp (and others) files onto your local machine, in the SCDs/results folder.

If you are happy with the results, clear the scd-detection tiles directory (where the DEMs are stored from the last run):

Make sure you are in scd-detection (```scratch_root/ph122/scd-detection```), and do:

```rm -rf tiles```

You are now ready to repeat the process!





