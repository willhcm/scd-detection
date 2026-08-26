# Google Colab Notebook

Please note that scripts run on colab were all stored in one Google Drive folder during development, so the import's wont make sense based on the repo structure. The deliverable of this specific project was not the codebase.

This directory contains four sub-directories:

## ```colab/data prep/``` 

This directory contains notebooks used to generate training datasets:

- ```DataPreperation.ipynb``` uses ScaleNormalisedDataStack.py to generate and export training tiles of SCDs in DEM data.
- ```VetoLabelling``` uses VetoDataset.py to manually label (tile-level binary classification) training samples for the veto classifier

## ```colab/inference/```

This directory contains notebooks used for inference:

- ```Deployment.ipynb``` is the notebook for inference. Everything from target DEM tiling to exporting final predictions is contained here.
- ```InitialUNETDeployment``` is the notebook saved from initial field target selection / modelling at the very beginning of the IRP. it is listed here because it is referenced in the repository. The initial UNET model can be found in ```colab/training/```

## ```colab/misc/``` 

This directory contains a few notebooks used for a variety of non-critical tasks. For example, downscaling the DEM so it can be downloaded and opened in ArcGIS/QGIS without taking up too much memory.

## ```colab/training/```

This directory contains notebooks used to train different models used throughout the development process. The main notebooks are:

- ```TrainFold``` is the notebook used to train one fold of the Mask R-CNN. 
- ```VetoClassifier``` is the notebook used to train the veto classifier.
- ```CompareModels.ipynb``` was used throughout to choose the best model (ultimately the Mask R-CNN)

The other notebooks were used to train different models that were considered in the development process. The respective scripts (if applicable) containing the model classes can be found in ```code/depreciated/```