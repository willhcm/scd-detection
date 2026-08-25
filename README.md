# Automatic SCD Detection for Natural Hydrogen Exploration using a Scale-Normalised Mask R-CNN with an Image-Pyramid

This repository contains the code for the deep learning strategy for scale-invariant cross-geography SCD detection, as outlined in the paper (in deliverables/). Code for reproducibility and accessing source data (and datasets collated in this study) can be found in code/reproduce.ipynb. 

## Repository Structure



```text
code/
├── deployment/
│   ├── Deployer.py
│   ├── ModelWrapper.py
│   ├── PostProcesser.py
│   └── README.md
├── depreciated/
├── helpers/
│   ├── geostats.py
│   ├── MaskRCNNFunctions.py
│   ├── raster_helpers.py
│   ├── veto_helpers.py
│   └── README.md
├── loaders/
│   ├── ScaleNormalisedDataStack.py
│   ├── VetoDataset.py
│   └── README.md
├── modelling/
│   ├── helpers.py
│   ├── MaskRCNN.py
│   ├── TrainingFunctions.py
│   ├── Veto.py
│   └── README.md
└── notebooks/
    ├── colab/
    ├── field/
    ├── geostats/
    ├── plots/
    │   ├── Table2.ipynb
    │   ├── Fig2a.ipynb
    │   ├── Fig2b.ipynb
    │   └── Fig5b.ipynb
    ├── validate.ipynb
    └── reproduce.ipynb
    
```

### Reproducibility

---

the directory ```notebooks/``` contains code to reproduce:

- ```reproduce.ipynb``` The training of the Mask R-CNN and the Veto classifier, and run inference. 
- ```validate.ipynb``` Leave-One-Region-Out cross validation of the Mask R-CNN.
- ```plots/``` Code to reproduce the plots displayed in the report. Data for these files can be found in ```datastore/```

This codebase has no other functionality than to train and deploy the Mask R-CNN inference pipeline, so the reproducibility section is not large.

### Environments

---

Due to clashes with PyProj versions and Rasterio (specifically for casting geopandas dataframes to different Coordinates Reference Systems), two environments are used: 

- `environment-model.yml` — model training and inference (e.g., ```Reproduce.ipynb``` and ```Validate.ipynb```)
- `environment-analysis.yml` — geospatial/statistical analysis (e.g., ```Fig5b.ipynb```)

environment.yml files were generated with assistance from ChatGPT.

### Datastore

---

Contains the source data and model results needed to reproduce anything written in the reproducibility section above. Also contains .svg files for all figures seen in the report.

### Depreciated

---

The depreciated directory contains many files which were used in development. Similarly, many small functions which previously used have been moved out of the main code/ directory into the depreciated/ directory to ensure that the main directory is clear and concise. 

## Modelling Methods

---

Predictions of candidate SCDs are generated using the following model architecture and inference strategy.

![ModelFigure](datastore/assets/Model.png)
*Schematic of deep learning methodology for scale-invariant SCD detection: (a)
adapted Mask R-CNN architecture for SCD detection, (b) inference strategy using a Scale-
Normalised Image Pyramid (SNIP) to maximise the receptive field of the inference mech-
anism, (c) DEM and DEM-derivative input channels, (d) architecture of the veto classifier
used to screen candidates. RPN = Regional Proposal Network, FPN = Feature Pyramid
Network, ROI = Region or Interest, FiLM = Feature-Wise Linear Modulation, CNN = Convolutional Neural Network, FFN = Feed Forward Network*

## AI Statement

---

AI was used throughout the IRP period for assistance with code generation, especially for the use of the rasterio package which I had not previously used. If an AI tool has had influence on the code, this is stated either above the relevant code or at the top of the python script where it's assistance is clearly explained. 





