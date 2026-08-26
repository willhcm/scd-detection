# Datastore

The files in this directory can be used to reproduce plots and results demonstrated in ../code/notebooks/.

```IRPFinal/``` contains the .tex file for the report. The images can already be found in ```assets/```, and haven't been included in the tex project folder.

## Structure 

```text
datastore/
├── results/
│   ├── CVPerformance.pkl # Cross-validation Mask R-CNN performance (Fig2b)
│   ├── MaskRCNNFinal.pkl # Learning curves for Mask R-CNN (Fig2a)
│   └── veto_learning.pkl # Learning curves for Veto classifier(Fig2a)
├── spatial_analysis/
│   ├── CentroidsFinal.shp # And related files. Holds ESRI Point geometry for model predictions in the Field Region. Figure 5b.
│   └── FaultsFinal.shp # And related files. Holds faults mapped from BGS 50,000:1 geological maps. Figure 5b.
├── assets/
│   ├── Figures/
│   │   └── Figures 1 to 5 seen in the report as .svg files.
│   └── Model.png # image of model architecture for main repo readme
├── scd_sizes.pkl # Dictionary of SCD sizes in different regions (from GT labels). Used for Figure 2b.
├── SCDs.csv # Contains field soil gas hydrogen concentrations from SCDs in Somerset. Used for Table 2.
├── SCDs(Oake1).csv # Contains field soil gas hydrogen concentrations from one SCD in Somerset. Not used for any figures in the report.
├── Transect.csv # Contains field soil gas hydrogen concentrations from the 21 km transect in Somerset. Used for Table 2.
└── README.md # this page :) 
```

To access larger data, such as the custom training datasets for the Mask R-CNN and veto classifier, or an example DEM /RGB scene for inference, see below.

## Mask R-CNN training set

[Here](https://drive.google.com/drive/folders/1w9F2HbTzQ7DbAh4oUd7-YH5wfBN3375j?usp=sharing)

## Veto Classifier training set

[Here](https://drive.google.com/drive/folders/1CTs-U8nBEm4UzlBA94teXTSR1qlH4u6n?usp=sharing)

## Example DEM and RGB scene 

DEM and RGB scene for the southern margin of the Bristol Channel, where fieldwork was undertaken.

[Here](https://drive.google.com/drive/folders/12ZN8SewFaRC5t6njgYwXjDDFWkaSJTch?usp=sharing)

