# Helpers directory 

Python scripts here are used throughout the data preperation and modelling process

## geostats.py 

Used to perform geostatistics on the SCD predictions outputted by a model.

The StatsStack class reports the following statistics:

- KDE of SCD spatial distribution
- Morphology of predicted SCDs
    - Areas
    - Depth:Width ratio
    - Major Axis Length
    - Depths

## veto_helpers.py

Contains functions to help with data preperation for the veto classifier

Specifically:

- ```_make_annulus``` builds a scale-proportional ring (annulus) surrounding the predicted SCD 
- ```build_dem_context``` builds the larger DEM context (and slope map), and uses the annulus to generate scalar statistics of the SCD and its surrounding, including:
    - Annulus-SCD depth delta (mean, median, 90% percentile)
    - Absolute SCD size
    - Candidate slope
    - Annulus slope

## raster_helpers.py

A selection of miscellaneous functions to aid raster utilisation.

## MaskRCNNFunctions.py

Functions for MaskRCNN training and evaluation, including:

- ```collate_fn``` -- required for grouping instance masks for one sample
- ```mask_iou``` -- calculates Intersection over Union for two masks
- ```mask_rcnn_outputs_to_binary_mask``` --  converts instance masks batch to a list of masks (for final outputs)
- ```object_f1_from_instance_masks``` --  calculates object (SCD) level performance from mask rcnn instance masks (not merged masks)
- ```evaluate_maskrcnn_metrics``` -- calculates and returns a range of metrics for Mask R-CN performance.

and other functions for training the Mask R-CNN, specifically functions to implement incremental learning and fine-tuning.
