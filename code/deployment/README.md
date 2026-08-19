# Deployment

The two files in this directory are used for inference. The deployer is the main object which is used to generate final predictions, and it calls the postprocesser internally.

The inference workflow works in this (not exhaustive) order:

1. Models are instantiated (using ModelWrapper.py)
2. Predictions are generated at each inputted resolution (using an image--pyramid approach)
3. Predictions are merged into one layer using a support-based voting system 
4. The veto classifier screens out bad predictions
5. Final predicted are sieved based on native DEM resolutions and a typical minimum SCD sizes.