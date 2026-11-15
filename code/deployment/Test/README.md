# Instance-level deployment pipeline

This experimental pipeline keeps each Mask R-CNN instance and applies the
veto model to that instance directly. It does not merge probability maps or
create connected components before vetoing. Accepted instance masks are
stamped into the final raster, which is polygonized only by `save_predictions`.

Use the same `deployment.config.init(...)` setup as the standard deployer,
or use the `SCDModel` wrapper:

```python
from deployment.Test.Inference import SCDModel

model = SCDModel(
    MaskRCNN_model_state=maskrcnn_state,
    veto_model_state=veto_state,
    dem_path=dem_path,
    device="cuda",
    resolutions=[1.0],
    out_path=output_path,
)
results = model.predict()
```

`score_threshold` filters Mask R-CNN instances and `veto_threshold` filters
those instances before the final stamp.

When multiple resolutions are configured, each resolution is vetoed on its own
grid first. Accepted masks are then transformed to the first resolution's grid
and overlaps are resolved in favor of the highest Mask R-CNN instance score.
Duplicate accepted masks with at least 0.2 mask IoU are suppressed before
stamping. Contained masks are also suppressed by default when 99% or more of
the smaller mask is covered by a larger mask, regardless of detector score.

The command-line equivalent is:

```bash
python scripts/predict.py --restructured_model --run_name instance_test
```
