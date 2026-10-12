from pathlib import Path
import sys
CODE_DIR = Path("../../code").resolve()
sys.path.insert(0, str(CODE_DIR))
from modelling.MaskRCNN import MaskRCNN
import torch

class ModelWrapper:
    """
    A wrapper class for the Mask R-CNN model used for prediction.
    """

    def __init__(self, model_dict, score_threshold=0.75, mask_threshold=0.50):

        self.model = MaskRCNN()
        self.model.load_state_dict(model_dict)
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

    # edited to return actual prob map not thresholded masks.
    def _pred_mask_rcnn(
        self,
        images,
        resolutions,
        device):

        images = images.to(device)
        resolutions = resolutions.to(device)
        height, width = images.shape[-2:]

        # Torchvision detection models expect a list of
        # tensors with shape [C, H, W].
        image_list = list(images.unbind(0))
        if resolutions.dim() == 1:
            resolutions = resolutions.unsqueeze(1)

        outputs = self.model(image_list, resolutions=resolutions)


        probability_maps = []

        for output in outputs:

            scores = output["scores"].detach().cpu()
            raw_masks = (output["masks"].detach().cpu()[:, 0])
            keep = scores >= self.score_threshold

            if keep.any():

                kept_masks = raw_masks[keep]

                kept_masks = torch.where(kept_masks >= self.mask_threshold, kept_masks, torch.zeros_like(kept_masks))

                tile_probability = (kept_masks.max(dim=0).values)

            else:
                tile_probability = torch.zeros((height, width), dtype=torch.float32)

            probability_maps.append(tile_probability.float())

        return probability_maps

    def predict(self, images, resolutions, device):
        return self._pred_mask_rcnn(images, resolutions, device)