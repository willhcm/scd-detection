from MaskFilm import MaskRCNN
import torch

PRED_ARGS = {
             'MaskRCNN': {'score_threshold': 0.70,
                          'mask_threshold': 0.50},
             }


# simplified for just maskrcnn.
# remade to include pred_args nicely, and handle prediction more elegantly.
# need to make batch-friendly.
class ModelWrapper:

    def __init__(self, model_dict):

        self.model = MaskRCNN()
        self.model.load_state_dict(model_dict)
        self.pred_args = PRED_ARGS

    def _pred_mask_rcnn(self, images, resolutions, device):
        images = images.to(device)
        resolutions = resolutions.to(device)
        height, width = images.shape[-2:]

        # ai assistance with batch/channel here - was confused
        # torchvision detection models expect a list of [C, H, W]
        image_list = list(images.unbind(0))
        if resolutions.dim() == 1:
            resolutions = resolutions.unsqueeze(1)
 
        outputs = self.model(image_list, resolutions=resolutions)

        score_threshold = self.pred_args.get("score_threshold", 0.75)
        mask_threshold = self.pred_args.get("mask_threshold", 0.75)

        masks_out = []

        for output in outputs:
            scores = output["scores"].detach().cpu()
            raw_masks = output["masks"].detach().cpu()

            # roi level object confidence
            keep = scores >= score_threshold

            # mask level confidence
            instance_masks = raw_masks[keep, 0] >= mask_threshold

            if len(instance_masks) > 0:
                pred_mask = instance_masks.any(dim=0)
            else:
                pred_mask = torch.zeros((height, width), dtype=torch.bool)

            masks_out.append(pred_mask.float())

        return masks_out
    
    def predict(self, images, resolutions, device):
        return self._pred_mask_rcnn(images, resolutions, device)