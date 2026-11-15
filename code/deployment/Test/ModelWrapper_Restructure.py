import torch

from modelling.MaskRCNN import MaskRCNN


class InstanceModelWrapper:
    """Mask R-CNN wrapper that preserves scored instance masks."""

    def __init__(self, model_dict, score_threshold=0.60, mask_threshold=0.40):
        self.model = MaskRCNN()
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold

    def predict_instances(self, images, resolutions, device):
        images = images.to(device)
        resolutions = resolutions.to(device)
        self.model.to(device)
        if resolutions.dim() == 1:
            resolutions = resolutions.unsqueeze(1)

        outputs = self.model(
            list(images.unbind(0)),
            resolutions=resolutions,
        )
        instances = []
        for output in outputs:
            scores = output["scores"].detach().cpu()
            masks = output["masks"].detach().cpu()[:, 0]
            keep = scores >= self.score_threshold
            image_instances = []
            for mask, score in zip(masks[keep], scores[keep]):
                image_instances.append({
                    "mask": (mask >= self.mask_threshold).numpy(),
                    "score": float(score),
                })
            instances.append(image_instances)
        return instances
