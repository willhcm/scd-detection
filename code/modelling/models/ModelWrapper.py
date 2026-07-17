from SegNet import SegNet
from FPNCentreNet import FPNCentreNet
from MaskRCNN import MaskRCNN
import torch
from FPNCentreNet import decode_multilevel_predictions

# args required to instantiate each model correctly.
INIT_ARGS = {}
PRED_ARGS = {'SegNet': {'threshold': 0.75},
             'MaskRCNN': {'score_threshold': 0.77,
                          'mask_threshold': 0.55},
            'FPNCentreNet': {'threshold_0': 0.6,
                             'threshold_1': 0.7,
                             'threshold_2': 0.8,
                             'threshold_3': 0.7}
             }


# remade to include pred_args nicely, and handle prediction more elegantly.
class ModelWrapper:

    def __init__(self, model_type, model_dict):
        self.model_type = model_type
        self.model = model_type()
        self.model.load_state_dict(model_dict)
        self.pred_args  = self._resolve_pred_args()

    def _resolve_pred_args(self):

        if isinstance(self.model, SegNet):
            return PRED_ARGS['SegNet']
        elif isinstance(self.model, MaskRCNN):
            return PRED_ARGS['MaskRCNN']
        elif isinstance(self.model, FPNCentreNet):
            return PRED_ARGS['FPNCentreNet']
        else:
            print('enter valid model')

    def _predict_seg_net(self, images, device):
        images = images.to(device)
        logits = self.model(images)

        preds = (
            torch.sigmoid(logits)
            >= self.pred_args["threshold"]
        ).float()

        # Remove batch and channel dimensions: [1, 1, H, W] -> [H, W]
        return preds[0, 0].detach().cpu()

    def _pred_mask_rcnn(self, images, device):
        images = images.to(device)
        height, width = images.shape[-2:]

        outputs = self.model(images)
        output = outputs[0]

        scores = output["scores"].detach().cpu()
        raw_masks = output["masks"].detach().cpu()

        keep = scores >= self.pred_args.get(
            "mask_rcnn_threshold",
            0.5,
        )

        instance_masks = (
            raw_masks[keep, 0]
            >= self.pred_args.get("mask_threshold", 0.5)
        )

        if len(instance_masks) > 0:
            pred_mask = instance_masks.any(dim=0)
        else:
            pred_mask = torch.zeros(
                (height, width),
                dtype=torch.bool,
            )

        return pred_mask.float()

    def _pred_fpn_centrenet(self, images, device):
        images = images.to(device)

        height, width = images.shape[-2:]

        outputs = self.model(images)

        detections = decode_multilevel_predictions(
            outputs,
            image_index=0,
        )

        pred_mask = torch.zeros(
            (height, width),
            dtype=torch.float32,
        )

        yy, xx = torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )

        for detection in detections:

            level = detection['level']
            score_threshold = self.pred_args.get(
            f'threshold_{level}',
            0.5,
        )
            if isinstance(detection, dict):
                
                cx = detection.get("cx")
            
                cy = detection.get("cy")
                
                radius = detection.get("radius")

                score = detection.get('score')

            else:
                cx, cy, radius, score = detection[:4]

            if torch.is_tensor(score):
                score = score.item()

            if score < score_threshold:
                continue


            # Convert scalar tensors to Python values.
            if torch.is_tensor(cx):
                cx = cx.item()

            if torch.is_tensor(cy):
                cy = cy.item()

            if torch.is_tensor(radius):
                radius = radius.item()

            radius = max(float(radius), 1.0)

            circle = (
                (xx - float(cx)) ** 2
                + (yy - float(cy)) ** 2
            ) <= radius ** 2

            pred_mask[circle] = 1.0

        return pred_mask

    def predict(self, images, device):
        """
        Return one [H, W] CPU prediction mask for every model type.
        """

        if isinstance(self.model, SegNet):
            return self._predict_seg_net(images, device)

        elif isinstance(self.model, MaskRCNN):
            return self._pred_mask_rcnn(images, device)

        elif isinstance(self.model, FPNCentreNet):
            return self._pred_fpn_centrenet(images, device)

        raise TypeError(
            f"Unsupported model type: {type(self.model).__name__}"
        )