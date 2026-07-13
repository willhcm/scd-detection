from SegNet import SegNet
from FPNCentreNet import FPNCentreNet
from MaskRCNN import MaskRCNN
import torch
from FPNCentreNet import decode_multilevel_predictions, decode_centernet_predictions, simple_detection_nms

# args required to instantiate each model correctly.
INIT_ARGS = {}
PRED_ARGS = {'SegNet': {'threshold': 0.75},
             'MaskRCNN': {'score_threshold': 0.5,
                          'mask_threshold': 0.5},
            'FPNCentreNet': {'decode_args': {}}
             }


class ModelWrapper():

    def __init__(self, model_type, model_dict):

        if model_type not in ['SegNet', 'MaskRCNN', 'FPNCentreNet']:
            print('please enter a valid model type')
            return

        self.model_type = model_type
        self.model = model_type(**INIT_ARGS)
        self.model.load_state_dict(model_dict)
        self.pred_args = PRED_ARGS[model_type]
    
    def _predict_seg_net(self, images, device):
        images = images.to(device)
        logits = self.model(images)
        preds = (torch.sigmoid(logits) >= self.pred_args['threshold']).float()

        return preds

    def _pred_mask_rcnn(self, images, device):

        images = images.to(device)
        height, width = images.shape[-2:]
        output = self.model(images)
        scores = output["scores"].detach().cpu()
        raw_pred_masks = output["masks"].detach().cpu()
        keep = scores >= self.pred_args['score_threshold']
        pred_instance_masks = (
                    raw_pred_masks[keep, 0] >= self.pred_args[['mask_threshold']]
                )

        # convert from instance masks to combined masks.s
        if pred_instance_masks.shape[0] > 0:
            pred_mask = pred_instance_masks.any(dim=0)
        else:
            pred_mask = torch.zeros(
                (height, width),
                dtype=torch.bool,
            )
        
        return pred_mask

    def _pred_fpn_centrenet(self, images, device):
        images = images.to(device)
        # may need to add an extra dim as functions built before work for batches e.g. [B, H, W] not [H, W]
        outputs = self.model(images)
        detections = decode_multilevel_predictions(outputs, image_index=0)

        return detections

    def predict(self, images, device):
        """ 
        Public method
        
        Takes images on cpu, puts to device """

        if type(self.model) == SegNet:
            return self._predict_seg_net(self, images, device)
        
        elif type(self.model) == MaskRCNN:
            return self._pred_mask_rcnn(self, images, device)
        
        elif type(self.model) == FPNCentreNet:
            return self._pred_fpn_centrenet(self, images, device)
        else: # now already handled in init.
            print(f'Please enter a valid model, got {type(self.model)}')

    