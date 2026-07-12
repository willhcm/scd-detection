from SegNet import SegNet
from FPNCentreNet import FPNCentreNet
from MaskRCNN import MaskRCNN

# args required to instantiate each model correctly.
# fill later.
init_args = {}


class ModelWrapper():

    def __init__(self, model_type, model_dict):

        self.model_type = model_type
        self.model = model_type(**init_args)
        self.model.load_state_dict(model_dict)
    
    def _predict_seg_net(self, images):
        ...

    def _pred_mask_rcnn(self, images):
        ...

    def _pred_fpn_centrenet(self, images):
        ...

    def predict(self, images):

        if type(self.model) == SegNet:
            return self._predict_seg_net(self, images)
        
        elif type(self.model) == MaskRCNN:
            return self._pred_mask_rcnn(self, images)
        
        elif type(self.model) == FPNCentreNet:
            return self._pred_fpn_centrenet(self, images)
        else:
            print(f'Please enter a valid model, got {type(self.model)}')

    