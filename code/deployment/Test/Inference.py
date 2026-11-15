from deployment.Inference import SCDModel as BaseSCDModel
import deployment.config as config

from deployment.Test.Deployer_Restructure import Deployer


class SCDModel(BaseSCDModel):
    """SCDModel using instance-level inference and vetoing."""

    def _create_deployer(self, hyperparams, requested_returns):
        config.init(
            dem_path=self.dem_path,
            resolutions=tuple(self.resolutions),
            device=self.device,
            nodata=self.NODATA if self.NODATA is not None else -9999,
            to_veto=self.to_veto,
            **(hyperparams or {}),
            requested_returns=requested_returns,
        )
        return Deployer(self.MaskRCNN_model_state, self.veto_model_state)

    def set_params(self, **params):
        config.update(**params)
        self.deployer = Deployer(
            self.MaskRCNN_model_state,
            self.veto_model_state,
        )
