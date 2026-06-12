from policyflow_torch.agents import PolicyFlowCfg
from isaaclab.utils import configclass

@configclass
class DebrisCaptureCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class ExcavationCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 5e-4
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.005
    brownian_reg_loss_scale = 0.005

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class PegInHoleCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class SampleCollectionCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class ScrewdrivingCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class SolarPanelAssemblyCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 5e-4
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0008
    brownian_reg_loss_scale = 0.0002

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class LandingCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class VelocityTrackingCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0

@configclass
class WaypointNavigationCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.0025
    brownian_reg_loss_scale = 0.0025

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0