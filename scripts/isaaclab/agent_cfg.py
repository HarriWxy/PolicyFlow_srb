from policyflow_torch.agents import PolicyFlowCfg
from isaaclab.utils import configclass

@configclass
class FrankaArmCfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-4
    discount_factor = 0.98
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.004
    brownian_reg_loss_scale = 0.002

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0


@configclass
class NavigationCfg(PolicyFlowCfg):
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
class QuadcopterCfg(PolicyFlowCfg):
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
class AnymalDCfg(PolicyFlowCfg):
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
class UnitreeH1Cfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
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
class UnitreeGo2Cfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.008
    brownian_reg_loss_scale = 0.002

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0


@configclass
class UnitreeG1Cfg(PolicyFlowCfg):
    desired_kl = 0.01
    learning_rate = 1e-3
    discount_factor = 0.99
    lam = 0.95
    time_limit_bootstrap = True
    mini_batches = 4
    learning_epochs = 5
    gaussian_entropy_loss_scale = 0.002
    brownian_reg_loss_scale =0.006

    ratio_clip = 0.2
    clip_predicted_values = True
    value_clip = 0.2
    value_loss_scale = 1.0
    grad_norm_clip = 1.0
    
@configclass
class CabinetCfg(PolicyFlowCfg):
    desired_kl = 0.02
    learning_rate = 5.0e-4
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
