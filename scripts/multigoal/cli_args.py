from __future__ import annotations

import torch
from policyflow_torch.agents import PolicyFlowCfgInstance, PPOCfgInstance


def get_policyflow_agent_cfg():
    cfg = PolicyFlowCfgInstance()
    cfg.desired_kl = 0.01
    cfg.learning_rate = 2e-4
    cfg.discount_factor = 0.99
    cfg.lam = 0.95
    cfg.time_limit_bootstrap = True
    cfg.mini_batches = 4
    cfg.learning_epochs = 5
    cfg.gaussian_entropy_loss_scale = 0.001
    cfg.brownian_reg_loss_scale = 0.25
    cfg.ratio_clip = 0.2
    cfg.clip_predicted_values = True
    cfg.value_clip = 0.2
    cfg.value_loss_scale = 1.0
    cfg.grad_norm_clip = 1.0
    return cfg

def get_ppo_agent_cfg():
    cfg = PPOCfgInstance()
    cfg.desired_kl = 0.01
    cfg.learning_rate = 2e-4
    cfg.discount_factor = 0.99
    cfg.lam = 0.95
    cfg.time_limit_bootstrap = True
    cfg.mini_batches = 4
    cfg.learning_epochs = 5
    cfg.entropy_loss_scale = 0.001
    cfg.ratio_clip = 0.2
    cfg.clip_predicted_values = True
    cfg.value_clip = 0.2
    cfg.value_loss_scale = 1.0
    cfg.grad_norm_clip = 1.0
    return cfg

def get_policyflow_models_cfg():
    return {
        "critic": {
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [256, 128, 64],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
        "actor": {
            "x_dim": 0,
            "emb_dim": 0,
            "hidden_dims": [256, 128, 64],
            "activations": ["mish", "mish", "mish", "linear"],
            "timestep_emb_type": "fourier",
        },
        "flow_sample_steps": 10,
    }

def get_ppo_models_cfg():
    return {
        "critic": {
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [256, 128, 64],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
        "actor": {
            "log_std_max": 4.0,
            "log_std_min": -20.0,
            "std_init": 1.0,
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [256, 128, 64],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
    }


def get_runner_cfg():
    return {
        "max_iterations": 40000,
        "rollouts": 32,
        "save_interval": 100,
        "log_dir": "runs/multigoal",
        "experiment_name": None,
    }
