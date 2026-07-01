from __future__ import annotations

def get_policyflow_models_cfg():
    return {
        "critic": {
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [512, 256, 128],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
        "actor": {
            "x_dim": 0,
            "emb_dim": 0,
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [512, 256, 128],
            "timestep_emb_type": "fourier",
        },
        "flow_sample_steps": 10,
    }

def get_ppo_models_cfg():
    return {
        "critic": {
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [512, 256, 128],
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
            "hidden_dims": [512, 256, 128],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
    }

def get_exo_models_cfg():
    """Model config for Exo agent (Actor-Critic with Gaussian policy)."""
    return {
        "critic": {
            "activations": ["mish", "mish", "mish", "linear"],
            "hidden_dims": [512, 256, 128],
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
            "hidden_dims": [512, 256, 128],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
    }

def get_runner_cfg():
    return {
        "max_iterations": 40000,
        "rollouts": 24,
        "save_interval": 100,
        "log_dir": "runs/ppo/mujoco",
        "experiment_name": None,
    }

