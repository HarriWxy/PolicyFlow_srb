from __future__ import annotations

import argparse


def get_parser():
    # add argparse arguments
    parser = argparse.ArgumentParser(description="Train an RL agent with PolicyFlow.")
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during training.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Length of the recorded video (in steps).",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of environments to simulate."
    )
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed used for the environment"
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="RL Policy training iterations.",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory.")
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name."
    )
    parser.add_argument("--rollouts", type=int, default=None, help="Rollouts.")
    parser.add_argument(
        "--save_interval", type=int, default=None, help="Save interval."
    )
    parser.add_argument(
        "--actor_hidden_dims",
        type=int,
        nargs="+",
        default=None,
        help="Actor hidden dims.",
    )
    parser.add_argument(
        "--actor_activations",
        type=str,
        nargs="+",
        default=None,
        help="Actor activations.",
    )
    parser.add_argument(
        "--critic_hidden_dims",
        type=int,
        nargs="+",
        default=None,
        help="Critic hidden dims.",
    )
    parser.add_argument(
        "--critic_activations",
        type=str,
        nargs="+",
        default=None,
        help="Critic activations.",
    )
    parser.add_argument(
        "--obs_embeding_dims", type=int, default=64, help="Observation embeding dims."
    )
    return parser


def get_models_cfg():
    return {
        "critic": {
            "activations": ["elu", "elu", "elu", "linear"],
            "hidden_dims": [512, 256, 128],
            "init_fade": True,
            "init_gain": 1.0,
            "input_normalization": False,
            "recurrent": False,
        },
        "actor": {
            "x_dim": 0,
            "emb_dim": 0,
            "activations": ["elu", "elu", "elu", "linear"],
            "hidden_dims": [512, 256, 128],
            "timestep_emb_type": "fourier",
        },
        "flow_sample_steps": 10,
    }


def get_runner_cfg():
    return {
        "max_iterations": 40000,
        "rollouts": 24,
        "save_interval": 100,
        "log_dir": "runs/policyflow",
        "experiment_name": None,
    }


def override_runner_cfg(runner_cfg: dict, args_cli) -> dict:
    """
    Batch override parameters in runner_cfg with CLI arguments, only overriding keys that exist in runner_cfg

    Args:
        runner_cfg: Runner configuration dictionary
        args_cli: Command line arguments object

    Returns:
        Updated runner_cfg dictionary
    """
    # Define mapping from CLI arguments to runner_cfg keys
    cli_to_runner_mapping = {
        "max_iterations": "max_iterations",
        "experiment_name": "experiment_name",
        # Add more mappings as needed
        "log_dir": "log_dir",
        "rollouts": "rollouts",
        "save_interval": "save_interval",
    }

    # Iterate through mappings, only override existing keys
    for cli_attr, runner_key in cli_to_runner_mapping.items():
        if hasattr(args_cli, cli_attr) and runner_key in runner_cfg:
            cli_value = getattr(args_cli, cli_attr)
            if cli_value is not None:  # Only override non-None values
                runner_cfg[runner_key] = cli_value

    return runner_cfg


def override_model_cfg(model_cfg: dict, args_cli) -> dict:
    """
    Batch override parameters in model_cfg with CLI arguments, supporting nested keys

    Args:
        model_cfg: Model configuration dictionary
        args_cli: Command line arguments object

    Returns:
        Updated model_cfg dictionary
    """
    # Define mapping from CLI arguments to model_cfg nested paths
    cli_to_model_mapping = {
        # Add more mappings as needed
        "actor_hidden_dims": ["actor", "hidden_dims"],
        "actor_activations": ["actor", "activations"],
        "critic_hidden_dims": ["critic", "hidden_dims"],
        "critic_activations": ["critic", "activations"],
    }

    # Iterate through mappings, only override existing paths
    for cli_attr, model_path in cli_to_model_mapping.items():
        if hasattr(args_cli, cli_attr):
            cli_value = getattr(args_cli, cli_attr)
            if cli_value is not None:  # Only override non-None values
                # Check if the path exists in model_cfg
                current = model_cfg
                path_exists = True
                for key in model_path[:-1]:
                    if key not in current:
                        path_exists = False
                        break
                    current = current[key]

                if path_exists and model_path[-1] in current:
                    _set_nested_dict_value(model_cfg, model_path, cli_value)

    return model_cfg


def _set_nested_dict_value(d: dict, path: list, value):
    """
    Safely set a value in a nested dictionary using a path list

    Args:
        d: Dictionary to modify
        path: List of keys representing the path (e.g., ['actor', 'flow_model_para', 'lr'])
        value: Value to set
    """
    current = d
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value
