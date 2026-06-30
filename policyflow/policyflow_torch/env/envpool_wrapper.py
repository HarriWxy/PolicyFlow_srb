from typing import Any, Tuple, Dict
import torch
from .base import Wrapper
import envpool
import numpy as np


class EnvPoolWrapper(Wrapper):
    def __init__(self, env_id: str, num_envs: int = 1, **kwargs) -> None:
        """EnvPool environment wrapper

        :param env_id: The environment ID
        :type env_id: str
        :param num_envs: Number of parallel environments
        :type num_envs: int
        :param kwargs: Additional arguments to pass to envpool.make
        """
        env = envpool.make(env_id, env_type="gymnasium", num_envs=num_envs, **kwargs)
        super().__init__(env)
        self._num_envs = num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        return self._device

    def _filter_info(self, env_info):
        """Filter envpool info to only include tensor/numeric values."""
        info = {"log": {}}
        if isinstance(env_info, dict):
            for key, value in env_info.items():
                if isinstance(value, np.ndarray):
                    info["log"][key] = torch.from_numpy(value).float().to(device=self.device)
                elif isinstance(value, torch.Tensor):
                    info["log"][key] = value.float().to(device=self.device)
                elif isinstance(value, (int, float)):
                    info["log"][key] = torch.tensor([value], dtype=torch.float, device=self.device)
                # skip dict, list, string and other non-numeric types
        return info

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        actions_np = actions.cpu().numpy()
        obs, reward, terminated, truncated, env_info = self._env.step(actions_np)

        # Convert to tensors
        obs_tensor = torch.from_numpy(obs).float().to(device=self.device)
        reward_tensor = torch.from_numpy(reward).float().to(device=self.device)
        terminated_tensor = torch.from_numpy(terminated).to(device=self.device)

        info = self._filter_info(env_info)
        info["time_outs"] = torch.from_numpy(truncated).to(device=self.device)

        # Create observations dictionary
        observations_dict = {
            "actor_observations": obs_tensor.clone(),
            "critic_observations": obs_tensor.clone(),
        }

        return (
            observations_dict,
            reward_tensor,
            terminated_tensor,
            info,
        )

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        obs, env_info = self._env.reset()
        
        # Convert to tensors
        obs_tensor = torch.from_numpy(obs).float().to(device=self.device)
        
        info = self._filter_info(env_info)
        
        observations_dict = {
            "actor_observations": obs_tensor.clone(),
            "critic_observations": obs_tensor.clone(),
        }
        return observations_dict, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        return None

    def close(self) -> None:
        """Close the environment"""
        if hasattr(self._env, 'close'):
            self._env.close()