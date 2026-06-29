from typing import Any, Tuple, Dict
import torch
import warnings
from policyflow_torch.env import Wrapper
from collections import deque

try:
    from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
except:
    pass


def _sanitize_tensor(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """检测并修复张量中的 NaN/Inf 值，用 0 替换。返回是否包含非法值。"""
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    bad_mask = nan_mask | inf_mask
    if bad_mask.any():
        n_bad = bad_mask.sum().item()
        n_total = tensor.numel()
        warnings.warn(
            f"[NaN Guard] {name}: {n_bad}/{n_total} values are NaN/Inf, replacing with 0"
        )
        tensor = tensor.clone()
        tensor[bad_mask] = 0.0
    return tensor


class IsaacLabEnvWrapper(Wrapper):
    def __init__(
        self,
        env,
        using_historical_obs: bool = False,
        critic_obs_len: int = 1,
        actor_obs_len: int = 1,
    ) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(
            env.unwrapped, DirectRLEnv
        ):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        super().__init__(env)

        self._using_historical_obs = using_historical_obs
        if self._using_historical_obs:
            self.actor_obs_buffer = deque(maxlen=actor_obs_len)
            self.critic_obs_buffer = deque(maxlen=critic_obs_len)

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        # record step information
        obs_dict, reward, terminated, truncated, env_info = self._env.step(actions)

        # === NaN Guard: 在数据进入 agent 之前消毒 ===
        obs_dict["policy"] = _sanitize_tensor(obs_dict["policy"], "obs/policy")
        if "critic" in obs_dict:
            obs_dict["critic"] = _sanitize_tensor(obs_dict["critic"], "obs/critic")
        reward = _sanitize_tensor(reward, "reward")

        # compute dones for compatibility with PolicyFlow
        dones = (terminated | truncated).to(dtype=torch.long)

        if self._using_historical_obs:
            env_ids = dones.nonzero(as_tuple=False).flatten()
            for i in range(self.actor_obs_buffer.maxlen):
                self.actor_obs_buffer[i][env_ids] *= 0.0
            for i in range(self.critic_obs_buffer.maxlen):
                self.critic_obs_buffer[i][env_ids] *= 0.0

            self.actor_obs_buffer.append(obs_dict["policy"])

            if "critic" in obs_dict:
                self.critic_obs_buffer.append(obs_dict["critic"])
            else:
                self.critic_obs_buffer.append(obs_dict["policy"])

            actor_obs = torch.cat(
                [self.actor_obs_buffer[i] for i in range(self.actor_obs_buffer.maxlen)],
                dim=-1,
            )
            critic_obs = torch.cat(
                [
                    self.critic_obs_buffer[i]
                    for i in range(self.critic_obs_buffer.maxlen)
                ],
                dim=-1,
            )
            observations_dict = {
                "actor_observations": actor_obs,
                "critic_observations": critic_obs,
            }
        else:
            observations_dict = {
                "actor_observations": obs_dict["policy"],
                "critic_observations": (
                    obs_dict["critic"] if "critic" in obs_dict else obs_dict["policy"]
                ),
            }

        if not self.unwrapped.cfg.is_finite_horizon:
            env_info["time_outs"] = truncated

        return (
            observations_dict,
            reward,
            dones,
            env_info,
        )

    def reset(self) -> Tuple[Dict[str, torch.Tensor], Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        obs_dict, env_info = self._env.reset()

        # === NaN Guard: 消毒 reset 后的观测值 ===
        obs_dict["policy"] = _sanitize_tensor(obs_dict["policy"], "reset/obs/policy")
        if "critic" in obs_dict:
            obs_dict["critic"] = _sanitize_tensor(obs_dict["critic"], "reset/obs/critic")

        if self._using_historical_obs:
            for _ in range(self.actor_obs_buffer.maxlen):
                self.actor_obs_buffer.append(torch.zeros_like(obs_dict["policy"]))
            if "critic" in obs_dict:
                for _ in range(self.critic_obs_buffer.maxlen):
                    self.critic_obs_buffer.append(torch.zeros_like(obs_dict["critic"]))
            else:
                for _ in range(self.critic_obs_buffer.maxlen):
                    self.critic_obs_buffer.append(torch.zeros_like(obs_dict["policy"]))

            self.actor_obs_buffer.append(obs_dict["policy"])
            if "critic" in obs_dict:
                self.critic_obs_buffer.append(obs_dict["critic"])
            else:
                self.critic_obs_buffer.append(obs_dict["policy"])

            actor_obs = torch.cat(
                [self.actor_obs_buffer[i] for i in range(self.actor_obs_buffer.maxlen)],
                dim=-1,
            )
            critic_obs = torch.cat(
                [
                    self.critic_obs_buffer[i]
                    for i in range(self.critic_obs_buffer.maxlen)
                ],
                dim=-1,
            )
            observations_dict = {
                "actor_observations": actor_obs,
                "critic_observations": critic_obs,
            }
        else:
            observations_dict = {
                "actor_observations": obs_dict["policy"],
                "critic_observations": (
                    obs_dict["critic"] if "critic" in obs_dict else obs_dict["policy"]
                ),
            }
        return observations_dict, env_info

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self._env.render_mode

    def close(self):  # noqa: D102
        return self._env.close()
