from typing import Tuple
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
import torch
import numpy as np
import os


class MultiGoalEnv:
    """
    Batched version of MultiGoalEnv that can handle multiple environments in parallel.
    Uses PyTorch tensors for efficient computation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dt = cfg.get("dt", 0.01)
        self.num_envs = cfg.get("num_envs", 1024)
        self.num_obs = 4  # 2 for state (x, y) and 2for state (v_x, v_y)
        self.num_actions = 2
        self.max_episode_length = cfg.get("max_episode_length", 1000)
        self.check_collide_wall_flag = cfg.get("check_collide_wall", True)
        self.random_init_episode_length = cfg.get("random_init_episode_length", True)

        self.unwrapped = self

        # Initialize buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs),
            dtype=torch.float,
            device=cfg.device,
            requires_grad=False,
        )
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=cfg.device, requires_grad=False
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=cfg.device, requires_grad=False
        )
        self.truncated_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=cfg.device, requires_grad=False
        )

        # State and goal buffers
        self.pos_state_buf = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float,
            device=cfg.device,
            requires_grad=False,
        )
        self.vel_state_buf = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.float,
            device=cfg.device,
            requires_grad=False,
        )
        # Goals placed evenly on a circle
        self.num_goals = int(cfg.get("num_goals", 5))
        goal_radius = float(cfg.get("goal_radius", 5.0))
        center_list = cfg.get("goal_center", [0.0, 0.0])
        goal_center = torch.tensor(center_list, dtype=torch.float, device=cfg.device)
        angles = torch.linspace(
            0.0, 2 * torch.pi, steps=self.num_goals + 1, device=cfg.device
        )[:-1]
        xs = goal_center[0] + goal_radius * torch.cos(angles)
        ys = goal_center[1] + goal_radius * torch.sin(angles)
        self.goal_positions = torch.stack([xs, ys], dim=1).to(dtype=torch.float)

        # Environment parameters
        self.xlim = cfg.get("xlim", [-7.0, 7.0])
        self.ylim = cfg.get("ylim", [-7.0, 7.0])

        self.extras = {}
        self.device = cfg.device

        self._prepare_reward_function()

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = OmegaConf.to_container(self.cfg.rewards.scales)
        for key in list(self.reward_scales.keys()):
            self.reward_scales[key] *= self.dt
            if self.reward_scales[key] == 0:
                self.reward_scales.pop(key)
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "reward_" + name
            self.reward_functions.append(getattr(self, name))
        # reward episode sums
        self.episode_reward_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def close(self):
        pass

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # fill extras
        self.extras["log"] = {}
        for key in self.episode_reward_sums.keys():
            self.extras["log"]["rew_" + key] = torch.mean(
                self.episode_reward_sums[key] / self.episode_length_buf.clip(min=1)
            )
            self.episode_reward_sums[key][env_ids] = 0.0

        # reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Reset state
        self.pos_state_buf[env_ids] = 14.0 * (
            torch.rand((len(env_ids), 2), device=self.device, dtype=torch.float) - 0.5
        )
        self.vel_state_buf[env_ids] = torch.zeros(
            (len(env_ids), 2), dtype=torch.float, device=self.device
        )

        if self.cfg.send_timeouts:
            self.extras["time_outs"] = self.truncated_buf.clone()

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.random_init_episode_length:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        clip_actions = self.cfg.clip_actions
        actions_clipped = torch.clip(actions, -clip_actions, clip_actions).to(
            self.device
        )

        # Update state
        self.pos_state_buf += (self.vel_state_buf) * self.dt
        self.vel_state_buf += actions_clipped * self.dt

        reward_buf = self.compute_reward(actions)

        self.episode_length_buf += 1

        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # Update observation (state)
        self.obs_buf[:] = torch.cat((self.pos_state_buf, self.vel_state_buf), dim=-1)

        obs_dict = {
            "critic_observations": self.obs_buf.clone(),
            "actor_observations": self.obs_buf.clone(),
        }

        return obs_dict, reward_buf, self.reset_buf.clone(), self.extras

    def compute_reward(self, actions):
        reward_buf = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i](actions) * self.reward_scales[name]
            reward_buf += rew
            self.episode_reward_sums[name] += rew
        if self.cfg.only_positive_rewards:
            reward_buf = torch.clip(reward_buf, min=0.0)
        return reward_buf

    def check_termination(self):
        # Check for wall collisions
        if self.check_collide_wall_flag:
            self.reset_buf[:] = self.check_collide_wall(
                torch.arange(self.num_envs, device=self.device)
            )

        # Check for timeouts
        self.truncated_buf[:] = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.truncated_buf

    def check_collide_wall(self, env_ids):
        # Check if agent is outside the environment bounds
        flag_x = torch.logical_or(
            self.pos_state_buf[env_ids, 0] < self.xlim[0],
            self.pos_state_buf[env_ids, 0] > self.xlim[1],
        )
        flag_y = torch.logical_or(
            self.pos_state_buf[env_ids, 1] < self.ylim[0],
            self.pos_state_buf[env_ids, 1] > self.ylim[1],
        )
        return torch.logical_or(flag_x, flag_y)

    def reward_distance_cost(self, actions) -> torch.Tensor:
        cur_positions = self.pos_state_buf.unsqueeze(1)  # [batch_size, 1, 2]
        goal_positions = self.goal_positions.unsqueeze(0)  # [1, num_goals, 2]
        dists_to_goals = torch.sum(
            (cur_positions - goal_positions) ** 2, dim=2
        )  # [batch_size, num_goals]
        min_dists = torch.min(dists_to_goals, dim=1)[0]  # [batch_size]
        return torch.exp(-0.3 * min_dists) + torch.exp(-0.1 * min_dists)

    def reward_action_cost(self, actions) -> torch.Tensor:
        return torch.sum(actions**2, dim=1)

    def draw_path_only(self, ax, obs_list, resets_list, legend=False):
        obs_list = torch.stack(obs_list, dim=0).cpu().numpy()  # [T, N, 2]
        resets_list = torch.stack(resets_list, dim=0).cpu().numpy()  # [T, N]

        for i in range(self.num_envs):
            # Find first reset for this agent
            first_reset_idx = np.where(resets_list[:, i])[0]
            if len(first_reset_idx) > 0:
                first_reset_idx = first_reset_idx[0]
            else:
                first_reset_idx = len(resets_list)
            ax.plot(
                obs_list[: first_reset_idx, i, 0],
                obs_list[: first_reset_idx, i, 1],
                linestyle="-",
                color=f"C{i}",
                label=f"agent {i}",
                linewidth=0.5,
                zorder=1,
            )
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        # Plot goal points
        goal_pos = self.goal_positions.cpu().numpy()
        ax.scatter(
            goal_pos[:, 0],
            goal_pos[:, 1],
            color="w",
            marker="o",
            s=250,
            zorder=1,
            edgecolors="k",
        )
        goal = ax.scatter(
            goal_pos[:, 0],
            goal_pos[:, 1],
            color="darkturquoise",
            marker="*",
            s=150,
            label=f"goal",
            zorder=2,
        )
        start = ax.scatter(
            0.0,
            0.0,
            color="w",
            edgecolors="k",
            marker="o",
            s=150,
            label=f"start",
            zorder=2,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if legend:
            ax.legend(handles=[goal, start], ncol=2, loc="upper right")

    def evaluate(
        self, obs_list, actions_list, resets_list, value_model
    ):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams.update({"font.size": 12})
        # plot
        obs_list = torch.stack(obs_list, dim=0).cpu().numpy()  # [T, N, 2]
        actions_list = torch.stack(actions_list, dim=0).cpu().numpy()  # [T, N, 2]
        resets_list = torch.stack(resets_list, dim=0).cpu().numpy()  # [T, N]
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(121)
        for i in range(self.num_envs):
            # Find first reset for this agent
            first_reset_idx = np.where(resets_list[:, i])[0]
            if len(first_reset_idx) > 0:
                first_reset_idx = first_reset_idx[0]
            else:
                first_reset_idx = len(resets_list)
            ax1.plot(
                obs_list[: first_reset_idx, i, 0],
                obs_list[: first_reset_idx, i, 1],
                linestyle="-",
                color=f"C{i}",
                label=f"agent {i}",
                linewidth=0.5,
                zorder=1,
            )

        ax1.set_xlim(self.xlim)
        ax1.set_ylim(self.ylim)
        # Plot goal points
        goal_pos = self.goal_positions.cpu().numpy()
        ax1.scatter(
            goal_pos[:, 0],
            goal_pos[:, 1],
            color="w",
            marker="o",
            s=250,
            zorder=1,
            edgecolors="k",
        )
        goal = ax1.scatter(
            goal_pos[:, 0],
            goal_pos[:, 1],
            color="darkturquoise",
            marker="*",
            s=150,
            label=f"goal",
            zorder=2,
        )
        start = ax1.scatter(
            0.0,
            0.0,
            color="w",
            edgecolors="k",
            marker="o",
            s=150,
            label=f"start",
            zorder=2,
        )
        ax1.set_title("Path")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend(handles=[goal, start], ncol=2)

        # plot state action pairs
        actions = actions_list[0]
        ax2 = fig.add_subplot(122)
        ax2.set_xlim((-8, 8))
        ax2.set_ylim((-8, 8))
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Fisrt-Step Action")
        sampled_actions = ax2.scatter(
            actions[:, 0],
            actions[:, 1],
            marker="*",
            color="blue",
            s=1.5,
            label="Action Samples",
        )
        ax2.legend(handles=[sampled_actions])

        fig.tight_layout()
        plt.show()
