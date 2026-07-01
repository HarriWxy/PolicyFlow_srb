from __future__ import annotations
import itertools
import torch
from typing import Any, Mapping, Optional, Union, Dict, Tuple, Callable
from policyflow_torch.modules import Network, GaussianNetwork
from policyflow_torch.agents import ActorCriticBase
from policyflow_torch.storage import ReplayBuffer


class Exo(ActorCriticBase):
    """Exo agent: Actor-Critic with exponential soft-clipped surrogate + KL regularization.

    Key differences from PPO:
    - Exponential soft-clipping of policy ratio (controlled by beta)
    - KL divergence regularization between old and new policy
    - Huber loss for value function
    """

    def __init__(
        self,
        models: Mapping[str, Network],
        replay_buffer: ReplayBuffer,
        cfg: dict = dict(),
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:

        super().__init__(models, replay_buffer, cfg, device)

        self._learning_rate = cfg.get("learning_rate", 2e-4)
        self._discount_factor = cfg.get("discount_factor", 0.99)
        self._lambda = cfg.get("lam", 0.95)
        self._time_limit_bootstrap = cfg.get("time_limit_bootstrap", True)
        self._mini_batches = cfg.get("mini_batches", 4)
        self._learning_epochs = cfg.get("learning_epochs", 5)
        self._beta = cfg.get("beta", 5.0)
        self._kl_coeff = cfg.get("kl_coeff", 0.8)
        self._entropy_loss_scale = cfg.get("entropy_loss_scale", 0.01)
        self._clip_range = cfg.get("clip_range", 0.2)
        self._value_loss_scale = cfg.get("value_loss_scale", 1.0)
        self._grad_norm_clip = cfg.get("grad_norm_clip", 1.0)

        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.model_dict["actor"].parameters(),
                self.model_dict["critic"].parameters(),
            ),
            lr=self._learning_rate,
        )
        self._register_serializable("optimizer")

    def init_replay_buffer(
        self,
        critic_observation_size: Union[int, Tuple[int]],
        actor_observation_size: Union[int, Tuple[int]],
        action_size: Union[int, Tuple[int]],
    ) -> None:
        """Initialize the agent replay buffer tensors."""
        self.eval_mode()
        self._action_size = action_size
        if self.replay_buffer is not None:
            self.replay_buffer.create_tensor(
                name="critic_observations",
                size=critic_observation_size,
                dtype=torch.float32,
            )
            self.replay_buffer.create_tensor(
                name="actor_observations",
                size=actor_observation_size,
                dtype=torch.float32,
            )
            self.replay_buffer.create_tensor(
                name="next_critic_observations",
                size=critic_observation_size,
                dtype=torch.float32,
            )
            self.replay_buffer.create_tensor(
                name="actions", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="actions_std", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="actions_mean", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="rewards", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="terminated", size=1, dtype=torch.bool
            )
            self.replay_buffer.create_tensor(
                name="actions_log_prob", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(name="values", size=1, dtype=torch.float32)
            self.replay_buffer.create_tensor(
                name="returns", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="advantages", size=1, dtype=torch.float32
            )

    def draw_actions(
        self, observations_dict: Dict[str, torch.Tensor], env_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], None]]:

        mean, std = self.model_dict["actor"].forward(
            observations_dict["actor_observations"],
            compute_std=True,
        )
        action_distribution = torch.distributions.Normal(mean, std)
        actions = action_distribution.sample().detach()
        actions_logp = action_distribution.log_prob(actions).sum(-1)

        info = {
            "actions_log_prob": actions_logp.detach(),
            "actions_mean": mean.detach(),
            "actions_std": std.detach(),
        }

        return actions, info

    def process_transition(
        self,
        observations_dict: Dict[str, torch.Tensor],
        environement_info: Dict[str, Any],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations_dict: Dict[str, torch.Tensor],
        dones: torch.Tensor,
        actions_info: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if self.replay_buffer is not None:
            values = self.model_dict["critic"](
                observations_dict["critic_observations"].flatten(start_dim=1)
            )

            # time-limit (truncation) bootstrapping
            truncated = environement_info.get("time_outs", torch.zeros_like(dones))
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            self.replay_buffer.add_samples(
                critic_observations=observations_dict["critic_observations"],
                next_critic_observations=next_observations_dict["critic_observations"],
                actor_observations=observations_dict["actor_observations"],
                actions_log_prob=actions_info["actions_log_prob"],
                actions_mean=actions_info["actions_mean"],
                actions_std=actions_info["actions_std"],
                actions=actions,
                rewards=rewards,
                terminated=dones,
                values=values,
            )

    def compute_gae(self) -> None:
        """Compute the Generalized Advantage Estimator (GAE)."""
        rewards = self.replay_buffer.get_tensor_by_name("rewards")
        values = self.replay_buffer.get_tensor_by_name("values")
        dones = self.replay_buffer.get_tensor_by_name("terminated")
        next_critic_observations = self.replay_buffer.get_tensor_by_name(
            "next_critic_observations"
        )[-1]

        with torch.inference_mode():
            last_values = self.model_dict["critic"](
                next_critic_observations.flatten(start_dim=1)
            ).detach()

        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        memory_size = rewards.shape[0]

        for i in reversed(range(memory_size)):
            next_values = values[i + 1] if i < memory_size - 1 else last_values
            advantage = (
                rewards[i]
                - values[i]
                + self._discount_factor
                * not_dones[i]
                * (next_values + self._lambda * advantage)
            )
            advantages[i] = advantage

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.replay_buffer.set_tensor_by_name("returns", returns)
        self.replay_buffer.set_tensor_by_name("advantages", advantages)

    def _exp_clip_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        """Exponential soft-clipping of policy ratio (Exo algorithm).

        Uses exponential functions to smoothly clip the ratio instead of hard clipping.
        Controlled by beta (sharpness) and clip_range (center offset).

        Args:
            ratio: The probability ratio pi_new / pi_old

        Returns:
            Clipped ratio
        """
        dq0 = 1.0
        dq1 = ratio
        cr = self._clip_range
        beta = self._beta

        # Clip upper bound with exponential decay
        dq2 = torch.where(
            dq1 > cr + dq0,
            cr + dq0 + 1.0 / beta - torch.exp(beta * (cr + dq0 - dq1)) / beta,
            dq1,
        )
        # Clip lower bound with exponential growth
        dq2 = torch.where(
            dq2 < -cr + dq0,
            -cr + dq0 - 1.0 / beta + torch.exp(beta * (dq2 - (dq0 - cr))) / beta,
            dq2,
        )
        return dq2

    def update(self) -> Dict[str, Union[float, torch.Tensor]]:
        self.compute_gae()
        self.train_mode()
        info = self.update_actor_critic()
        self.eval_mode()
        self.replay_buffer.reset()
        return info

    def update_actor_critic(self) -> Dict[str, Union[float, torch.Tensor]]:

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_kl_loss = 0

        for _ in range(self._learning_epochs):
            sampled_batches = self.replay_buffer.sample_all(
                names=[
                    "critic_observations",
                    "actor_observations",
                    "actions",
                    "actions_mean",
                    "actions_std",
                    "actions_log_prob",
                    "values",
                    "returns",
                    "advantages",
                ],
                mini_batches=self._mini_batches,
            )

            for (
                sampled_critic_observations,
                sampled_actor_observations,
                sampled_actions,
                sampled_actions_mean,
                sampled_actions_std,
                sampled_actions_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:
                # --- Actor update ---
                mean, std = self.model_dict["actor"].forward(
                    sampled_actor_observations,
                    compute_std=True,
                )
                action_distribution = torch.distributions.Normal(mean, std)
                actions_log_prob_new = action_distribution.log_prob(
                    sampled_actions
                ).sum(-1)

                # Policy ratio
                ratio = torch.exp(actions_log_prob_new - sampled_actions_log_prob)

                # Exponential soft-clipped surrogate (Exo algorithm)
                clipped_ratio = self._exp_clip_ratio(ratio)
                Q = -torch.min(
                    ratio * sampled_advantages,
                    clipped_ratio * sampled_advantages,
                )

                # KL divergence between old policy and new policy
                old_dist = torch.distributions.Independent(
                    torch.distributions.Normal(sampled_actions_mean, sampled_actions_std),
                    reinterpreted_batch_ndims=1,
                )
                new_dist = torch.distributions.Independent(
                    action_distribution,
                    reinterpreted_batch_ndims=1,
                )
                kl_loss = self._kl_coeff * torch.distributions.kl_divergence(
                    old_dist, new_dist
                ).mean()

                # Entropy bonus
                if self._entropy_loss_scale:
                    entropy_loss = (
                        -self._entropy_loss_scale
                        * action_distribution.entropy().sum(dim=-1).mean()
                    )
                else:
                    entropy_loss = 0.0

                policy_loss = Q.mean() + kl_loss + entropy_loss

                # --- Critic update (Huber loss) ---
                predicted_values = self.model_dict["critic"](
                    sampled_critic_observations
                )
                value_loss = self._value_loss_scale * torch.nn.functional.huber_loss(
                    sampled_returns, predicted_values
                )

                # --- Optimization step ---
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward()

                if self._grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.model_dict["critic"].parameters(),
                            self.model_dict["actor"].parameters(),
                        ),
                        self._grad_norm_clip,
                    )
                self.optimizer.step()

                cumulative_policy_loss += Q.mean().item()
                cumulative_value_loss += value_loss.item()
                cumulative_kl_loss += kl_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

        n_updates = self._learning_epochs * self._mini_batches
        output = {
            "Loss/policy_loss": cumulative_policy_loss / n_updates,
            "Loss/entropy_loss": cumulative_entropy_loss / n_updates,
            "Loss/value_loss": cumulative_value_loss / n_updates,
            "Loss/kl_loss": cumulative_kl_loss / n_updates,
            "Policy/policy_std": std.mean().item(),
            "Loss/learning_rate": self._learning_rate,
        }

        return output

    def eval_mode(self):
        self.set_mode("eval")

    def train_mode(self):
        self.set_mode("train")

    def to(self, device: str) -> Exo:
        for model in self.model_dict.values():
            if model is not None:
                model.to(device)
        return self

    def export_onnx(self) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
        dummy_input = torch.randn(1, self.model_dict["actor"].input_size).to(self.device)
        return self.model_dict["actor"], (dummy_input,), {}

    def get_inference_policy(self, device=None):
        def policy(obs_dict):
            mean, _ = self.model_dict["actor"].forward(
                obs_dict["actor_observations"],
                compute_std=True,
            )
            return mean

        return policy
