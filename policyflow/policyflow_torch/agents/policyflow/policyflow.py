from __future__ import annotations
import itertools
import torch
from typing import Any, Mapping, Optional, Union, Dict, Tuple, Callable
from policyflow_torch.modules import Network, ContinuousNormalizingFlow
from policyflow_torch.agents import PolicyFlowBase
from policyflow_torch.storage import ReplayBuffer
from policyflow_torch.utils.kl_adaptive import KLAdaptiveLR


class PolicyFlow(PolicyFlowBase):
    def __init__(
        self,
        models: Mapping[
            str, Union[ContinuousNormalizingFlow | Network | torch.nn.Module]
        ],
        replay_buffer: ReplayBuffer,
        cfg: dict = dict(),
        device: Optional[Union[str, torch.device]] = None,
        optim_params: Optional[dict] = None,
    ) -> None:
        if optim_params is None:
            optim_params = {"weight_decay": 1e-5}

        super().__init__(models, replay_buffer, cfg, device)

        self._desired_kl = cfg.get("desired_kl", 0.01)
        self._learning_rate = cfg.get("learning_rate", 1e-4)
        self._discount_factor = cfg.get("discount_factor", 0.99)
        self._lambda = cfg.get("lam", 0.95)
        self._time_limit_bootstrap = cfg.get("time_limit_bootstrap", True)
        self._mini_batches = cfg.get("mini_batches", 1)
        self._learning_epochs = cfg.get("learning_epochs", 1)
        self._gaussian_entropy_loss_scale = cfg.get("gaussian_entropy_loss_scale", 0.0)
        self._brownian_reg_loss_scale = cfg.get("brownian_reg_loss_scale", 0.0)
        self._ratio_clip = cfg.get("ratio_clip", 0.2)
        self._clip_predicted_values = cfg.get("clip_predicted_values", True)
        self._value_clip = cfg.get("value_clip", 0.1)
        self._value_loss_scale = cfg.get("value_loss_scale", 1.0)
        self._grad_norm_clip = cfg.get("grad_norm_clip", 1.0)
        self._degenerate2gaussian = cfg.get("degenerate2gaussian", False)

        self.optimizer = torch.optim.AdamW(
            itertools.chain(
                self.model_dict["actor"].model.parameters(),
                self.model_dict["critic"].parameters(),
            ),
            lr=self._learning_rate,
            **optim_params,
        )
        self.lr_schedule = KLAdaptiveLR(
            self.optimizer,
            **cfg.get(
                "learning_rate_scheduler_kwargs",
                {
                    "kl_threshold": self._desired_kl,
                },
            ),
        )
        self._register_serializable("optimizer")

    def init_replay_buffer(
        self,
        critic_observation_size: Union[int, Tuple[int]],
        actor_observation_size: Union[int, Tuple[int]],
        action_size: Union[int, Tuple[int]],
    ) -> None:
        """Initialize the agent"""
        self.eval_mode()
        self._action_size = action_size
        self._register_serializable("_action_size")

        # create tensors in replay_buffer
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
                name="actions_prior", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="flow_x0", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="delta_actions", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="delta_actions_std", size=action_size, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="delta_actions_log_prob", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="rewards", size=1, dtype=torch.float32
            )
            self.replay_buffer.create_tensor(
                name="terminated", size=1, dtype=torch.bool
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

        x0 = torch.randn(
            (
                observations_dict["actor_observations"].shape[0],
                self._action_size,
            ),
            device=self.device,
        )
        
        if self._degenerate2gaussian:
            x0 = torch.zeros_like(x0)

        actions_prior, std = self.model_dict["actor"].sample(
            x0=x0,
            condition=observations_dict["actor_observations"],
            n_samples=observations_dict["actor_observations"].shape[0],
        )
        delta_action_distribution = torch.distributions.Normal(
            torch.zeros_like(actions_prior), std
        )
        delta_actions = delta_action_distribution.sample().detach()
        actions = actions_prior.detach() + delta_actions
        delta_actions_logp = delta_action_distribution.log_prob(delta_actions).sum(-1)

        info = {
            "actions_prior": actions_prior.detach(),
            "delta_actions": delta_actions.detach(),
            "delta_actions_std": std.detach(),
            "delta_actions_log_prob": delta_actions_logp.detach(),
            "flow_x0": x0.clone(),
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
            # compute values
            values = self.model_dict["critic"](observations_dict["critic_observations"].flatten(start_dim=1))

            # time-limit (truncation) boostrapping
            truncated = environement_info.get("time_outs", torch.zeros_like(dones))
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in replay_buffer
            self.replay_buffer.add_samples(
                critic_observations=observations_dict["critic_observations"],
                next_critic_observations=next_observations_dict["critic_observations"],
                actor_observations=observations_dict["actor_observations"],
                actions_prior=actions_info["actions_prior"],
                flow_x0=actions_info["flow_x0"],
                delta_actions=actions_info["delta_actions"],
                delta_actions_std=actions_info["delta_actions_std"],
                delta_actions_log_prob=actions_info["delta_actions_log_prob"],
                rewards=rewards,
                terminated=dones,
                values=values,
            )

    def compute_gae(self) -> torch.Tensor:
        """Compute the Generalized Advantage Estimator (GAE)"""
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

        # advantages computation
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
        # returns computation
        returns = advantages + values
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.replay_buffer.set_tensor_by_name("returns", returns)
        self.replay_buffer.set_tensor_by_name("advantages", advantages)

    def update(self) -> Dict[str, Union[float, torch.Tensor]]:
        self.compute_gae()
        self.train_mode()

        cumulative_policy_loss = 0
        cumulative_gaussian_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_brownian_reg_loss = 0
        delta_vel_max = -100.0
        delta_vel_min = 100.0

        kl_divergences = []

        # learning epochs
        for _ in range(self._learning_epochs):

            # sample mini-batches from replay_buffer
            sampled_batches = self.replay_buffer.sample_all(
                names=[
                    "critic_observations",
                    "actor_observations",
                    "flow_x0",
                    "actions_prior",
                    "delta_actions",
                    "delta_actions_std",
                    "delta_actions_log_prob",
                    "values",
                    "returns",
                    "advantages",
                ],
                mini_batches=self._mini_batches,
            )

            # mini-batches loop
            for (
                sampled_critic_observations,
                sampled_actor_observations,
                sampled_flow_x0,
                sampled_actions_prior,
                sampled_delta_actions,
                sampled_delta_actions_std,
                sampled_delta_actions_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:
                if not self._degenerate2gaussian and self._brownian_reg_loss_scale > 0:
                    delta_vel, delta_std_new, brownian_reg_loss = self.model_dict[
                        "actor"
                    ].compute_flow_variation(
                        x1=sampled_actions_prior,
                        condition=sampled_actor_observations,
                        x0=sampled_flow_x0,
                        compute_brownian_reg_loss=True,
                    )
                    if "anneal_coef" in self.model_dict:
                        brownian_reg_loss = (
                            self._brownian_reg_loss_scale
                            * self.model_dict["anneal_coef"].forward()
                            * brownian_reg_loss
                        )
                    else:
                        brownian_reg_loss = (
                            self._brownian_reg_loss_scale * brownian_reg_loss
                        )
                else:
                    delta_vel, delta_std_new = self.model_dict[
                        "actor"
                    ].compute_flow_variation(
                        x1=sampled_actions_prior,
                        condition=sampled_actor_observations,
                        x0=sampled_flow_x0,
                        compute_brownian_reg_loss=False,
                    )
                    brownian_reg_loss = 0

                action_distribution_new = torch.distributions.Normal(
                    delta_vel, delta_std_new
                )
                actions_log_prob_new = action_distribution_new.log_prob(
                    sampled_delta_actions
                ).sum(-1)

                # compute approximate KL divergence
                kl_divergences.append(
                    self._compute_kl_divergence(
                        delta_vel,
                        delta_std_new,
                        torch.zeros_like(delta_vel),
                        sampled_delta_actions_std,
                    )
                )

                # compute entropy loss
                if self._gaussian_entropy_loss_scale:
                    gaussian_entropy_loss = (
                        -self._gaussian_entropy_loss_scale
                        * action_distribution_new.entropy().sum(dim=-1).mean()
                    )
                else:
                    gaussian_entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(actions_log_prob_new - sampled_delta_actions_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(
                    ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                )
                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value regression loss
                predicted_values = self.model_dict["critic"](
                    sampled_critic_observations
                )
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(
                        predicted_values - sampled_values,
                        min=-self._value_clip,
                        max=self._value_clip,
                    )
                value_loss = self._value_loss_scale * torch.nn.functional.mse_loss(
                    sampled_returns, predicted_values
                )

                # optimization step
                self.optimizer.zero_grad()
                (
                    policy_loss
                    + gaussian_entropy_loss
                    + brownian_reg_loss
                    + value_loss
                ).backward()
                if self._grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(
                            self.model_dict["actor"].model.parameters(),
                            self.model_dict["critic"].parameters(),
                        ),
                        self._grad_norm_clip,
                    )
                self.optimizer.step()

                if self.model_dict["actor"].using_ema:
                    self.model_dict["actor"].ema_update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._gaussian_entropy_loss_scale:
                    cumulative_gaussian_entropy_loss += gaussian_entropy_loss.item()
                if self._brownian_reg_loss_scale > 0:
                    cumulative_brownian_reg_loss += brownian_reg_loss.item()

                if torch.max(delta_vel) > delta_vel_max:
                    delta_vel_max = torch.max(delta_vel)
                if torch.min(delta_vel) < delta_vel_min:
                    delta_vel_min = torch.min(delta_vel)

            # update learning rate
            kl = torch.tensor(kl_divergences, device=self.device).mean()
            self.lr_schedule.step(kl.item())

        self.model_dict["actor"].update()

        output = {
            "Loss/policy_loss": cumulative_policy_loss
            / (self._learning_epochs * self._mini_batches),
            "Loss/gaussian_entropy_loss": cumulative_gaussian_entropy_loss
            / (self._learning_epochs * self._mini_batches),
            "Loss/value_loss": cumulative_value_loss
            / (self._learning_epochs * self._mini_batches),
            "Policy/mean_noise_std": delta_std_new.mean().item(),
            "Policy/delta_vel_max": delta_vel_max.item(),
            "Policy/delta_vel_min": delta_vel_min.item(),
            "Loss/learning_rate": self.lr_schedule.get_last_lr()[0],
            "Loss/kl": kl.item(),
        }

        if self._brownian_reg_loss_scale > 0:
            output["Loss/brownian_reg_loss"] = cumulative_brownian_reg_loss / (
                self._learning_epochs * self._mini_batches
            )
            if "anneal_coef" in self.model_dict:
                self.model_dict["anneal_coef"].step_update()
                output["Loss/brownian_reg_anneal_coef"] = (
                    self.model_dict["anneal_coef"].forward().item()
                )

        self.eval_mode()
        self.replay_buffer.reset()

        return output

    def _compute_kl_divergence(
        self,
        actions_mean: torch.Tensor,
        actions_std: torch.Tensor,
        last_action_mean: torch.Tensor,
        last_action_std: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            std_ratio = actions_std / last_action_std + 1.0e-5
            std_drifted = torch.square(last_action_std) + torch.square(
                last_action_mean - actions_mean
            )
            kl = torch.sum(
                torch.log(std_ratio)
                + std_drifted / (2.0 * torch.square(actions_std))
                - 0.5,
                axis=-1,
            )  # type: ignore
            kl_mean = torch.mean(kl).detach()
        return kl_mean

    def eval_mode(self):
        self.set_mode("eval")

    def train_mode(self):
        self.set_mode("train")

    def to(self, device: str) -> PolicyFlowBase:
        for model in self.model_dict.values():
            if model is not None:
                model.to(device)
        return self

    def export_onnx(self) -> Tuple[torch.nn.Module, torch.Tensor, Dict]:
        pass

    def get_inference_policy(self, device: str = None) -> Callable:
        self.eval_mode()
        if device is not None:
            self.to(device)

        def actor_policy(obs_dict):
            x0 = torch.randn(
                (
                    obs_dict["actor_observations"].shape[0],
                    self._action_size,
                ),
                device=self.device,
            )
            if self._degenerate2gaussian:
                x0 = torch.zeros_like(x0)
            mean, std = self.model_dict["actor"].sample(
                x0=x0,
                condition=obs_dict["actor_observations"],
                n_samples=obs_dict["actor_observations"].shape[0],
            )
            action_distribution = torch.distributions.Normal(mean, std)
            actions = action_distribution.sample().detach()
            return actions

        return actor_policy
