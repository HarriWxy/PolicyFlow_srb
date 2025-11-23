from typing import Optional, Union, Callable

import torch
import torch.nn as nn
import copy
import math

from .flow_net import (
    ConditionNetBase,
    IdentityCondition,
    FlowNetBase,
    LearnableVariance,
)
from .utils import at_least_ndim, SAMPLING_STEP_SCHEDULE


class ContinuousNormalizingFlow:
    def __init__(
        self,
        x_dims,
        # ----------------- Neural Networks ----------------- #
        nn_flow: FlowNetBase,
        nn_condition: Optional[ConditionNetBase] = None,
        # ------------------ Training Params ---------------- #
        ema_rate: float = 0.995,
        using_ema: bool = False,
        sample_steps: int = 10,
        sample_step_schedule: Union[str, Callable] = "uniform_continuous",
        interpolation_type: str = "rectified_flow",  # stochastic_interpolant, trigflow, rectified_flow
        device: Union[torch.device, str] = "cpu",
    ):
        self.device = device
        self.ema_rate = ema_rate
        self.using_ema = using_ema
        self.sample_steps = sample_steps
        self.interpolation_type = interpolation_type

        # ===================== Sampling Schedule ====================
        if (
            interpolation_type == "stochastic_interpolant"
            or interpolation_type == "rectified_flow"
        ):
            final_t = 1.0
        elif interpolation_type == "trigflow":
            final_t = math.pi / 2.0
        else:
            raise ValueError(
                f"Interpolation type {interpolation_type} is not supported."
            )
        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SAMPLING_STEP_SCHEDULE.keys():
                self.sample_step_schedule = SAMPLING_STEP_SCHEDULE[
                    sample_step_schedule
                ]([0.0, final_t], self.sample_steps)
            else:
                raise ValueError(
                    f"Sampling step schedule {sample_step_schedule} is not supported."
                )
        elif callable(sample_step_schedule):
            self.sample_step_schedule = sample_step_schedule(
                [0.0, final_t], self.sample_steps
            )
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        time_steps = []
        for i in range(self.sample_steps):
            t = self.sample_step_schedule[i]
            time_steps.append(t)
            delta_t = self.sample_step_schedule[i + 1] - self.sample_step_schedule[i]
            time_steps.append(t + delta_t / 2)
        time_steps.append(self.sample_step_schedule[self.sample_steps])
        self.time_steps_tensor = torch.tensor(
            time_steps, dtype=torch.float32, device=self.device
        )
        print(f"time_steps_tensor: {self.time_steps_tensor}")

        # nn_condition is None means that the model is not conditioned on any input.
        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = nn.ModuleDict(
            {
                "flow": nn_flow.to(self.device),
                "condition": nn_condition.to(self.device),
                "variance": LearnableVariance(dims=x_dims).to(self.device),
            },
        )
        self.model_ema = copy.deepcopy(self.model).requires_grad_(False)
        self.model_last = copy.deepcopy(self.model).requires_grad_(False)

        self.model.train()
        self.model_ema.eval()
        self.model_last.eval()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)

    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_ema": self.model_ema.state_dict(),
                "model_last": self.model_last.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
        self.model_last.load_state_dict(checkpoint["model_last"])

    def compute_flow_variation(
        self, x1, condition, x0=None, compute_brownian_reg_loss=False
    ):
        # x0 is the samples of source distribution.
        # x0 x1 is None, then we assume x1 is from a standard Gaussian distribution.
        if x0 is None:
            x0 = torch.randn_like(x1)
        else:
            assert x0.shape == x1.shape, "x0 and x1 must have the same shape"

        # t = torch.rand((x1.shape[0],), device=self.device)

        idx = torch.randint(
            low=0,
            high=self.time_steps_tensor.shape[0],
            size=(x1.shape[0],),
            device=self.device,
        )
        t = self.time_steps_tensor[idx]  # shape: (batch_size,)

        alpha = at_least_ndim(t, x1.dim())

        if self.interpolation_type == "rectified_flow":
            xt = (1.0 - alpha) * x0 + alpha * x1
        elif self.interpolation_type == "stochastic_interpolant":
            xt = (
                (1.0 - alpha) * x0
                + alpha * x1
                + torch.sqrt(2.0 * alpha * (1.0 - alpha).clip(min=1e-6))
                * torch.randn_like(x1)
            )
        elif self.interpolation_type == "trigflow":
            xt = torch.cos(alpha) * x0 + torch.sin(alpha) * x1
        else:
            raise ValueError(
                f"Interpolation type {self.interpolation_type} is not supported."
            )

        with torch.inference_mode():
            condition_embeded_last = self.model_last["condition"](condition)
            vel_field_last = self.model_last["flow"](
                xt, t, condition_embeded_last
            ).detach()

        condition_embeded = self.model["condition"](condition)
        vel_field = self.model["flow"](xt, t, condition_embeded)

        delta_vel = vel_field - vel_field_last
        std = torch.ones_like(x1) * self.model["variance"].std

        if compute_brownian_reg_loss:
            beta = 1.0 # temperature coeff
            if self.interpolation_type == "rectified_flow":
                brownian_reg_loss = torch.nn.functional.mse_loss(
                    (1 - alpha) * vel_field, beta * (xt - alpha * vel_field_last)
                )
            elif self.interpolation_type == "stochastic_interpolant":
                brownian_reg_loss = torch.nn.functional.mse_loss(
                    (2.0 * ((alpha - 0.5) ** 2) + 0.5) * vel_field,
                    beta * (xt - alpha * vel_field_last),
                )
            elif self.interpolation_type == "trigflow":
                brownian_reg_loss = torch.nn.functional.mse_loss(
                    torch.cos(alpha) * vel_field,
                    beta * (torch.cos(alpha) * xt - torch.sin(alpha) * vel_field_last),
                )
            return delta_vel, std, brownian_reg_loss
        else:
            return delta_vel, std

    def update(self):
        if self.using_ema:
            for name, param in self.model_ema.named_parameters():
                if name in dict(self.model_last.named_parameters()):
                    target_param = dict(self.model_last.named_parameters())[name]
                    target_param.data.copy_(param.data)
        else:
            for name, param in self.model.named_parameters():
                if name in dict(self.model_last.named_parameters()):
                    target_param = dict(self.model_last.named_parameters())[name]
                    target_param.data.copy_(param.data)

    # ==================== Sampling: Solving a straight ODE flow ======================
    def sample(
        self,
        x0: torch.Tensor,
        condition: torch.Tensor,
        # ----------------- sampling ----------------- #
        n_samples: int = 1,
    ):
        x0 = x0.to(self.device)

        model = self.model if not self.using_ema else self.model_ema

        xt = x0.clone()
        condition_embeded = model["condition"](condition)

        for i in range(self.sample_steps):
            t = torch.full(
                (n_samples,),
                self.sample_step_schedule[i],
                dtype=torch.float32,
                device=self.device,
            )

            delta_t = self.sample_step_schedule[i + 1] - self.sample_step_schedule[i]
            vel_t = model["flow"](xt, t, condition_embeded)
            xt_middle = xt + vel_t * delta_t / 2
            vel_t = model["flow"](xt_middle, t + delta_t / 2, condition_embeded)
            xt = xt + delta_t * vel_t

        std = torch.ones_like(xt) * model["variance"].std
        return xt.detach(), std.detach()
