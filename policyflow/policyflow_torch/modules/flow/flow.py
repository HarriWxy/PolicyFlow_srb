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
    FlowMlp
)
from .utils import at_least_ndim, SAMPLING_STEP_SCHEDULE


class ContinuousNormalizingFlow:
    def __init__(
        self,
        x_dims,
        # ----------------- Neural Networks ----------------- #
        nn_flow: FlowNetBase | FlowMlp,
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
        self.interpolation_type = interpolation_type # rectified_flow

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
            self.sample_step_schedule = sample_step_schedule( # uniform_continuous
                [0.0, final_t], self.sample_steps
            )
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        # 构建训练用的时间步采样表（含中点，排除边界 t=0 和 t=final_t 以避免数值不稳定）
        time_steps = []
        for i in range(self.sample_steps):
            t = self.sample_step_schedule[i]
            delta_t = self.sample_step_schedule[i + 1] - self.sample_step_schedule[i]
            # 跳过 t=0（第一个区间的起点），只保留中点
            if i > 0:
                time_steps.append(t)
            time_steps.append(t + delta_t / 2)
        # 跳过 t=final_t（最后一个区间的终点）
        self.time_steps_tensor = torch.tensor(
            time_steps, dtype=torch.float32, device=self.device
        )
        print(f"time_steps_tensor: {self.time_steps_tensor}")

        # nn_condition is None means that the model is not conditioned on any input.
        if nn_condition is None: # conditionlinearlayer
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
    @torch.inference_mode()
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

        # 预分配所有时间步的 t 和 delta_t 张量，避免循环内反复分配
        schedule = self.sample_step_schedule
        t_all = schedule[:-1].unsqueeze(1).expand(-1, n_samples).to(self.device)  # (steps, n_samples)
        dt_all = (schedule[1:] - schedule[:-1]).to(self.device)                    # (steps,)

        for i in range(self.sample_steps):
            t = t_all[i]
            delta_t = dt_all[i]
            # 二阶龙格-库塔法（Runge-Kutta 2nd Order, RK2）
            vel_t = model["flow"](xt, t, condition_embeded)  # flow.flow_net FlowMlp(FlowNetBase)
            xt_middle = xt + vel_t * delta_t / 2
            vel_t = model["flow"](xt_middle, t + delta_t / 2, condition_embeded)
            xt = xt + delta_t * vel_t

        std = torch.ones_like(xt) * model["variance"].std
        return xt, std
    
    @torch.inference_mode()
    def sample_dor(self,
        x0: torch.Tensor,
        condition: torch.Tensor,
        # ----------------- sampling ----------------- #
        n_samples: int = 1,
    ):
        x0 = x0.to(self.device)

        model = self.model if not self.using_ema else self.model_ema

        # 预先计算 condition embedding，作为闭包捕获
        condition_embeded = model["condition"](condition)

        def velocity_fn(t_scalar, xt):
            """将标量 t 和状态 xt 转换为 flow 网络所需的输入格式"""
            t_tensor = torch.full(
                (n_samples,),
                t_scalar,
                dtype=torch.float32,
                device=self.device,
            )
            # flowmodel = torch.compile(model["flow"])
            return model["flow"](xt, t_tensor, condition_embeded)

        t_start = self.sample_step_schedule[0].item()
        t_end = self.sample_step_schedule[-1].item()

        # 使用 DOPRI5 自适应 ODE 求解器
        xt = dopri5_solve(velocity_fn, x0, t_range=(t_start, t_end))

        std = torch.ones_like(xt) * model["variance"].std
        return xt, std

def dopri5_solve(f, y0, t_range=(0.0, 1.0), atol=1e-5, rtol=1e-5):
    """
    Dormand-Prince (DOPRI5) 自适应 ODE 求解器实现（带 NaN 安全保护）
    """
    t0, t1 = t_range
    y = y0
    t = t0
    h = 0.01  # 初始步长估计值
    
    # DOPRI5 Butcher Tableau 系数 (标准系数)
    # a: 阶段权重, b: 5阶结果权重, b_star: 4阶误差估算权重
    a = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ]
    b = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
    b_star = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
    
    alpha_max = 5.0   # 最大步长增长倍数
    alpha_min = 0.2   # 最小步长缩减倍数
    max_iter = 10000  # 最大迭代次数防止死循环

    # 初始化 k1 (FSAL: First Same As Last)
    k1 = f(t, y)

    # NaN/Inf 检测：如果初始评估就有问题，直接返回
    if torch.isnan(k1).any() or torch.isinf(k1).any():
        return y

    iteration = 0
    while t < t1 and iteration < max_iter:
        iteration += 1
        # 防止最后一步超出 t1
        if t + h > t1:
            h = t1 - t
        
        # 1. 计算 7 个阶段 (Stages k2 ~ k7)
        k = [None] * 7
        k[0] = k1
        nan_in_stages = False
        for i in range(1, 7):
            # 计算当前阶段的 y 值: y + h * sum(a_ij * kj)
            y_stage = y + h * sum(a[i][j] * k[j] for j in range(i))
            k[i] = f(t + a[i][0]*h if i < len(a[i]) else t, y_stage) # 简化处理时间点
            # NaN/Inf 检测
            if torch.isnan(k[i]).any() or torch.isinf(k[i]).any():
                nan_in_stages = True
                break
        
        if nan_in_stages:
            # 缩小步长重试，而不是返回 NaN
            h = h * alpha_min
            if h < 1e-10:
                break  # 步长已经太小，放弃
            continue

        # 2. 计算 5 阶预测值 y5 和局部误差 e
        y5 = y + h * sum(b[i] * k[i] for i in range(7))
        e = h * sum((b[i] - b_star[i]) * k[i] for i in range(7))

        # NaN/Inf 检测
        if torch.isnan(y5).any() or torch.isinf(y5).any():
            h = h * alpha_min
            if h < 1e-10:
                break
            continue

        # 3. 误差归一化 (对应论文公式 10) [[quote:Q_0ivobvs]]
        # err = sqrt(mean((e / (atol + max(|y|, |y5|) * rtol))^2))
        denom = atol + torch.max(torch.abs(y), torch.abs(y5)) * rtol
        err = torch.sqrt(torch.mean((e / denom)**2))

        # NaN/Inf 检测
        if torch.isnan(err) or torch.isinf(err):
            h = h * alpha_min
            if h < 1e-10:
                break
            continue

        # 4. 步长接受与状态更新 (Accept/Reject) [[quote:Q_1xts1fa]]
        if err <= 1.0:
            y = y5
            t = t + h
            k1 = k[6]  # FSAL 优化: 当前步的 k7 是下一步的 k1 [[quote:Q_0ivobvs]]
        
        # 5. 动态调整步长 (Step Size Adjustment) [[quote:Q_1xts1fa]]
        # h_new = h * min(alpha_max, max(alpha_min, 0.9 * err^(-1/6)))
        err_val = err.item()  # 转为 Python float 避免 tensor 比较问题
        if err_val == 0: # 防止除零
            h_factor = alpha_max
        else:
            h_factor = 0.9 * (err_val**(-1/6))
        
        h = h * min(alpha_max, max(alpha_min, h_factor))
        # 限制步长范围防止数值问题
        h = max(h, 1e-8)

    return y