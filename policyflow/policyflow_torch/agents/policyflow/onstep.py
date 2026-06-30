

import torch
import torch.nn as nn

class OneStepFlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 模型输入: 当前状态 z_t, 起始时间 t, 目标时间 r, 观测 o
        # 输出: 区间平均速度 u (velocity)
        self.net = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))

    def forward(self, z_t, t, r, o):
        # u_theta(z_t, t, r | o)
        return self.net(torch.cat([z_t, t, r, o], dim=-1))

class OFPTrainer:
    def __init__(self, model, ema_model):
        self.model = model
        self.ema_model = ema_model # 用于 Self-Guided 的教师模型

    def compute_loss(self, o, a_expert):
        # 1. 基础采样: 随机时间 t, m, r (t < m < r) 和 噪声 epsilon
        t, m, r = sample_time_schedule() 
        epsilon = torch.randn_like(a_expert)
        z_t = interpolate(epsilon, a_expert, t)

        # --- 损失 A: Boundary Anchoring (基础 Flow Matching) ---
        # 确保模型知道如何从噪声走到目标，维持多步生成能力
        u_pred = self.model(z_t, t, r, o)
        loss_flow = torch.mean((u_pred - (a_expert - epsilon))**2)

        # --- 损失 B: Self-Consistency Training (核心：压缩路径) ---
        # 原理: 从 t 直接跳到 r 的结果，应该等于 先跳到 m 再跳到 r 的结果
        # z_r = z_t + (r - t) * u(z_t, t, r)
        with torch.no_grad():
            z_m = z_t + (m - t) * self.model(z_t, t, m, o)
            u_target = (a_expert - z_m) / (r - m) # 目标速度
        
        loss_sc = torch.mean((self.model(z_m, m, r, o) - u_target)**2)

        # --- 损失 C: Self-Guided Regularization (核心：锐化预测) ---
        # 利用 CFG (Classifier-Free Guidance): 条件预测 vs 无条件预测
        u_cond = self.model(z_t, t, r, o)      # 有观测 o
        u_uncond = self.model(z_t, t, r, None) # 无观测 phi
        
        # 引导信号: 将预测向高密度专家模式推，远离模糊的平均值
        guidance = u_cond + scale * (u_cond - u_uncond)
        loss_sg = torch.mean((u_pred - guidance)**2)

        return loss_flow + lambda1 * loss_sc + lambda2 * loss_sg

    def train_step(self, batch):
        loss = self.compute_loss(batch.o, batch.a)
        loss.backward()
        # ... optimizer step & EMA update ...

# ==========================================
# 推理阶段：这就是 OFP 为什么快的原因 (NFE=1)
# ==========================================
def inference(model, o, prev_action_chunk=None):
    # 1. Warm-Start: 如果有之前的动作块，则加噪初始化；否则用纯噪声
    if prev_action_chunk is not None:
        z_0 = warm_start(prev_action_chunk) 
    else:
        z_0 = torch.randn(action_dim)

    # 2. One-Step Jump: 直接从 t=0 跳到 r=1
    # a = z_0 + (1 - 0) * u_theta(z_0, 0, 1 | o)
    u_one_step = model(z_0, t=0, r=1, o=o)
    action = z_0 + u_one_step
    
    return action
