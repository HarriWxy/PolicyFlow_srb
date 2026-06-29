import torch

# @torch.compile
def dopri5_solve(f, y0, t_range=(0.0, 1.0), atol=1e-5, rtol=1e-5):
    """
    Dormand-Prince (DOPRI5) 自适应 ODE 求解器实现
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

    # 初始化 k1 (FSAL: First Same As Last)
    k1 = f(t, y)

    while t < t1:
        # 防止最后一步超出 t1
        if t + h > t1:
            h = t1 - t
        
        # 1. 计算 7 个阶段 (Stages k2 ~ k7)
        k = [None] * 7
        k[0] = k1
        for i in range(1, 7):
            # 计算当前阶段的 y 值: y + h * sum(a_ij * kj)
            y_stage = y + h * sum(a[i][j] * k[j] for j in range(i))
            k[i] = f(t + a[i][0]*h if i < len(a[i]) else t, y_stage) # 简化处理时间点

        # 2. 计算 5 阶预测值 y5 和局部误差 e
        y5 = y + h * sum(b[i] * k[i] for i in range(7))
        e = h * sum((b[i] - b_star[i]) * k[i] for i in range(7))

        # 3. 误差归一化 (对应论文公式 10) [[quote:Q_0ivobvs]]
        # err = sqrt(mean((e / (atol + max(|y|, |y5|) * rtol))^2))
        denom = atol + torch.max(torch.abs(y), torch.abs(y5)) * rtol
        err = torch.sqrt(torch.mean((e / denom)**2))

        # 4. 步长接受与状态更新 (Accept/Reject) [[quote:Q_1xts1fa]]
        if err <= 1.0:
            y = y5
            t = t + h
            k1 = k[6]  # FSAL 优化: 当前步的 k7 是下一步的 k1 [[quote:Q_0ivobvs]]
        
        # 5. 动态调整步长 (Step Size Adjustment) [[quote:Q_1xts1fa]]
        # h_new = h * min(alpha_max, max(alpha_min, 0.9 * err^(-1/6)))
        if err == 0: # 防止除零
            h_factor = alpha_max
        else:
            h_factor = 0.9 * (err**(-1/6))
        
        h = h * min(alpha_max, max(alpha_min, h_factor))

    return y


if __name__ == "__main__":
    # 测试 DOPRI5 求解器
    def test_ode(t, y):
        return -1 * y  #+ torch.sin(torch.tensor([t],dtype=torch.float32,device=y.device))  # 简单的 ODE: dy/dt = -2y + sin(t)

    y0 = torch.tensor([1.0]).to("cuda")  # 初始条件
    t_range = (0.0, 5.0)
    result = dopri5_solve(test_ode, y0, t_range)
    print(f"Result at t={t_range[1]}: {result}")