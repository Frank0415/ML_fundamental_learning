"""
rectified_flow.py — Rectified Flow ODE 推理核心

纯 numpy 实现，不依赖 torch。提供：
- rectified_flow_step: 单步 ODE 更新
- rectified_flow_sample: 完整循环采样

背景：
  Rectified Flow 是一种 flow matching 框架，在 t∈[0,1] 上定义线性插值路径：
      x_t = (1-t)·x_0 + t·x_1
  其中 x_0 来自数据分布 p_data，x_1 来自噪声分布 p_noise（通常 N(0,I)）。
  
  训练目标：学习一个 vector field v_θ(x_t, t) 使得 ODE
      dx_t/dt = v_θ(x_t, t)
  的终态 x_0 服从 p_data。

  推理时，从 x_1 ~ N(0,I) 出发，沿 v_θ 做数值积分回到 x_0。
  由于 ODE 是可逆的，确定性积分意味着：
      - 相同的 x_1 + 相同的 v_θ + 相同的积分器 → 相同的 x_0（可复现）
      - 不需要噪声采样（区别于 DDPM 的随机去噪）

与 score-based 去噪的关系：
  - Score-based: 预测噪声 ε_θ 或 score ∇_x log p_t(x)，反向 SDE/ODE
    依赖 α_t / σ_t 时间调度，公式较复杂。
  - Rectified flow: 直接预测 vector field v_θ = x_1 - x_0，
    将扩散过程重写为两个分布之间的概率传输，ODE 形式简洁。
  - 等价性：在 flow matching 框架下，v_θ 与 ε_θ 可通过缩放互相转换：
      v_θ = ε_θ（当使用 v-prediction 参数化时取特殊缩放）。

与 Euler / Heun / DDIM 的对比：
  - Euler: 最简单的一阶 ODE 求解器，x_{k+1} = x_k + h·f(x_k)。
    步数少时精度低，但计算量小。
  - Heun: 二阶 Runge-Kutta（中点法），比 Euler 精度高，需两次模型前向。
  - DDIM: 专为 DDPM 参数化设计的确定性采样器，等价于积分概率流 ODE，
    与 Euler 在特定 sigma 间隔下数值行为相近。
  - 本模块实现的是最简单的 Euler rectified flow，提供基线 ODE 行为，
    未来可扩展为 Heun / DPM-Solver 等高阶积分器。

  x_t 和 x_1 (noise)、x_0 (data) 之间的互换关系：
      x_1 = (x_t - (1-t)·x_0) / t
      x_0 = (x_t - t·x_1) / (1-t)
  这允许在给定 v_θ 时重建 x_0 或 x_1 估计。
"""

import numpy as np


def rectified_flow_step(x_t, t, t_next, v_theta):
    """
    Rectified flow 单步 Euler ODE 更新。

    公式：
        x_{next} = x_t + (t_next - t) · v_theta(x_t, t)

    这是最简单的 ODE 积分 — 沿向量场方向做一步欧拉推进。
    由于 t 通常递减（从 1→0），dt = t_next - t 通常为负值。

    参数：
        x_t: 当前 latent，shape 任意，dtype 任意（float32/float64）。
        t: 当前时间步（标量），rectified flow 中 t∈[0,1]。
        t_next: 下一步时间（标量），需满足 t_next < t（或 t_next > t 若反向积分）。
        v_theta: 模型预测的 vector field，shape 与 x_t 相同。

    返回：
        x_next: 更新后的 latent，dtype 与 x_t 一致。

    示例：
        >>> import numpy as np
        >>> x_t = np.array([1.0, 2.0])
        >>> v = np.array([3.0, -1.0])  # 常数向量场
        >>> rectified_flow_step(x_t, t=1.0, t_next=0.5, v_theta=v)
        array([2.5, 1.5])  # x_t + (0.5 - 1.0) * v = x_t - 0.5 * v
    """
    dt = float(t_next) - float(t)
    x_next = x_t + dt * v_theta
    return x_next.astype(x_t.dtype)


def rectified_flow_sample(
    v_theta_fn,
    x_T,
    timesteps,
    callback=None,
    seed=None,
):
    """
    完整 rectified flow 采样循环。

    从初始噪声 x_T（t=T_start）出发，按 timesteps 序列递减，
    每步调用 v_theta_fn(x_t, t) 获取 vector field，做 Euler 积分，
    最终到达 t=T_end 的"干净" latent x_0。

    参数：
        v_theta_fn: 可调用对象，签名为 v_theta_fn(x_t, t) → vector_field。
                    其中 x_t 为当前 latent，t 为标量时间步。
                    返回的 vector_field 应与 x_t 同 shape。
        x_T: 初始 latent（噪声），t=timesteps[0] 时的状态。
        timesteps: 时间步序列，从起始（噪声端）到终止（数据端），
                   共 N+1 个点，执行 N 步 ODE 积分。
                   例：timesteps = np.linspace(1.0, 0.0, 29) → 28 步。
        callback: 可选回调，签名为 callback(i, t, t_next, x_t)。
                  在每步更新后调用，用于记录中间状态或调试。
        seed: 确定性随机种子（预留，当前实现中 rectified flow 为确定性 ODE，
              无随机采样，但保留接口用于未来可能的后处理采样器）。

    返回：
        x_0: 最终 latent（t=timesteps[-1] 时的状态），dtype 与 x_T 一致。

    异常：
        ValueError: 若 timesteps 长度不足（需要 ≥ 2 个点）。

    示例：
        >>> import numpy as np
        >>> def v_fn(x, t):
        ...     return -x  # 所有点流向原点
        >>> x_T = np.random.randn(100, 2)
        >>> timesteps = np.linspace(1.0, 0.0, 29)
        >>> x_0 = rectified_flow_sample(v_fn, x_T, timesteps)
        >>> np.allclose(x_0, np.zeros_like(x_0), atol=1e-2)  # 收敛到原点附近
        True
    """
    timesteps = np.asarray(timesteps, dtype=np.float64)
    if len(timesteps) < 2:
        raise ValueError(
            f"timesteps 至少需要 2 个点（起始+终止），收到 {len(timesteps)} 个点"
        )

    # 验证 timesteps 单调性
    diffs = np.diff(timesteps)
    if not (np.all(diffs > 0) or np.all(diffs < 0)):
        raise ValueError("timesteps 必须是单调序列（全递增或全递减）")

    x_t = np.asarray(x_T).copy()
    n_steps = len(timesteps) - 1

    for i in range(n_steps):
        t = float(timesteps[i])
        t_next = float(timesteps[i + 1])

        # 获取 vector field
        v = v_theta_fn(x_t, t)

        # 执行单步 ODE 积分
        x_t = rectified_flow_step(x_t, t, t_next, v)

        if callback is not None:
            callback(i, t, t_next, x_t)

    return x_t.astype(x_T.dtype)
