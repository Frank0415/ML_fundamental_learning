"""
scheduler.py — 噪声调度器与 ODE 步进

纯 numpy 实现，不依赖 torch。提供两类 scheduler：
- EulerScheduler：在 sigma 空间进行 Euler 法 ODE 积分，支持确定性噪声注入。
- RectifiedFlowScheduler：在 t∈[0,1] 空间进行 rectified flow 风格 ODE 积分。

两类 scheduler 共享相同的 step 公式（Euler 法）：
    x_next = x_t + (t_next - t) * v
区别在于时间空间的参数化：sigma（log-linear 间隔）vs t（线性间隔）。

设计原则：
    1. 纯函数接口，无隐藏状态（仅 self.sigmas/self.timesteps 存储 schedule）。
    2. 单元测试友好：所有方法输出仅依赖输入和确定性 seed。
    3. dtype 提示：输出 dtype 与输入 x_t 一致。
    4. 边界防御：num_steps ≤ 0 抛 ValueError，单调性可在 step 时隐式保证。

与 score-based 去噪的区别：
    - Score-based（DDPM/DDIM）：预测噪声 ε_θ，用 score function ∇_x log p(x)
      进行 Langevin 动力学或概率流 ODE。公式复杂，依赖 α_t/σ_t 缩放。
    - Rectified flow：预测 vector field v_θ = x_1 - x_0，ODE 形式简单
      dx = v_θ dt。参数化在 t∈[0,1]，无需 σ 的指数缩放。
"""

import numpy as np


class EulerScheduler:
    """
    Euler 法 ODE 求解器 — 用于 score-based / DDIM 风格推理。

    在 sigma 空间按 log-linear 间隔执行确定性（或可选随机）更新。
    sigma 从大到小（噪声→干净），与 DDPM 前向加噪过程一致。

    用法：
        scheduler = EulerScheduler(num_steps=50, sigma_min=0.002, sigma_max=80.0)
        x_t = randn(...) * sigma_max  # 初始噪声
        for i in range(len(scheduler.sigmas) - 1):
            v = v_theta(x_t, scheduler.sigmas[i])  # 模型前向
            x_t = scheduler.step(v, x_t, i)
    """

    def __init__(self, sigmas=None, num_steps=None, sigma_min=0.002,
                 sigma_max=80.0, seed=None, stochastic=False):
        """
        参数：
            sigmas: 自定义 sigma 序列（若提供则忽略 num_steps 等参数）。
            num_steps: 采样步数（默认 50）。生成 log-linear sigma 序列。
            sigma_min: 最小 sigma（对应干净 latent）。
            sigma_max: 最大 sigma（对应初始噪声）。
            seed: 确定性随机种子（控制 noise 注入和任何随机操作）。
            stochastic: 是否在每步加噪声（SDE 模式），默认 False（确定性 ODE）。
        """
        if sigmas is not None:
            self.sigmas = np.asarray(sigmas, dtype=np.float64)
            if len(self.sigmas) < 2:
                raise ValueError(f"sigmas 至少需要 2 个元素（起始+终止），收到 {len(self.sigmas)}")
        else:
            n = num_steps if num_steps is not None else 50
            if n <= 0:
                raise ValueError(f"num_steps 必须 > 0，收到 {n}")
            # num_steps = ODE 步数 → 生成 num_steps+1 个 sigma 点
            self.sigmas = self.get_sigmas(n + 1, sigma_min, sigma_max)

        self.seed = seed
        self.stochastic = stochastic
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    @staticmethod
    def get_sigmas(num_steps, sigma_min=0.002, sigma_max=80.0):
        """
        生成 log-linear 间隔的 sigma 序列（降序）。

        从 sigma_max 到 sigma_min，在对数空间均匀采样 num_steps 个点。
        这是 DDIM/DDPM 推理中最常用的 sigma schedule。

        参数：
            num_steps: 采样步数。
            sigma_min: 最小 sigma（默认 0.002）。
            sigma_max: 最大 sigma（默认 80.0）。

        返回：
            shape (num_steps,) 的 numpy 数组，单调递减。

        异常：
            ValueError: 若 num_steps ≤ 0。
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps 必须 > 0，收到 {num_steps}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min 必须 > 0（log10 域需要正值），收到 {sigma_min}")
        return np.logspace(
            np.log10(sigma_max), np.log10(sigma_min), num_steps, dtype=np.float64
        )

    def step(self, v, x_t, i):
        """
        单步 Euler 更新。

        公式：x_next = x_t + (sigma_next - sigma_i) * v

        参数：
            v: 模型预测的 vector field / score-based 输出，shape 与 x_t 一致。
            x_t: 当前 latent，任意 shape。
            i: 当前步索引（0-based, 0 ≤ i < len(self.sigmas) - 1）。

        返回：
            x_next: 更新后的 latent，dtype 与 x_t 一致。

        异常：
            IndexError: 若 i 超出范围。
        """
        if i < 0 or i >= len(self.sigmas) - 1:
            raise IndexError(
                f"步索引 i={i} 越界，有效范围 [0, {len(self.sigmas) - 2}]"
            )

        sigma_i = self.sigmas[i]
        sigma_next = self.sigmas[i + 1]
        dt = sigma_next - sigma_i  # 负值（sigma 递减）

        x_next = x_t + dt * v

        if self.stochastic:
            # 可选噪声注入（Langevin 风格），仅在 sigma > 0 时有效
            noise_scale = np.sqrt(-dt) if dt < 0 else 0.0
            if noise_scale > 0:
                noise = self._rng.randn(*x_t.shape).astype(x_t.dtype)
                x_next = x_next + noise_scale * noise

        return x_next.astype(x_t.dtype)

    def set_timesteps(self, num_steps, sigma_min=0.002, sigma_max=80.0):
        """重新设置 ODE 步数并重采样 sigma 序列（num_steps 步 → num_steps+1 个 sigma 点）。"""
        self.sigmas = self.get_sigmas(num_steps + 1, sigma_min, sigma_max)

    def __len__(self):
        """返回 sigma 数量（含起始点）。实际 step 次数 = len(sigmas) - 1。"""
        return len(self.sigmas)


class RectifiedFlowScheduler:
    """
    Rectified Flow 风格 ODE 求解器 — 在 t∈[0,1] 空间积分。

    t=1 为纯噪声（x_1 ~ N(0,I)），t=0 为干净数据（x_0）。
    timestep 在 [t_start, t_end] 上线性均匀分布，默认从 1.0 递减到 0.0。

    与 EulerScheduler 的区别：
        - EulerScheduler 工作在 sigma 空间（log-linear 间隔），
          sigma 的值域取决于扩散过程的参数化。
        - RectifiedFlowScheduler 工作在 t∈[0,1] 空间（线性间隔），
          t 直接表示"噪声占比"：x_t = (1-t)·x_0 + t·x_1。
        - 两者 step 公式相同（Euler ODE），但时间/参数空间不同于噪声水平。

    用法：
        scheduler = RectifiedFlowScheduler(num_steps=28)
        x_t = randn(...)  # t=1 的噪声
        for i in range(len(scheduler.timesteps) - 1):
            v = v_theta(x_t, scheduler.timesteps[i])
            x_t = scheduler.step(v, x_t, i)
    """

    def __init__(self, num_steps=28, t_start=1.0, t_end=0.0, seed=None):
        """
        参数：
            num_steps: 采样步数（默认 28，SD3 常用值）。
            t_start: 起始 t（噪声端，默认 1.0）。
            t_end: 终止 t（数据端，默认 0.0）。
            seed: 确定性随机种子。
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps 必须 > 0，收到 {num_steps}")

        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.num_steps = num_steps
        self.timesteps = self._compute_timesteps(num_steps)

        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def _compute_timesteps(self, num_steps):
        """计算 timestep 序列：从 t_start 到 t_end 的 num_steps+1 个线性间隔点。"""
        return np.linspace(self.t_start, self.t_end, num_steps + 1, dtype=np.float64)

    def step(self, v, x_t, i):
        """
        单步 Euler 更新（rectified flow 风格）。

        公式：x_next = x_t + (t_next - t_i) * v

        参数：
            v: 模型预测的 vector field v_θ(x_t, t_i)，shape 与 x_t 一致。
            x_t: 当前 latent，任意 shape。
            i: 当前步索引（0-based, 0 ≤ i < self.num_steps）。

        返回：
            x_next: 更新后的 latent，dtype 与 x_t 一致。

        异常：
            IndexError: 若 i 超出范围。
        """
        if i < 0 or i >= self.num_steps:
            raise IndexError(
                f"步索引 i={i} 越界，有效范围 [0, {self.num_steps - 1}]"
            )

        t_i = self.timesteps[i]
        t_next = self.timesteps[i + 1]
        dt = t_next - t_i  # 负值（t 递减）

        x_next = x_t + dt * v
        return x_next.astype(x_t.dtype)

    def set_timesteps(self, num_steps):
        """重新设置步数（重采样 timestep 序列）。"""
        if num_steps <= 0:
            raise ValueError(f"num_steps 必须 > 0，收到 {num_steps}")
        self.num_steps = num_steps
        self.timesteps = self._compute_timesteps(num_steps)

    def __len__(self):
        """返回 timestep 数量（含起始点）。实际 step 次数 = len - 1。"""
        return len(self.timesteps)
