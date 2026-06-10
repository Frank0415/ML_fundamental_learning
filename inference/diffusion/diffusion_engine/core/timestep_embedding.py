"""
timestep_embedding.py — 时间步正弦编码

纯 numpy 实现，不依赖 torch。提供：
- sinusoidal_embedding: 标准 Transformer 风格的正弦/余弦时间步嵌入
- timestep_to_float: 将 int/float/list/ndarray 统一转为 float

工作原理：
  给定时间步 t ∈ [0,1]（rectified flow）或任意实数（sigma 空间），
  使用一组对数间隔的频率生成 (dim,) 维嵌入向量，前半为 sin，后半为 cos。
  这与 Transformer 位置编码和 DiT adaLN 中的 timestep embedding 一致。

与 DiT adaLN modulation 的关系：
  sinusoidal_embedding 输出可直接通过一个小的 MLP 映射为 adaLN 的
  scale / shift / gate 参数，作为后续 transformer block 的条件注入。
"""

import numpy as np


def sinusoidal_embedding(t, dim=256, max_period=10000.0, dtype=np.float32):
    """
    正弦-余弦时间步嵌入（Transformer 风格）。

    参数：
        t: 标量或 (B,) 数组，时间步值。rectified flow 中通常 t∈[0,1]。
        dim: 嵌入维度（默认 256）。必须是偶数，输出维度 = dim。
        max_period: 最大周期，控制最低频率（默认 10000.0）。
        dtype: 输出 dtype。

    返回：
        形状为 (dim,) 的向量（单标量输入）或 (B, dim) 矩阵（数组输入）。

    实现细节：
        - 频率从 1/(max_period) 到 1，在对数空间均匀采样 dim/2 个频率。
        - 前半 dim/2 维为 sin(2π * freq * t)，后半为 cos(2π * freq * t)。
        - 与 PyTorch 中常用的 `get_timestep_embedding` 等效。
    """
    if dim % 2 != 0:
        raise ValueError(f"dim 必须是偶数，收到 {dim}")

    t = np.asarray(t, dtype=np.float64)
    scalar_input = t.ndim == 0
    if scalar_input:
        t = t.reshape(1)  # 转为 (1,) 便于广播

    half = dim // 2
    # 对数空间均匀采样的频率指数，从 0 到 -(half-1)*log10(max_period)/half
    freqs = np.exp(
        -np.arange(half, dtype=np.float64) * np.log(max_period) / (half - 1)
    )
    # 形状: (half,) —— 从 1/max_period 到 1 的对数间隔

    # t 形状 (B,) × freqs 形状 (half,) → (B, half)
    args = np.outer(t, freqs * np.pi * 2)

    emb_sin = np.sin(args)  # (B, half)
    emb_cos = np.cos(args)  # (B, half)
    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (B, dim)

    if scalar_input:
        emb = emb[0]  # 回到 (dim,)

    return emb.astype(dtype)


def timestep_to_float(t):
    """
    将时间步从 int / float / list / ndarray 统一转为 Python float。

    用途：
        - 在 scheduler 中，timestep 可能是 int（步索引）、float（sigma/t 值）
          或 numpy scalar。本函数确保统一转换为 float 用于日志和调试。

    参数：
        t: 任意标量类型或包含单个元素的数组。

    返回：
        Python float。

    异常：
        ValueError: 如果 t 是包含多个元素的数组。
    """
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.size != 1:
        raise ValueError(
            f"timestep_to_float 需要标量，收到 shape={t_arr.shape}"
        )
    return float(t_arr.flat[0])
