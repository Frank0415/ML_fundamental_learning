"""
DiT-style Transformer Block（AdaLN-Zero toy 版）

基于 DiT (Scalable Diffusion Models with Transformers) 的 transformer block，
使用 AdaLN (Adaptive Layer Normalization) 代替 bare Pre-Norm。

设计要点：
- 用 timestep embedding 生成 scale/shift/gate 三组参数，
  分别调制 attention 和 FFN 层的 LayerNorm 与 residual。
- 支持纯 image self-attention 或 joint attention（当提供 text tokens 时）。
- 不使用 RoPE（DiT 用 2D sinusoidal position embedding）。
- 不使用 KV cache（扩散每步全刷新）。
- 完整注释为中文。

与 minivLLM 的关键差异：
- minivLLM 使用静态 RMSNorm + GQA + causal + KV cache + RoPE
- 本模块使用 AdaLN + full attention + gate 调制 + 无因果 mask
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention, JointAttention


def _ada_ln_modulation(
    t_emb: torch.Tensor,
    dim: int,
    num_mod_params: int = 6,
) -> torch.Tensor:
    """
    从 timestep embedding 生成 AdaLN 调制参数。

    参数:
        t_emb: (B, D) timestep embedding
        dim: 隐藏维度 D
        num_mod_params: 要生成的参数量（默认 6: shift/scale/gate × 2 组 [attn + FFN]）

    返回:
        (B, num_mod_params * D) 调制参数：对 attn 和 FFN 各提供 shift/scale/gate
    """
    # 一个小的 SiLU MLP 来生成调制参数
    mod = F.silu(t_emb)
    mod = nn.Linear(dim, num_mod_params * dim, bias=True).to(t_emb.device)
    # 注意：在 forward 里每次创建 Linear 太低效。
    # 实际实现应该在 __init__ 里创建 nn.Linear 并在此调用。
    # 这里返回一个占位逻辑，实际调制在 DiTBlock.__init__ 中的调制 MLP 完成。
    return mod(t_emb)


class DiTBlock(nn.Module):
    """
    DiT Transformer Block with AdaLN modulation.

    输入：
        x: (B, N, D) image tokens
        t_emb: (B, D) timestep embedding
        text_tokens: (B, M, D) 可选 text tokens（为 None 时用 self-attention）

    输出：
        (B, N, D) 调制后的 image tokens

    内部结构：
        1. AdaLN 调制 → attention（self 或 joint）
        2. 残差连接（gate 调制）
        3. AdaLN 调制 → FFN（GELU + Linear）
        4. 残差连接（gate 调制）

    注意：这是 toy 简化版 AdaLN，真实 DiT 的 AdaLN-Zero 还会初始化为零，
    本实现跳过零初始化（在 T18 总结时标注为 toy 简化）。
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        """
        参数:
            hidden_size: 隐藏维度 D
            num_heads: attention head 数量
            mlp_ratio: FFN 扩展比（MLP 内部 dim = hidden_size * mlp_ratio）
        """
        super().__init__()
        self.hidden_size = hidden_size

        # ===== AdaLN 调制参数生成 =====
        # 调制参数: shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn
        # 共 6 组，每组一个 vector of size hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

        # ===== Attention =====
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=True)
        self.joint_attn = JointAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=True)

        # ===== FFN =====
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: (B, N, D) image tokens
            t_emb: (B, D) timestep embedding
            text_tokens: (B, M, D) 可选 text tokens

        返回:
            (B, N, D) 调制后的 image tokens
        """
        # 生成调制参数: (B, 6*D)
        mod_params = self.adaLN_modulation(t_emb)  # (B, 6*D)
        (
            shift_attn, scale_attn, gate_attn,
            shift_ffn, scale_ffn, gate_ffn,
        ) = mod_params.chunk(6, dim=1)
        # 每组的 shape: (B, D)，需要 unsqueeze 以便与 (B, N, D) 广播
        shift_attn = shift_attn.unsqueeze(1)  # (B, 1, D)
        scale_attn = scale_attn.unsqueeze(1)  # (B, 1, D)
        gate_attn = gate_attn.unsqueeze(1)    # (B, 1, D)
        shift_ffn = shift_ffn.unsqueeze(1)    # (B, 1, D)
        scale_ffn = scale_ffn.unsqueeze(1)    # (B, 1, D)
        gate_ffn = gate_ffn.unsqueeze(1)      # (B, 1, D)

        # ===== Attention 子层 =====
        # AdaLN 调制 norm1: norm(x) * (1 + scale) + shift
        normed_x = self.norm1(x)
        modulated = normed_x * (1.0 + scale_attn) + shift_attn  # (B, N, D)

        # Attention（joint 或 self）
        if text_tokens is not None:
            # Joint attention: modulated tokens + text tokens 一起 attend
            # 注意：text_tokens 也受调制？在 toy 实现中暂不对 text 做 AdaLN，
            # 直接拼接输入 joint attention（SD3 的真实现有独立的 text stream 调制）
            attn_out, _ = self.joint_attn(modulated, text_tokens)
            # 只取 image 部分
        else:
            attn_out = self.self_attn(modulated)

        # 残差连接（gate 调制）
        x = x + gate_attn * attn_out  # (B, N, D)

        # ===== FFN 子层 =====
        normed_x = self.norm2(x)
        modulated = normed_x * (1.0 + scale_ffn) + shift_ffn  # (B, N, D)
        ffn_out = self.ffn(modulated)
        x = x + gate_ffn * ffn_out  # (B, N, D)

        return x
