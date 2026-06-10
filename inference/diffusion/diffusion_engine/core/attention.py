"""
DiT/MMDiT 自注意力与联合注意力模块（torch 实现）

设计原则（与 minivLLM LLM attention 的关键差异）：
- Non-causal full self-attention（DiT 全局 attention，无 mask）
- 无 KV cache（扩散每步 latent 全刷新，无 token 延续性）
- 不使用 GQA（每 head 独立 QKV，与 LLM 不同）
- 不使用 RoPE（DiT 用 2D sinusoidal position embedding，非 rotary）
- 不依赖 flash-attn / xformers（纯 PyTorch，MPS 兼容）

注意：本模块不 import minivLLM 任何代码，不继承其 GQA + causal 逻辑。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Non-causal 多头自注意力（DiT 标准实现）。

    输入 shape: (B, N, D) -> 输出 shape: (B, N, D)
    其中 B=batch, N=token 数（latent patches）, D=hidden_size。

    设计要点：
    - no causal mask: 所有 patches 互相 attend
    - no KV cache: 不存储历史 K/V，每步刷新
    - no GQA: 每个 head 有独立的 Q、K、V 投影
    - no RoPE: position 由 embedding 层单独注入
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        """
        参数:
            dim: 隐藏维度 D（必须是 num_heads 的整数倍）
            num_heads: attention head 数量
            qkv_bias: QKV Linear 是否带 bias（DiT 通常 True）
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须是 num_heads {num_heads} 的整数倍"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 单次 Linear 投影 Q、K、V（合并为 3*dim 输出以提高效率）
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 输出投影
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：standard softmax(QK^T / sqrt(d)) V

        参数:
            x: (B, N, D) 输入 token 序列

        返回:
            (B, N, D) attention 输出
        """
        B, N, D = x.shape

        # 1. QKV 投影: (B, N, 3*D) -> 拆分为 3 个 (B, N, D)
        qkv = self.qkv(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # (B, N, 3, H, d)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, d)
        q, k, v = qkv.unbind(0)  # 各 (B, H, N, d)

        # 2. Scaled Dot-Product Attention（无 mask）
        # 使用 PyTorch 内置 SDPA，MPS 兼容
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,      # DiT: non-causal, 全 attending
            dropout_p=0.0,       # 推理模式不 dropout
            is_causal=False,     # ★ 非 causal（与 LLM 的关键差异）
        )  # (B, H, N, d)

        # 3. 合并 multi-head 并投影输出
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        out = self.proj(attn_out)  # (B, N, D)

        return out


class JointAttention(nn.Module):
    """
    MMDiT-style 联合注意力（toy 简化版）。

    输入两组 token 序列（如 image tokens + text tokens），拼接后做 full
    attention，再拆分回两组。这是 SD3 MMDiT "双流但每层 cross-attention 拼接"
    的 toy 简化——真实 SD3 是独立的 image/image-stream 和 text/text-stream 带
    cross-attention，这里简化为拼接后统一 attend。

    输入:
        x1: (B, N1, D) 第一组 tokens（如 image latent patches）
        x2: (B, N2, D) 第二组 tokens（如 text tokens）

    输出:
        y1: (B, N1, D) 第一组输出（attend 了两组的信息）
        y2: (B, N2, D) 第二组输出（attend 了两组的信息）
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ):
        """
        参数:
            dim: 隐藏维度 D
            num_heads: attention head 数量
            qkv_bias: QKV Linear 是否带 bias
        """
        super().__init__()
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：拼接两组 tokens、做 full attention、拆分回来。

        参数:
            x1: (B, N1, D) 第一组 tokens
            x2: (B, N2, D) 第二组 tokens

        返回:
            (y1, y2)，各为 (B, N1, D) 和 (B, N2, D)
        """
        B, N1, D = x1.shape
        N2 = x2.shape[1]

        # 拼接: (B, N1+N2, D)
        combined = torch.cat([x1, x2], dim=1)

        # Full attention over all tokens
        attn_out = self.attn(combined)  # (B, N1+N2, D)

        # 拆分
        y1 = attn_out[:, :N1, :]  # (B, N1, D)
        y2 = attn_out[:, N1:, :]  # (B, N2, D)

        return y1, y2
