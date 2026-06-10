"""
Tiny DiT（toy 规模的 Diffusion Transformer）

基于 DiT（Scalable Diffusion Models with Transformers, Peebles & Xie 2023）
的 toy 实现，用于测试 shape 系统和模块组合，**不作为真实图像生成用途**。

Toy 简化清单（在 T18 总结时需标注以下简化）：
- 使用可学习 position embedding 而非 2D sin/cos embedding
- AdaLN-Zero 使用最简版（无 zero-initialization gate）
- 不做 MMDiT 完整双流，仅支持 joint attention toy 版本
- 仅支持 patch_size=2（hardcoded，不处理 patch_size!=2 情况）
- 预测目标：epsilon（noise prediction），非 v-prediction
- 不与 minivLLM 共享任何代码

关键设计说明：
- Timestep embedding 使用 sinusoidal encoding（torch 版重新实现），
  虽 T10 已有 numpy 版本，但 TinyDiT 需要 torch 兼容的版本以便梯度流。
- 文本条件注入通过简单 nn.Linear 投影实现，不做 T5/CLIP 完整编码。
- Patchify/unpatchify 基于 reshape 和 transpose 实现，无卷积 patch embed。
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .transformer_block import DiTBlock


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding（torch 实现）。

    与 T10 的 numpy 版本等价，但使用 torch 以便整合进 nn.Module。

    参数:
        t: (B,) 或 (B, 1) timestep 值（通常在 [0, 1] 范围）
        dim: embedding 维度（必须是偶数）
        max_period: 最大周期（控制频率范围）

    返回:
        (B, dim) sinusoidal embedding
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)  # (B, 1)
    # 确保 t 是 float
    t = t.float()

    half = dim // 2
    # 频率: 1 / (10000^(2i/dim))
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    )  # (half,)

    # t * freqs: (B, 1) * (half,) -> (B, half)
    args = t * freqs.unsqueeze(0)  # (B, half)

    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
    return embedding


class PatchEmbed(nn.Module):
    """
    将 image latent 切分为 patches。

    输入: (B, C, H, W)
    输出: (B, N, D)  其中 N = (H/p) * (W/p), D = p * p * C
    """

    def __init__(self, in_channels: int = 4, patch_size: int = 2, hidden_size: int = 64):
        """
        参数:
            in_channels: 输入通道数 C
            patch_size: patch 边长 p
            hidden_size: 输出隐藏维度 D（必须等于 p*p*C）
        """
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Patchify: (B, C, H, W) -> (B, N, p*p*C)

        其中 N = (H/p) * (W/p)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # 确保尺寸可被 patch_size 整除
        assert H % p == 0 and W % p == 0, f"Image size {H}x{W} 不能被 patch_size={p} 整除"

        # reshape: (B, C, H/p, p, W/p, p) -> (B, H/p, W/p, C*p*p) -> (B, N, D)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H/p, W/p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)  # (B, N, D)
        return x


class Unpatchify(nn.Module):
    """
    将 patch tokens 重组为 image latent。

    输入: (B, N, p*p*C)
    输出: (B, C, H, W)  其中 H=W=sqrt(N)*p
    """

    def __init__(self, out_channels: int = 4, patch_size: int = 2, hidden_size: int = 64):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify: (B, N, p*p*C) -> (B, C, H, W)
        """
        B, N, D = x.shape
        p = self.patch_size
        C = self.out_channels

        # N = (H/p)*(W/p)，假设正方形 H=W
        grid_size = int(math.sqrt(N))
        assert grid_size * grid_size == N, f"Token 数 {N} 不是完全平方数，假设正方形 latent"

        # reshape: (B, grid, grid, C, p, p) -> (B, C, grid*p, grid*p)
        x = x.reshape(B, grid_size, grid_size, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, grid, p, grid, p)
        x = x.reshape(B, C, grid_size * p, grid_size * p)  # (B, C, H, W)
        return x


class TinyDiT(nn.Module):
    """
    最小化的 DiT 模型（toy 规模）。

    参数规模很小（hidden_size=64, depth=2, num_heads=4），仅用于 shape 测试
    和 pipeline 集成调试。不训练，无预训练权重。

    输入:
        x: (B, C, H, W) 噪声 latent
        t: (B,) 或 (B, 1) timestep
        text_tokens: (B, L, D_text) 可选文本 token embedding

    输出:
        (B, C, H, W) epsilon prediction（噪声预测）

    内部流程:
        1. patchify: (B,C,H,W) -> (B, N, D)
        2. pos_embed: 加法注入可学习 position embedding
        3. timestep_embedding: (B,) -> (B, D) sinusoidal encoding + MLP
        4. text_proj (可选): (B, L, D_text) -> (B, L, D)
        5. DiTBlock × depth: 逐步调制
        6. Final LayerNorm + Linear: (B, N, D) -> (B, N, p*p*C)
        7. unpatchify: (B, N, p*p*C) -> (B, C, H, W)
    """

    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        text_dim: int = 64,
        max_text_len: int = 16,
    ):
        """
        参数:
            in_channels: 输入通道数（latent 通道，通常 4 或 16）
            patch_size: patch 边长 p（toy 版本仅支持 p=2）
            hidden_size: 隐藏维度 D
            depth: DiT block 数量
            num_heads: attention head 数量
            text_dim: 文本 embedding 维度（输入维度）
            max_text_len: 最大文本 token 数
        """
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads

        # 每个 patch 展开为 p*p*C 维
        patch_dim = patch_size * patch_size * in_channels

        # == Patch 化 / 逆 patch 化 ==
        self.patch_embed = PatchEmbed(in_channels, patch_size, hidden_size)
        self.unpatch = Unpatchify(in_channels, patch_size, hidden_size)

        # 将 patch_dim 投影到 hidden_size（如果不等）
        self.patch_proj = nn.Linear(patch_dim, hidden_size, bias=True) if patch_dim != hidden_size else nn.Identity()

        # == 可学习 position embedding（toy 简化） ==
        # 假设最大 latent size 为 64×64，patch_size=2 → N_max = 32*32 = 1024
        max_tokens = 1024
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, hidden_size))

        # == Timestep embedding 投影 ==
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        # == Text 投影 ==
        self.text_proj = nn.Linear(text_dim, hidden_size, bias=True)

        # == DiT blocks ==
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])

        # == 最终输出头 ==
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_dim, bias=True)

        # == 初始化 ==
        self._init_weights()

    def _init_weights(self):
        """基本的权重初始化（xavier uniform + 零初始化 pos_embed）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # pos_embed 以小的正态噪声初始化
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播：预测噪声 epsilon。

        参数:
            x: (B, C, H, W) 噪声 latent
            t: (B,) 或 (B, 1) timestep，范围 [0, 1]
            text_tokens: (B, L, D_text) 可选文本条件

        返回:
            (B, C, H, W) 预测的噪声 epsilon
        """
        B, C, H, W = x.shape
        p = self.patch_size
        N = (H // p) * (W // p)
        D = self.hidden_size

        # 1. Patchify: (B, C, H, W) -> (B, N, p*p*C)
        tokens = self.patch_embed(x)  # (B, N, patch_dim)

        # 2. 投影到 hidden_size
        tokens = self.patch_proj(tokens)  # (B, N, D)

        # 3. 注入 position embedding
        tokens = tokens + self.pos_embed[:, :N, :]  # (B, N, D)

        # 4. Timestep embedding: (B,) -> (B, D)
        t_emb = timestep_embedding(t, D)  # (B, D)
        t_emb = self.t_embedder(t_emb)    # (B, D)

        # 5. Text 投影（如果提供）: (B, L, D_text) -> (B, L, D)
        text_emb = None
        if text_tokens is not None:
            text_emb = self.text_proj(text_tokens)  # (B, L, D)

        # 6. 通过 DiT blocks
        for block in self.blocks:
            tokens = block(tokens, t_emb, text_emb)  # (B, N, D)

        # 7. 最终 output head: (B, N, D) -> (B, N, patch_dim)
        tokens = self.final_norm(tokens)
        tokens = self.final_linear(tokens)  # (B, N, patch_dim)

        # 8. Unpatchify: (B, N, patch_dim) -> (B, C, H, W)
        output = self.unpatch(tokens)  # (B, C, H, W)

        return output
