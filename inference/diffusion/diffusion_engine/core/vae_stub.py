"""
vae_stub.py — Toy VAE（编解码器占位实现）

提供：
- ToyVAE: nn.Module，Conv2d 8x downsample + ConvTranspose2d 8x upsample

关键说明：
  本模块是 toy 实现 —— 不复制 SD3/FLUX VAE 权重，不依赖 diffusers。
  仅用于 pipeline shape 验证和端到端 smoke test。

真实 SD3 VAE 的要点（仅供对比参考）：
  - SD3 VAE 基于 AutoencoderKL（SD 1.5 同款改进版）
  - encoder: 4 层 stride-2 conv（16x downsample），中间带 residual blocks
  - decoder: 4 层 stride-2 conv transpose（16x upsample）
  - latent channels: 4（RGB 经 encoder 压缩为 4 通道）
  - latent_scale_factor: 0.18215（SD3 与 SDXL 共用此值）

ToyVAE 简化清单（T18 总结时需标注）：
  - 仅 8x downsample（3 层 stride-2，非 16x）
  - 无 residual blocks / attention / group_norm
  - 无 KL 正则化或 latent quantization
  - 不保证编解码像素一致性（toy 级质量）
"""

import torch
import torch.nn as nn


class ToyVAE(nn.Module):
    """
    Toy VAE 编解码器。

    编码: (B, 3, H, W) → 8x ↓ → (B, 4, H/8, W/8)
    解码: (B, 4, H/8, W/8) → 8x ↑ → (B, 3, H, W)

    参数:
        latent_channels: latent 通道数（默认 4，与 SD3 一致）。
        img_channels: RGB 输入通道数（默认 3）。
        base_channels: encoder 起手通道数（默认 16）。

    注意:
        - latent_scale_factor = 0.18215：与 SD3/FLUX 一致，但本 stub 内部不实际使用
          （在真实实现中，encode 后 latent = encoder(x) * scale_factor
           decode 前 z = raw_latent / scale_factor）
        - 输入 H、W 必须是 8 的整数倍
    """

    # SD3/FLUX 的 latent 缩放因子 —— 本 module 内部不用，但 pipeline 可能引用
    latent_scale_factor: float = 0.18215

    def __init__(
        self,
        latent_channels: int = 4,
        img_channels: int = 3,
        base_channels: int = 16,
    ):
        """
        参数:
            latent_channels: 输出 latent 通道数（默认 4）。
            img_channels: 输入 RGB 通道数（默认 3）。
            base_channels: encoder 起始通道数（默认 16）。
        """
        super().__init__()
        self.latent_channels = latent_channels
        self.img_channels = img_channels

        # == Encoder: 3 层 stride-2 Conv2d → 8x downsample ==
        # (B, 3, H, W)   → Conv2d(3→16, s=2)  → (B, 16, H/2, W/2)
        # (B, 16, H/2)    → Conv2d(16→32, s=2) → (B, 32, H/4, W/4)
        # (B, 32, H/4)    → Conv2d(32→4, s=2)  → (B, 4,  H/8, W/8)
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, latent_channels, kernel_size=3, stride=2, padding=1),
        )

        # == Decoder: 3 层 stride-2 ConvTranspose2d → 8x upsample ==
        # (B, 4, H/8)  → ConvT2d(4→32, s=2)  → (B, 32, H/4)
        # (B, 32, H/4) → ConvT2d(32→16, s=2) → (B, 16, H/2)
        # (B, 16, H/2) → ConvT2d(16→3, s=2)  → (B, 3,  H)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, base_channels * 2,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                base_channels * 2, base_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
            nn.SiLU(),
            nn.ConvTranspose2d(
                base_channels, img_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1,
            ),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 初始化所有 conv 权重，bias 置零。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """
        将像素图像编码为 latent。

        参数:
            x_pixel: (B, 3, H, W) RGB 图像，H、W 需为 8 的整数倍。

        返回:
            (B, 4, H/8, W/8) latent 表示。

        注意:
            在真实 SD3 VAE 中，这里还会乘以 self.latent_scale_factor。
            本 toy 实现不做 scale 转换。
        """
        return self.encoder(x_pixel)

    def decode(self, z_latent: torch.Tensor) -> torch.Tensor:
        """
        将 latent 解码为像素图像。

        参数:
            z_latent: (B, 4, H_lat, W_lat) latent 表示。

        返回:
            (B, 3, H_lat*8, W_lat*8) RGB 图像（值域非约束，建议 clamp 到 [0,1]）。

        注意:
            在真实 SD3 VAE 中，这里会先将 z_latent 除以 self.latent_scale_factor。
            本 toy 实现不做 scale 转换。
        """
        return self.decoder(z_latent)

    def forward(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """
        encode → decode 往返（用于测试 shape 保持性）。

        注意：由于无 KL 正则化且是 toy 级网络，不保证像素值相近。
        """
        z = self.encode(x_pixel)
        return self.decode(z)
