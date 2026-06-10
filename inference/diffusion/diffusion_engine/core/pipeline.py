"""
pipeline.py — 扩散推理主流程（完整 denoising loop）

提供：
- DiffusionPipeline: 从 prompt 到 image 的端到端推理流程

★★★ 流程概览（6 步） ★★★
  1. encode prompt  → cond/uncond embedding（含 cache）
  2. init latent    → 用 seed 初始化噪声 latent
  3. scheduler timesteps → 预计算 ODE 积分时间步
  4. denoising loop → for t in timesteps:
     a. denoiser forward（batched 或 sequential CFG）
     b. CFG in vector field space（v_cfg = v_uncond + s * (v_cond - v_uncond)）
     c. Euler step（latents = latents + dt * v_cfg）
  5. decode latent  → VAE decoder → RGB image

★★★ CFG 在 vector field 层面做，不是 latent 层面 ★★★
  正确: v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
  错误: x_cfg = x_uncond + cfg_scale * (x_cond - x_uncond)  ← 不要在 latent 上做 CFG！

  原因: CFG 是对模型预测方向（vector field）的引导，不是对 latent 本身的插值。
  在 rectified flow 框架下，v_θ 直接给出概率流方向，CFG 在此方向加强 cond 的
  贡献，保持 ODE 积分的一致性。

★★★ Batched vs Sequential CFG ★★★
  - BATCHED: 拼接 [latents, latents] + [uncond_emb, cond_emb] → 一次 forward
    显存高一倍（2× batch），但快（一次 forward + 一次 backward-free 推理）
  - SEQUENTIAL: 两次 forward（uncond + cond）
    显存低（单 batch），但慢（两次 forward）

★★★ diffusion cache ≠ LLM KV cache ★★★
  本模块涉及的是 prompt embedding cache 和 ping-pong latent buffer：
  - prompt embedding cache: 缓存 text encoder 输出，避免同一 prompt 重复编码
  - ping-pong buffer: 两个 latent buffer 交替使用，避免 per-step malloc/free
  而不是 LLM 的 attention KV cache。

未使用 minivLLM 任何代码。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .memory_manager import CFGMode, LatentBufferManager, MemoryStats
from .scheduler import RectifiedFlowScheduler
from .text_conditioning import PromptEmbeddings, TextConditioner


class DiffusionPipeline:
    """
    扩散推理主管道。

    封装从 prompt → latent → denoising loop → image 的完整流程。
    支持 batched 和 sequential 两种 CFG 模式，以及可选的 memory_stats 追踪。

    用法示例（需 torch 已安装）:
        >>> from diffusion_engine.core import TinyDiT, ToyTextConditioner, ToyVAE
        >>> from diffusion_engine.core.scheduler import RectifiedFlowScheduler
        >>> from diffusion_engine.core.pipeline import DiffusionPipeline
        >>>
        >>> dit = TinyDiT(in_channels=4, patch_size=2, hidden_size=64, depth=2,
        ...               num_heads=4, text_dim=64, max_text_len=16)
        >>> scheduler = RectifiedFlowScheduler(num_steps=28)
        >>> conditioner = ToyTextConditioner(hidden_size=64, max_seq_len=16)
        >>> vae = ToyVAE()
        >>>
        >>> pipeline = DiffusionPipeline(dit, scheduler, conditioner, vae)
        >>> images = pipeline.run("a cat", num_steps=4, height=64, width=64, seed=0)
        >>> print(images.shape)  # (1, 3, 64, 64)
    """

    def __init__(
        self,
        denoiser: nn.Module,
        scheduler: RectifiedFlowScheduler,
        conditioner: TextConditioner,
        vae: nn.Module,
        memory_manager: Optional[LatentBufferManager] = None,
    ):
        """
        参数:
            denoiser: 去噪网络（如 TinyDiT），需支持 forward(x, t, text_tokens)。
            scheduler: 噪声调度器（RectifiedFlowScheduler），用于 timestep 计算。
            conditioner: 文本条件编码器（TextConditioner 协议）。
            vae: VAE 解码器（如 ToyVAE），需支持 decode(z) → x_pixel。
            memory_manager: 可选 LatentBufferManager（若提供则使用 ping-pong buffer）。
        """
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.vae = vae
        self.memory_manager = memory_manager
        self.memory_stats = MemoryStats()

    def _get_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        """从 denoiser 参数推导设备和数据类型。"""
        try:
            param = next(self.denoiser.parameters())
            return param.device, param.dtype
        except StopIteration:
            # 无参数模型（极端情况），fallback 到 cpu/float32
            return torch.device("cpu"), torch.float32

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 28,
        cfg_scale: float = 7.5,
        height: int = 64,
        width: int = 64,
        seed: int = 0,
        mode: CFGMode = CFGMode.BATCHED,
    ) -> torch.Tensor:
        """
        执行完整扩散推理。

        参数:
            prompt: 正向文本提示（如 "a cat sitting on a chair"）。
            negative_prompt: 负向提示（默认空字符串，无负面引导）。
            num_steps: ODE 积分步数（默认 28）。
            cfg_scale: CFG 引导强度（默认 7.5；1.0 表示无 CFG）。
            height: 输出图像高度（像素，需为 8 的整数倍，默认 64）。
            width: 输出图像宽度（像素，需为 8 的整数倍，默认 64）。
            seed: 随机种子（控制初始噪声和缓存 embedding 的确定性）。
            mode: CFG 执行模式（CFGMode.BATCHED 或 CFGMode.SEQUENTIAL）。

        返回:
            (1, 3, height, width) RGB 图像 tensor（值域非约束，建议 clamp 到 [0,1]）。

        异常:
            ValueError: 若 height/width 不是 8 的整数倍。
            RuntimeError: 若 scheduler 步数设置失败。
        """
        # ── 环境准备 ──────────────────────────────────────────────────
        device, dtype = self._get_device_and_dtype()
        self.memory_stats.start()

        # 验证 latent 尺寸
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"height={height} 和 width={width} 必须是 8 的整数倍（VAE 8x downsample）"
            )

        B = 1  # batch size（推理固定为 1）
        latent_channels = getattr(self.denoiser, "in_channels", 4)
        latent_h = height // 8
        latent_w = width // 8

        # ── 步骤 1: Encode prompts（含 cache） ────────────────────────
        cond_embeddings = self.conditioner.encode(
            prompt=prompt,
            negative_prompt="",  # 仅缓存 cond
        )
        cond_emb = cond_embeddings.cond  # (B, max_seq_len, hidden_size)

        # 无条件 embedding（用于 CFG）
        if cfg_scale != 1.0 and negative_prompt is not None:
            uncond_embeddings = self.conditioner.encode(
                prompt=negative_prompt,
                negative_prompt="",
            )
            uncond_emb = uncond_embeddings.cond
        else:
            uncond_emb = None  # cfg_scale=1.0 时不需要 uncond

        # ── 步骤 2: 初始化噪声 latent ────────────────────────────────
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            B, latent_channels, latent_h, latent_w,
            generator=generator, device=device, dtype=dtype,
        )

        # ── 步骤 3: 预计算 timesteps ──────────────────────────────────
        self.scheduler.set_timesteps(num_steps)
        timesteps = torch.from_numpy(self.scheduler.timesteps).to(
            device=device, dtype=dtype
        )
        # timesteps: (num_steps + 1,) — 从 t_start 到 t_end
        # 实际执行 `num_steps` 次 step

        # ── 步骤 4: Denoising loop ────────────────────────────────────
        for i in range(num_steps):
            t = timesteps[i]       # 当前 timestep（标量 tensor）
            t_next = timesteps[i + 1]  # 下一步 timestep
            t_batch = t.unsqueeze(0).expand(B)  # (B,)

            # ── 4a: Denoiser forward（CFG） ─────────────────────────
            if cfg_scale == 1.0:
                # 无 CFG — 仅 cond forward
                v_cfg = self.denoiser(latents, t_batch, cond_emb)

            elif mode == CFGMode.BATCHED:
                # Batched CFG: 拼接 cond 和 uncond → 一次 forward
                # latent cat: (B, C, H, W) → (2B, C, H, W)
                latents_cat = torch.cat([latents, latents], dim=0)
                text_cat = torch.cat([uncond_emb, cond_emb], dim=0)
                t_cat = t_batch.expand(B * 2)

                v_cat = self.denoiser(latents_cat, t_cat, text_cat)
                v_uncond, v_cond = v_cat.chunk(2, dim=0)

                # CFG in vector field space（见 T5 共识）
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

            elif mode == CFGMode.SEQUENTIAL:
                # Sequential CFG: 两次 forward
                v_uncond = self.denoiser(latents, t_batch, uncond_emb)
                v_cond = self.denoiser(latents, t_batch, cond_emb)

                # CFG in vector field space
                v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

            else:
                raise ValueError(f"未知 CFGMode: {mode}")

            # ── 4b: Euler step（ODE 积分） ───────────────────────────
            # 公式: x_next = x_t + (t_next - t) * v
            # 这是 rectified flow ODE 的 Euler 积分
            dt = t_next - t  # 标量（通常负值，因为 t 递减）
            latents = latents + dt * v_cfg

            # ── 可选: 使用 memory_manager 的 ping-pong buffer ────────
            if self.memory_manager is not None:
                self.memory_manager.get("x_next").copy_(latents)
                self.memory_manager.swap("x_t", "x_next")
                latents = self.memory_manager.get("x_t")

        # ── 步骤 5: Decode latent → image ─────────────────────────────
        images = self.vae.decode(latents)

        return images

    def profile_run(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_steps: int = 4,
        cfg_scale: float = 7.5,
        height: int = 64,
        width: int = 64,
        seed: int = 0,
        mode: CFGMode = CFGMode.BATCHED,
    ) -> dict:
        """
        执行推理并记录 memory stats（用于 benchmark）。

        参数:
            同 run()。

        返回:
            {
                "image_shape": tuple,
                "memory_snapshot": dict (来自 MemoryStats.snapshot()),
                "num_steps": int,
                "cfg_scale": float,
                "mode": str,
                "latent_shape": tuple,
            }
        """
        self.memory_stats.start()
        images = self.run(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            height=height,
            width=width,
            seed=seed,
            mode=mode,
        )
        snapshot = self.memory_stats.snapshot()

        return {
            "image_shape": tuple(images.shape),
            "memory_snapshot": snapshot,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "mode": mode.value,
            "latent_shape": (1, getattr(self.denoiser, "in_channels", 4), height // 8, width // 8),
        }
