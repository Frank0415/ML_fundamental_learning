"""
diffusion_engine.core — DiT/MMDiT 核心模块

模块结构：
    scheduler.py         — EulerScheduler + RectifiedFlowScheduler（纯 numpy）
    rectified_flow.py    — Rectified flow ODE 步进与采样（纯 numpy）
    timestep_embedding.py — 正弦-余弦时间步嵌入（纯 numpy）
    attention.py         — SelfAttention（non-causal full attention，需 torch）
    transformer_block.py — DiTBlock（AdaLN 调制 + self/joint attention + FFN，需 torch）
    dit.py               — TinyDiT（toy-scale 扩散 transformer，需 torch）
    text_conditioning.py — ToyTextConditioner + PromptEmbeddings（需 torch）
    memory_manager.py    — LatentBufferManager + MemoryStats + CFGMode（需 torch）
    vae_stub.py          — ToyVAE（需 torch）
    pipeline.py          — DiffusionPipeline（需 torch）

所有纯 numpy 模块可脱离 torch 独立使用。PyTorch 模块需要 torch ≥ 2.7。
"""

# T10 纯 numpy 模块 — 始终可用
from .scheduler import EulerScheduler, RectifiedFlowScheduler
from .rectified_flow import rectified_flow_step, rectified_flow_sample
from .timestep_embedding import sinusoidal_embedding, timestep_to_float

# T11 模块 — 需 torch，以惰性方式导入
try:
    from .attention import SelfAttention, JointAttention
    from .transformer_block import DiTBlock
    from .dit import TinyDiT, PatchEmbed, Unpatchify
    _HAS_TORCH_MODULES = True
except ImportError:
    _HAS_TORCH_MODULES = False

# T12 模块 — 需 torch，以惰性方式导入
try:
    from .text_conditioning import (
        PromptEmbeddings,
        TextConditioner,
        ToyTextConditioner,
        HFCachedTextConditioner,
    )
    from .memory_manager import (
        CFGMode,
        LatentBufferManager,
        EmbeddingCache,
        MemoryStats,
        estimate_latent_buffer_bytes,
    )
    from .vae_stub import ToyVAE
    from .pipeline import DiffusionPipeline
    from .diffusion_gemma_pipeline import EntropyBoundedSampler, DiffusionGemmaPipeline
    _HAS_T12_MODULES = True
except ImportError:
    _HAS_T12_MODULES = False

__all__ = [
    # T10 (always available)
    "EulerScheduler",
    "RectifiedFlowScheduler",
    "rectified_flow_step",
    "rectified_flow_sample",
    "sinusoidal_embedding",
    "timestep_to_float",
    # T11 (torch required)
    "SelfAttention",
    "JointAttention",
    "DiTBlock",
    "TinyDiT",
    "PatchEmbed",
    "Unpatchify",
    # T12 (torch required)
    "PromptEmbeddings",
    "TextConditioner",
    "ToyTextConditioner",
    "HFCachedTextConditioner",
    "CFGMode",
    "LatentBufferManager",
    "EmbeddingCache",
    "MemoryStats",
    "estimate_latent_buffer_bytes",
    "ToyVAE",
    "DiffusionPipeline",
    "EntropyBoundedSampler",
    "DiffusionGemmaPipeline",
]
