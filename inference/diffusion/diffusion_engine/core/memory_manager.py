"""
memory_manager.py — 扩散推理显存管理器

提供：
- LatentBufferManager: latent 预分配、ping-pong buffer
- EmbeddingCache: prompt embedding 缓存包装
- MemoryStats: 显存统计接口（CUDA/MPS/CPU fallback）
- CFGMode: cfg 模式枚举（sequential / batched）

★★★ 为什么 diffusion 不需要 LLM KV cache，但需要 latent buffer manager？ ★★★

┌────────────────────────────┬─────────────────────────────────────────────────┐
│ LLM 推理                   │ Diffusion 推理                                    │
├────────────────────────────┤─────────────────────────────────────────────────┤
│ 自回归逐 token 追加        │ 去噪步间 latent 全刷新（非累加）                   │
│ KV cache: 过去 token 的    │ 无"过去 token"概念——latent 每步被 v_θ 更新的结果  │
│ key/value 历史跨步复用     │ 替换，旧 latent 不需要存                           │
│ 不存 KV cache → 每步重算   │ 每步必须做完整 forward（所有 patches 互相 attend）│
│ O(N) attention → O(N²)     │ O(N²) attention 是固有成本，无可避免               │
├────────────────────────────┤─────────────────────────────────────────────────┤
│ 推理优化焦点：              │ 推理优化焦点：                                     │
│ - continuous batching      │ - batched CFG（一次 forward 处理 cond+uncond）    │
│ - paged attention          │ - latent buffer 预分配（避免重复 malloc）          │
│ - prefix cache             │ - prompt embedding cache（避免重复 text encode）  │
│ - speculative decoding     │ - VAE tiling（降低 decode 峰值显存）               │
└────────────────────────────┴─────────────────────────────────────────────────┘

我们的内存管理就是 ping-pong buffer 预分配 + embedding cache 复用：
  - ping-pong：两个 buffer 交替使用，避免 per-step malloc/free
  - embedding cache：text encoder 输出缓存，避免同一 prompt 重复编码

未使用 minivLLM 任何代码。
"""

from __future__ import annotations

import enum
import sys
import tracemalloc
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .text_conditioning import PromptEmbeddings


# ══════════════════════════════════════════════════════════════════════════════
# 枚举
# ══════════════════════════════════════════════════════════════════════════════


class CFGMode(enum.Enum):
    """
    CFG（Classifier-Free Guidance）执行模式。

    - SEQUENTIAL: 两次 forward，先 uncond 再 cond，显存低但慢
    - BATCHED: 一次 forward（batch_size × 2），显存高一倍但快
    """

    SEQUENTIAL = "sequential"
    BATCHED = "batched"


# ══════════════════════════════════════════════════════════════════════════════
# LatentBufferManager — 预分配缓冲区池
# ══════════════════════════════════════════════════════════════════════════════


class LatentBufferManager:
    """
    Latent buffer 预分配管理器。

    在构造时预分配 5 个 buffer：
    - x_t: 当前 latent
    - x_next: 下一步 latent
    - noise: 初始噪声（用于 reset）
    - cfg_cond: CFG 条件结果
    - cfg_uncond: CFG 无条件结果

    支持 image shape (B, C, H, W) 或 video shape (B, C, T, H, W)。
    提供 ping-pong swap 和 reset 接口。

    设计说明：
        为什么预分配而不是每步 malloc？
        - PyTorch cudaMalloc 每次分配有固定开销（~10–50 μs）
        - 去噪 loop 可能有 20–50 步，如果每步销毁旧 latent 重新分配，累计开销 ~1–2 ms
        - 预分配 5 个 buffer 后，整个 loop 内零 malloc，显存用量也完全可预测
    """

    def __init__(
        self,
        image_shape: Tuple[int, ...] = (1, 4, 64, 64),
        video_shape: Optional[Tuple[int, ...]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        """
        参数：
            image_shape: 图像 latent shape (B, C, H, W)，默认 (1, 4, 64, 64)。
            video_shape: 视频 latent shape (B, C, T, H, W)，若提供则优先使用。
            device: 设备。
            dtype: 数据类型。
            seed: 噪声生成种子。
        """
        shape = video_shape if video_shape is not None else image_shape
        self._shape = shape
        self._device = device
        self._dtype = dtype
        self._seed = seed

        self._rng = torch.Generator(device=device).manual_seed(seed)

        # 预分配所有 buffer
        self._buffers: Dict[str, torch.Tensor] = {}

        # x_t 和 x_next 初始化为零（会被 noise 覆盖）
        self._buffers["x_t"] = torch.zeros(shape, device=device, dtype=dtype)
        self._buffers["x_next"] = torch.zeros(shape, device=device, dtype=dtype)

        # noise buffer: 初始噪声（用于 reset）
        self._buffers["noise"] = torch.randn(
            shape, generator=self._rng, device=device, dtype=dtype
        )

        # CFG buffer: 分别存储 cond 和 uncond 的 vector field
        self._buffers["cfg_cond"] = torch.zeros(shape, device=device, dtype=dtype)
        self._buffers["cfg_uncond"] = torch.zeros(shape, device=device, dtype=dtype)

    def get(self, name: str) -> torch.Tensor:
        """
        获取指定名称的 buffer 引用。

        参数：
            name: buffer 名称（"x_t" / "x_next" / "noise" / "cfg_cond" / "cfg_uncond"）。

        返回：
            torch.Tensor — buffer 的直接引用（非拷贝）。

        异常：
            KeyError: 若 name 不在已知 buffer 列表中。
        """
        if name not in self._buffers:
            raise KeyError(f"未知 buffer 名称 '{name}'，可用: {list(self._buffers.keys())}")
        return self._buffers[name]

    def swap(self, name1: str, name2: str) -> None:
        """
        Ping-pong 交换两个 buffer。

        在 denoising loop 的第 k 步：
            - 第 k 步更新写入 x_next
            - swap("x_t", "x_next") → x_next 变为 x_t（下一步的输入）
            - 旧 x_t 变为 x_next（下一步的写入目标）

        这避免了 per-step copy。
        """
        if name1 not in self._buffers or name2 not in self._buffers:
            raise KeyError(f"swap 缓冲区名称无效: {name1}, {name2}")
        self._buffers[name1], self._buffers[name2] = (
            self._buffers[name2],
            self._buffers[name1],
        )

    def reset(self) -> None:
        """
        用初始噪声重置所有 buffer。

        重新生成初始噪声并写入 noise buffer，同时重置 x_t 为初始噪声。
        用于多轮推理时重用同一 buffer 池。
        """
        self._rng = torch.Generator(device=self._device).manual_seed(self._seed)
        new_noise = torch.randn(
            self._shape,
            generator=self._rng,
            device=self._device,
            dtype=self._dtype,
        )
        self._buffers["noise"] = new_noise
        self._buffers["x_t"] = new_noise.clone()
        self._buffers["x_next"].zero_()
        self._buffers["cfg_cond"].zero_()
        self._buffers["cfg_uncond"].zero_()

    @property
    def shape(self) -> Tuple[int, ...]:
        """buffer 的 latens shape。"""
        return self._shape

    @property
    def device(self) -> str:
        """buffer 所在设备。"""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """buffer 数据类型。"""
        return self._dtype


# ══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache — prompt embedding 缓存包装
# ══════════════════════════════════════════════════════════════════════════════


class EmbeddingCache:
    """
    Prompt embedding 缓存包装。

    与 text_conditioning 模块的 PromptEmbeddings 配合使用。
    缓存 text encoder 输出，避免同一 prompt 重复编码。

    注意：这不是 LLM 的 KV cache。LLM KV cache 存储 attention key/value 历史
    用于自回归生成中跨步复用；而 embedding cache 存储的是 text encoder 的最终输出
    embedding，在整个 denoising loop 中不变。

    典型使用场景：
        cache = EmbeddingCache()
        emb = text_conditioner.encode("a cat")
        cache.put("main_prompt", emb)
        ...
        # 后续调用
        emb = cache.get("main_prompt")  # hit
    """

    def __init__(self):
        self._store: Dict[str, PromptEmbeddings] = {}

    def get(self, key: str) -> Optional[PromptEmbeddings]:
        """获取缓存的 embedding（若不存在返回 None）。"""
        return self._store.get(key)

    def put(self, key: str, embeddings: PromptEmbeddings) -> None:
        """存入 embedding。"""
        self._store[key] = embeddings

    def contains(self, key: str) -> bool:
        """检查 key 是否存在。"""
        return key in self._store

    def clear(self) -> None:
        """清空所有缓存。"""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


# ══════════════════════════════════════════════════════════════════════════════
# MemoryStats — 显存统计
# ══════════════════════════════════════════════════════════════════════════════


class MemoryStats:
    """
    显存统计接口。

    自动适配三种后端：
    - CUDA: torch.cuda.memory_stats() / memory_allocated() / max_memory_allocated()
    - MPS: torch.mps 提供 limited 统计（无 detailed 接口，fallback 到 CPU）
    - CPU: Python tracemalloc 模块（stdlib）

    暴露指标：
    - peak_allocated: 峰值已分配显存（bytes）
    - peak_reserved: 峰值已保留显存（bytes，仅 CUDA 可用）
    - current_allocated: 当前已分配显存（bytes）
    - allocation_count: 分配次数（仅 CUDA 可用时）

    注意：这些不是 LLM 的"KV cache 占用统计"——这里统计的是扩散模型的
    总 GPU 内存，包括 model weights、latent buffer、text embedding 等。
    """

    def __init__(self):
        self._use_cuda = torch.cuda.is_available() if hasattr(torch, "cuda") else False
        self._use_mps = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        self._use_tracemalloc = not self._use_cuda
        self._tracemalloc_started = False

    def start(self) -> None:
        """开始显存追踪（仅 CPU/MPS 需调用，CUDA 自动追踪）。"""
        if self._use_tracemalloc and not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True

    def stop(self) -> None:
        """停止显存追踪。"""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False

    @property
    def peak_allocated(self) -> int:
        """
        峰值已分配显存（bytes）。

        CUDA: torch.cuda.max_memory_allocated()
        MPS: torch.mps 不支持，返回 0
        CPU: tracemalloc.get_traced_memory()[1]
        """
        if self._use_cuda:
            return torch.cuda.max_memory_allocated()
        if self._tracemalloc_started:
            _current, peak = tracemalloc.get_traced_memory()
            return peak
        return 0

    @property
    def peak_reserved(self) -> int:
        """
        峰值已保留显存（bytes）— 仅 CUDA 支持。

        CUDA: torch.cuda.max_memory_reserved()
        其他: 返回 0
        """
        if self._use_cuda:
            return torch.cuda.max_memory_reserved()
        return 0

    @property
    def current_allocated(self) -> int:
        """
        当前已分配显存（bytes）。

        CUDA: torch.cuda.memory_allocated()
        MPS: torch.mps 不支持，返回 0
        CPU: tracemalloc.get_traced_memory()[0]
        """
        if self._use_cuda:
            return torch.cuda.memory_allocated()
        if self._tracemalloc_started:
            current, _peak = tracemalloc.get_traced_memory()
            return current
        return 0

    @property
    def allocation_count(self) -> int:
        """
        分配次数 — 仅 CUDA 全统计可用。

        CUDA: torch.cuda.memory_stats().get("allocation.all.current", 0)
        其他: 返回 0
        """
        if self._use_cuda:
            try:
                stats = torch.cuda.memory_stats()
                return stats.get("allocation.all.current", 0)
            except Exception:
                return 0
        return 0

    def snapshot(self) -> Dict[str, int]:
        """
        返回当前显存快照。

        返回：
            {
                "peak_allocated": int,
                "peak_reserved": int,
                "current_allocated": int,
                "allocation_count": int,
                "backend": str,  # "cuda" / "mps" / "cpu"
            }
        """
        return {
            "peak_allocated": self.peak_allocated,
            "peak_reserved": self.peak_reserved,
            "current_allocated": self.current_allocated,
            "allocation_count": self.allocation_count,
            "backend": "cuda" if self._use_cuda else ("mps" if self._use_mps else "cpu"),
        }

    def __repr__(self) -> str:
        s = self.snapshot()
        return (
            f"MemoryStats(backend={s['backend']}, "
            f"peak_alloc={s['peak_allocated'] / 1024**2:.1f}MB, "
            f"peak_reserved={s['peak_reserved'] / 1024**2:.1f}MB, "
            f"cur_alloc={s['current_allocated'] / 1024**2:.1f}MB)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数：估算 latent buffer 显存
# ══════════════════════════════════════════════════════════════════════════════


def estimate_latent_buffer_bytes(
    image_shape: Tuple[int, ...] = (1, 4, 64, 64),
    dtype_bytes: int = 4,
    num_buffers: int = 5,
) -> int:
    """
    估算 latent buffer 池的显存占用。

    参数：
        image_shape: latent shape (B, C, H, W)。
        dtype_bytes: 每个元素的字节数（float32=4, float16=2, bfloat16=2）。
        num_buffers: buffer 数量（默认 5）。

    返回：
        总 bytes。
    """
    total_elements = 1
    for d in image_shape:
        total_elements *= d
    return total_elements * dtype_bytes * num_buffers
