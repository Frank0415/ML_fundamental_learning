#!/usr/bin/env python3
"""
latent_buffer_manager.py — Latent Buffer 预分配管理器实验

★★★ Diffusion 主优化不是 LLM KV cache，但 latent buffer 预分配是真实收益 ★★★

- LLM 的 KV cache 存储的是 attention key/value 历史，用于自回归跨 token 复用。
  扩散 denoising 每步 latent 全刷新——上一步的 K/V 没有复用价值。
- 但 latent buffer 预分配是一个真实的优化点：
  去噪循环可能有 20–50 步，如果每步 malloc/free 旧 latent 重新分配，
  累计碎片化和分配开销不可忽略。预分配 4–5 个 buffer 后，整个 loop 内零 malloc。

本脚本实现：
  1. LatentBufferShape：@dataclass，支持 image/video/tokens 三种 shape
  2. LatentBufferManager：构造时预分配，提供 get/swap/reset/out_of_place_reset/stats
  3. --demo 模式：比较 in-place reset vs out-of-place reset 的 28 步推理

核心对比：
  - A. in-place reset：每次 step 在预分配 buffer 上直接覆盖——零额外分配
  - B. out-of-place reset：每次 step 分配新 buffer → 旧 buffer 被 GC——大量 malloc/free

在受限显存配置下的真实占比（1024² latent, fp16, 16ch）：
  - 5 buffers × 4MB = 20MB — 可忽略
  - 真正瓶颈在 attention activations：4096² × 16 heads × 4B ≈ 1 GB/layer

纯 numpy 实现，不依赖 torch。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# LatentBufferShape — shape 表示
# =============================================================================


@dataclass
class LatentBufferShape:
    """
    扩散模型 latent buffer 的 shape 表示。

    支持三种模式：
    - image:  (B, C, H, W)
    - video:  (B, C, T, H, W)
    - tokens: (B, N, D)  — patch 化后的 token

    使用示例：
        shape = LatentBufferShape.image(1, 16, 128, 128)
        print(shape.nbytes(dtype="float16"))  # → 524288
    """

    B: int = 1
    C: int = 4
    H: int = 64
    W: int = 64
    T: int = 1         # 视频帧数（image 时忽略）
    N: int = 256       # token 模式下的 token 数
    D: int = 64        # token 模式下的维度
    mode: str = "image"  # "image" / "video" / "tokens"

    @classmethod
    def image(cls, B: int, C: int, H: int, W: int) -> "LatentBufferShape":
        """创建 image shape。"""
        return cls(B=B, C=C, H=H, W=W, mode="image")

    @classmethod
    def video(cls, B: int, C: int, T: int, H: int, W: int) -> "LatentBufferShape":
        """创建 video shape。"""
        return cls(B=B, C=C, T=T, H=H, W=W, mode="video")

    @classmethod
    def tokens(cls, B: int, N: int, D: int) -> "LatentBufferShape":
        """创建 tokens shape（patch 化后）。"""
        return cls(B=B, N=N, D=D, mode="tokens")

    def to_tuple(self) -> Tuple[int, ...]:
        """转换为 numpy 可用的 shape tuple。"""
        if self.mode == "image":
            return (self.B, self.C, self.H, self.W)
        elif self.mode == "video":
            return (self.B, self.C, self.T, self.H, self.W)
        else:
            return (self.B, self.N, self.D)

    def numel(self) -> int:
        """总元素数。"""
        t = self.to_tuple()
        result = 1
        for d in t:
            result *= d
        return result

    def nbytes(self, dtype: str = "float32") -> int:
        """
        以指定 dtype 存储时占用的字节数。

        参数：
            dtype: "float32" → 4 bytes/elem, "float16" / "bfloat16" → 2 bytes/elem。

        返回：
            总字节数。
        """
        bytes_per_elem = 2 if dtype in ("float16", "bfloat16") else 4
        return self.numel() * bytes_per_elem

    def __repr__(self) -> str:
        return (
            f"LatentBufferShape({self.mode}: {self.to_tuple()}, "
            f"{self.numel():,} elements)"
        )


# =============================================================================
# LatentBufferManager — 预分配管理器
# =============================================================================


class LatentBufferManager:
    """
    扩散模型 latent buffer 预分配管理器。

    在构造时预分配固定数量的 buffer（x_t, x_next, noise + 可选的临时 buffer），
    整个 denoising loop 内零 malloc。提供 ping-pong swap 和 in-place reset。

    与 LLM KV cache 的区别：
      - KV cache 存储过去 token 的 key/value，用于自回归跨步复用
      - latent buffer 存储当前步的完整 latent，每步被新 latent 覆盖
      - latent buffer 的价值在消除 malloc 碎片化，不在跨步复用数据

    典型使用模式：
        mgr = LatentBufferManager(LatentBufferShape.image(1, 4, 64, 64))
        x_t = mgr.get("x_t")      # 当前 latent（零拷贝 view）
        x_next = mgr.get("x_next") # 下一步 latent
        # 每步推理后：
        mgr.swap("x_t", "x_next")  # ping-pong
        # 新一轮推理：
        mgr.reset("x_t")           # in-place 重新初始化为噪声
    """

    def __init__(
        self,
        shape: LatentBufferShape,
        dtype: str = "float32",
        seed: int = 42,
        num_temp_buffers: int = 2,
    ):
        """
        参数：
            shape: latent buffer 的维度（image / video / tokens）。
            dtype: 数据类型（"float32" / "float16"）。
            seed: 初始噪声随机种子。
            num_temp_buffers: 临时 buffer 数量（用于 CFG 中间结果等，默认 2）。
        """
        self._shape = shape
        self._dtype = dtype
        self._seed = seed
        self._np_dtype = np.float32 if dtype == "float32" else np.float16
        self._rng = np.random.RandomState(seed)

        # 统计计数器
        self._allocation_count: int = 0
        self._peak_allocated_bytes: int = 0
        self._current_allocated_bytes: int = 0
        self._swap_count: int = 0
        self._reset_count: int = 0

        # 预分配 buffer 字典
        self._buffers: Dict[str, np.ndarray] = {}
        shape_tuple = shape.to_tuple()

        # 核心 buffer：x_t（当前 latent）、x_next（下一步 latent）
        self._buffers["x_t"] = self._allocate(shape_tuple, "x_t")
        self._buffers["x_next"] = self._allocate(shape_tuple, "x_next")

        # 噪声 buffer：初始噪声（用于 reset）
        self._buffers["noise"] = self._allocate(shape_tuple, "noise")
        self._rng = np.random.RandomState(seed)  # 重置 RNG 保证确定性

        # 临时 buffer（CFG 中间结果等）
        for idx in range(num_temp_buffers):
            name = f"temp_{idx}"
            self._buffers[name] = self._allocate(shape_tuple, name)

    def _allocate(self, shape: Tuple[int, ...], name: str) -> np.ndarray:
        """
        分配 numpy 数组并更新统计。

        参数：
            shape: numpy shape tuple。
            name: buffer 名称（仅用于日志）。

        返回：
            全零的 numpy 数组。
        """
        arr = np.zeros(shape, dtype=self._np_dtype)
        nbytes = arr.nbytes
        self._allocation_count += 1
        self._current_allocated_bytes += nbytes
        self._peak_allocated_bytes = max(self._peak_allocated_bytes, self._current_allocated_bytes)
        return arr

    def get(self, name: str) -> np.ndarray:
        """
        获取指定名称的 buffer 引用。

        返回的是直接引用（非拷贝）——对返回数组的 in-place 修改会作用于 buffer 池。

        参数：
            name: buffer 名称（"x_t" / "x_next" / "noise" / "temp_0" 等）。

        返回：
            numpy ndarray 的直接 view。

        异常：
            KeyError: 若 name 不在已知 buffer 列表中。
        """
        if name not in self._buffers:
            raise KeyError(
                f"未知 buffer '{name}'，可用: {list(self._buffers.keys())}"
            )
        return self._buffers[name]

    def swap(self, name1: str, name2: str) -> None:
        """
        Ping-pong 交换两个 buffer。

        在 denoising loop 的第 k 步：
          - 第 k 步的更新写入 x_next
          - swap("x_t", "x_next") → x_next 成为下一步输入，旧 x_t 成为写入目标
          - 避免 per-step tensor copy

        参数：
            name1: 第一个 buffer 名称。
            name2: 第二个 buffer 名称。
        """
        if name1 not in self._buffers:
            raise KeyError(f"swap: 未知 buffer '{name1}'")
        if name2 not in self._buffers:
            raise KeyError(f"swap: 未知 buffer '{name2}'")
        self._buffers[name1], self._buffers[name2] = (
            self._buffers[name2],
            self._buffers[name1],
        )
        self._swap_count += 1

    def reset(self, name: str, generator: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        In-place reset：用新噪声覆盖指定 buffer。

        这是"预分配 + 写入"模式——不分配新内存，直接在已有 buffer 上写入噪声。
        典型用法：
            mgr.reset("x_t")  # x_t = 新噪声

        参数：
            name: 目标 buffer 名称。
            generator: 可选的自定义 RandomState（None 时用内部 seed 新生成）。

        返回：
            被写入的 buffer 引用（可直接使用）。
        """
        if name not in self._buffers:
            raise KeyError(f"reset: 未知 buffer '{name}'")
        rng = generator if generator is not None else np.random.RandomState(self._seed + self._reset_count)
        noise = rng.randn(*self._shape.to_tuple()).astype(self._np_dtype)
        # In-place 写入——不分配新数组
        self._buffers[name][:] = noise
        self._reset_count += 1
        return self._buffers[name]

    def out_of_place_reset(
        self, name: str, generator: Optional[np.random.RandomState] = None
    ) -> np.ndarray:
        """
        Out-of-place reset：分配新 buffer 替代旧 buffer（对照组）。

        与 in-place reset 的区别：
          - in-place: 在已有 buffer 上直接覆盖 → 零 malloc
          - out-of-place: 分配新数组 → 旧数组成为垃圾 → 增加 GC 压力

        参数：
            name: 目标 buffer 名称。
            generator: 可选的自定义 RandomState。

        返回：
            新分配的 numpy 数组。
        """
        rng = generator if generator is not None else np.random.RandomState(self._seed + self._reset_count)
        new_arr = rng.randn(*self._shape.to_tuple()).astype(self._np_dtype)
        # 替换旧 buffer：旧 buffer 失去引用，成为垃圾
        old_arr = self._buffers[name]
        self._buffers[name] = new_arr
        self._allocation_count += 1
        self._current_allocated_bytes += new_arr.nbytes - old_arr.nbytes
        self._peak_allocated_bytes = max(self._peak_allocated_bytes, self._current_allocated_bytes)
        return new_arr

    def stats(self) -> dict:
        """
        返回 buffer 管理器统计。

        返回：
            {
                "allocation_count": int,       # 总分配次数
                "current_allocated_bytes": int, # 当前已分配字节
                "peak_allocated_bytes": int,    # 峰值已分配字节
                "peak_reserved_bytes": int,     # 峰值预留字节（≈ peak_allocated，无 numpy 等价 reserved 概念）
                "swap_count": int,               # ping-pong 次数
                "reset_count": int,              # reset 次数
                "num_buffers": int,              # buffer 池中 buffer 数
                "fragmentation_estimate": float,  # 碎片化估计（gc get_objects 成本高，用 approx）
                "shape": str,                    # latent shape 字符串
                "dtype": str,                    # 数据类型
                "buffer_sizes_bytes": dict,      # 每个 buffer 的字节数
            }
        """
        buffer_sizes = {
            name: arr.nbytes for name, arr in self._buffers.items()
        }
        # 碎片化估计：当前分配 / 峰值分配 - 1 = "浪费比例"
        # 如果持续 out-of-place reset，这个值会显著升高
        total_now = sum(buffer_sizes.values())
        fragmentation = max(0.0, abs(total_now - self._peak_allocated_bytes) / max(1, self._peak_allocated_bytes))

        return {
            "allocation_count": self._allocation_count,
            "current_allocated_bytes": self._current_allocated_bytes,
            "peak_allocated_bytes": self._peak_allocated_bytes,
            "peak_reserved_bytes": self._peak_allocated_bytes,  # numpy 无 reserved 概念，近似为 allocated
            "swap_count": self._swap_count,
            "reset_count": self._reset_count,
            "num_buffers": len(self._buffers),
            "fragmentation_estimate": round(fragmentation, 6),
            "shape": str(self._shape.to_tuple()),
            "dtype": self._dtype,
            "buffer_sizes_bytes": buffer_sizes,
        }

    def clear(self) -> None:
        """清空所有 buffer（释放显存/内存）。"""
        self._buffers.clear()
        self._allocation_count = 0
        self._current_allocated_bytes = 0
        self._swap_count = 0
        self._reset_count = 0


# =============================================================================
# Demo 运行器
# =============================================================================


def run_inplace_demo(
    shape: LatentBufferShape,
    num_steps: int = 28,
    dtype: str = "float32",
    seed: int = 42,
) -> dict:
    """
    运行 in-place reset 的 28 步推理模拟。

    每步流程：
      1. mgr.get("x_t") 获取当前 latent
      2. mgr.get("x_next") 获取写入目标
      3. 模拟 denoiser forward（简单 numpy 计算 + sleep）
      4. mgr.swap("x_t", "x_next") 完成 ping-pong
      5. mgr.reset("noise") 可选（in-place 刷新噪声）

    参数：
        shape: latent buffer shape。
        num_steps: 推理步数。
        dtype: 数据类型。
        seed: 随机种子。

    返回：
        包含指标的结果字典。
    """
    mgr = LatentBufferManager(shape, dtype=dtype, seed=seed)

    step_latencies: List[float] = []
    rng = np.random.RandomState(seed)

    # 初始化：将噪声写入 x_t
    noise = rng.randn(*shape.to_tuple()).astype(np.float32 if dtype == "float32" else np.float16)
    mgr.get("x_t")[:] = noise

    stats_before = mgr.stats()

    for step in range(num_steps):
        start = time.perf_counter()

        x_t = mgr.get("x_t")
        x_next = mgr.get("x_next")

        # 模拟 denoiser forward：简单矩阵操作（模拟模型前向的计算量）
        # 实际 DiT 会做 attention + MLP，这里用 sin + cos 近似计算密度
        _denoised = np.sin(x_t * 0.1) * 0.5 + np.cos(x_t * 0.05) * 0.3
        x_next[:] = _denoised  # in-place 写入

        # ping-pong
        mgr.swap("x_t", "x_next")

        # in-place reset noise（备用 noise buffer 刷新）
        mgr.reset("noise")

        elapsed = time.perf_counter() - start
        step_latencies.append(elapsed)

    stats_after = mgr.stats()

    return {
        "mode": "in_place_reset",
        "num_steps": num_steps,
        "total_latency_s": round(sum(step_latencies), 4),
        "avg_latency_per_step_ms": round(np.mean(step_latencies) * 1000, 3),
        "min_latency_per_step_ms": round(np.min(step_latencies) * 1000, 3),
        "max_latency_per_step_ms": round(np.max(step_latencies) * 1000, 3),
        "allocation_count": stats_after["allocation_count"],
        "peak_allocated_bytes": stats_after["peak_allocated_bytes"],
        "peak_reserved_bytes": stats_after["peak_reserved_bytes"],
        "current_allocated_bytes": stats_after["current_allocated_bytes"],
        "fragmentation_estimate": stats_after["fragmentation_estimate"],
        "swap_count": stats_after["swap_count"],
        "reset_count": stats_after["reset_count"],
        "step_latencies_ms": [round(l * 1000, 3) for l in step_latencies],
    }


def run_outofplace_demo(
    shape: LatentBufferShape,
    num_steps: int = 28,
    dtype: str = "float32",
    seed: int = 42,
) -> dict:
    """
    运行 out-of-place reset 的 28 步推理模拟（对照组）。

    与 in-place 的区别：
      - 每步调用 out_of_place_reset 分配新 buffer
      - 旧 buffer 被丢弃（模拟 naive malloc/free 模式）
      - 不预分配 x_next 和 noise，每步动态分配

    参数：
        shape: latent buffer shape。
        num_steps: 推理步数。
        dtype: 数据类型。
        seed: 随机种子。

    返回：
        包含指标的结果字典。
    """
    mgr = LatentBufferManager(shape, dtype=dtype, seed=seed)

    step_latencies: List[float] = []
    rng = np.random.RandomState(seed)

    # 初始化
    noise = rng.randn(*shape.to_tuple()).astype(np.float32 if dtype == "float32" else np.float16)
    mgr.get("x_t")[:] = noise

    stats_before = mgr.stats()

    for step in range(num_steps):
        start = time.perf_counter()

        x_t = mgr.get("x_t")

        # 模拟 denoiser forward
        _denoised = np.sin(x_t * 0.1) * 0.5 + np.cos(x_t * 0.05) * 0.3

        # out-of-place：分配新 buffer 替代 x_t
        mgr.out_of_place_reset("x_t")

        # ping-pong（仍然需要 swap，但目标 buffer 是新的）
        mgr.swap("x_t", "x_next")

        # out-of-place reset noise
        mgr.out_of_place_reset("noise")

        elapsed = time.perf_counter() - start
        step_latencies.append(elapsed)

    stats_after = mgr.stats()

    return {
        "mode": "out_of_place_reset",
        "num_steps": num_steps,
        "total_latency_s": round(sum(step_latencies), 4),
        "avg_latency_per_step_ms": round(np.mean(step_latencies) * 1000, 3),
        "min_latency_per_step_ms": round(np.min(step_latencies) * 1000, 3),
        "max_latency_per_step_ms": round(np.max(step_latencies) * 1000, 3),
        "allocation_count": stats_after["allocation_count"],
        "peak_allocated_bytes": stats_after["peak_allocated_bytes"],
        "peak_reserved_bytes": stats_after["peak_reserved_bytes"],
        "current_allocated_bytes": stats_after["current_allocated_bytes"],
        "fragmentation_estimate": stats_after["fragmentation_estimate"],
        "swap_count": stats_after["swap_count"],
        "reset_count": stats_after["reset_count"],
        "step_latencies_ms": [round(l * 1000, 3) for l in step_latencies],
    }


def run_demo(
    num_steps: int = 28,
    image_shape: Tuple[int, ...] = (1, 4, 64, 64),
    dtype: str = "float32",
    output_dir: str = "results",
    seed: int = 42,
) -> dict:
    """
    运行完整的 latent buffer manager demo：in-place vs out-of-place 对比。

    参数：
        num_steps: 模拟的 denoising 步数。
        image_shape: latent shape。
        dtype: 数据类型。
        output_dir: 结果输出目录。
        seed: 随机种子。

    返回：
        包含两组实验结果的字典。
    """
    shape = LatentBufferShape(
        B=image_shape[0],
        C=image_shape[1] if len(image_shape) >= 2 else 4,
        H=image_shape[2] if len(image_shape) >= 3 else 64,
        W=image_shape[3] if len(image_shape) >= 4 else 64,
        mode="image",
    )

    print(f"\n  Shape: {shape}")
    print(f"  Dtype: {dtype}")
    print(f"  Steps: {num_steps}")
    print(f"  Per-buffer size: {shape.nbytes(dtype):,} bytes")
    print()

    # 运行 in-place demo
    print("  运行 in-place reset 实验...")
    inplace = run_inplace_demo(shape, num_steps, dtype, seed)

    # 运行 out-of-place demo
    print("  运行 out-of-place reset 实验...")
    outofplace = run_outofplace_demo(shape, num_steps, dtype, seed)

    # 计算差异
    alloc_diff = outofplace["allocation_count"] - inplace["allocation_count"]
    peak_diff = outofplace["peak_allocated_bytes"] - inplace["peak_allocated_bytes"]
    latency_diff = outofplace["total_latency_s"] - inplace["total_latency_s"]

    results = {
        "experiment": "latent_buffer_manager",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "shape": str(shape.to_tuple()),
            "mode": shape.mode,
            "numel": shape.numel(),
            "bytes_per_buffer": shape.nbytes(dtype),
            "dtype": dtype,
            "num_steps": num_steps,
        },
        "in_place_reset": inplace,
        "out_of_place_reset": outofplace,
        "comparison": {
            "allocation_count_diff": alloc_diff,
            "allocation_count_reduction_pct": (
                round(alloc_diff / outofplace["allocation_count"] * 100, 1)
                if outofplace["allocation_count"] > 0
                else 0
            ),
            "peak_allocated_diff_bytes": peak_diff,
            "peak_allocated_diff_mb": round(peak_diff / 1024**2, 2),
            "latency_diff_ms": round(latency_diff * 1000, 3),
            "fragmentation_in_place": inplace["fragmentation_estimate"],
            "fragmentation_out_of_place": outofplace["fragmentation_estimate"],
        },
    }

    return results


# =============================================================================
# 命令行接口
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "Latent Buffer Manager 实验 — 比较 in-place vs out-of-place reset，"
            "测量预分配 buffer 池的碎片化和分配优化收益。\n"
            "★★★ 这是 diffusion 的 latent buffer 预分配，不是 LLM 的 KV cache ★★★\n"
            "Latent buffer 解决的是 malloc 碎片化，不是跨步数据复用。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
========== 示例 ==========

# 默认 demo（28 步，image shape 1,4,64,64）
python latent_buffer_manager.py --demo

# 自定义 shape 和步数
python latent_buffer_manager.py --demo --num_steps 50 --image_shape 1 4 128 128

# 使用 float16 dtype（节省内存）
python latent_buffer_manager.py --demo --dtype float16

# 视频 shape demo
python latent_buffer_manager.py --demo --num_steps 28 --mode video --video_shape 1 4 16 32 32

# 仅验证 buffer 管理 API（不跑 demo）
python -c "
import numpy as np
from latent_buffer_manager import LatentBufferShape, LatentBufferManager
shape = LatentBufferShape.image(1, 4, 64, 64)
mgr = LatentBufferManager(shape)
mgr.reset('x_t')
print(mgr.stats())
"
""",
    )

    # === 运行模式 ===
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行 demo 实验：比较 in-place 和 out-of-place reset 的 28 步推理表现",
    )

    # === Demo 参数 ===
    demo_group = parser.add_argument_group("Demo 参数")
    demo_group.add_argument(
        "--num_steps",
        type=int,
        default=28,
        help="模拟的 denoising 步数（默认 28）",
    )
    demo_group.add_argument(
        "--image_shape",
        type=int,
        nargs=4,
        default=[1, 4, 64, 64],
        metavar=("B", "C", "H", "W"),
        help="图像 latent shape（B C H W，默认 1 4 64 64）",
    )
    demo_group.add_argument(
        "--video_shape",
        type=int,
        nargs=5,
        default=None,
        metavar=("B", "C", "T", "H", "W"),
        help="视频 latent shape（B C T H W），提供则覆盖 image_shape",
    )
    demo_group.add_argument(
        "--mode",
        type=str,
        choices=["image", "video", "tokens"],
        default="image",
        help="latent 模式（默认 image）",
    )
    demo_group.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="数据类型（默认 float32）",
    )

    # === 输出 ===
    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="结果输出目录（默认 results）",
    )
    output_group.add_argument(
        "--no_save",
        action="store_true",
        help="不保存结果文件，仅打印到 stdout",
    )

    # === 杂项 ===
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )

    return parser


def main() -> None:
    """主入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        print("\n提示：使用 --demo 运行实验，或 --help 查看完整帮助。")
        sys.exit(0)

    # 创建输出目录
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  Latent Buffer Manager 实验")
    print("  ★ Latent buffer 预分配 ≠ LLM KV cache ★")
    print("=" * 72)
    print(f"  参数: num_steps={args.num_steps}, dtype={args.dtype}, seed={args.seed}")
    print(f"  Buffer 管理: 预分配、ping-pong swap、in-place reset")

    # 构造 shape
    if args.video_shape is not None:
        shape_tuple = tuple(args.video_shape)
        print(f"  视频 shape: {shape_tuple}")
    elif args.mode == "tokens":
        shape_tuple = tuple(args.image_shape[:1]) + (args.image_shape[2] * args.image_shape[3], args.image_shape[1])
        print(f"  Tokens shape: {shape_tuple}")
    else:
        shape_tuple = tuple(args.image_shape)
        print(f"  图像 shape: {shape_tuple}")

    # 运行 demo
    results = run_demo(
        num_steps=args.num_steps,
        image_shape=shape_tuple,
        dtype=args.dtype,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # 打印结果
    print()
    print("─" * 72)
    print("  对比结果")
    print("─" * 72)
    inplace = results["in_place_reset"]
    outofplace = results["out_of_place_reset"]
    comp = results["comparison"]

    print(f"  In-place reset:")
    print(f"    每步平均延迟: {inplace['avg_latency_per_step_ms']:.3f} ms")
    print(f"    总分配次数:   {inplace['allocation_count']}")
    print(f"    峰值分配:     {inplace['peak_allocated_bytes']:,} bytes ({inplace['peak_allocated_bytes']/1024:.1f} KB)")
    print(f"    碎片化估计:   {inplace['fragmentation_estimate']:.6f}")
    print()
    print(f"  Out-of-place reset:")
    print(f"    每步平均延迟: {outofplace['avg_latency_per_step_ms']:.3f} ms")
    print(f"    总分配次数:   {outofplace['allocation_count']}")
    print(f"    峰值分配:     {outofplace['peak_allocated_bytes']:,} bytes ({outofplace['peak_allocated_bytes']/1024:.1f} KB)")
    print(f"    碎片化估计:   {outofplace['fragmentation_estimate']:.6f}")
    print()
    print(f"  节省:")
    print(f"    分配次数减少: {comp['allocation_count_diff']} ({comp['allocation_count_reduction_pct']}%)")
    print(f"    峰值分配差异: {comp['peak_allocated_diff_mb']} MB")
    print(f"    延迟差异:     {comp['latency_diff_ms']:.3f} ms")

    # 保存结果
    if not args.no_save:
        timestamp = results["timestamp"]
        json_path = os.path.join(args.output_dir, f"latent_buffer_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {json_path}")

    print()
    print("=" * 72)
    print("  实验完成。")
    print("=" * 72)


if __name__ == "__main__":
    main()
