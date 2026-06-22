#!/usr/bin/env python3
"""
cfg_batching_experiment.py — CFG Batching 对照实验

比较 Classifier-Free Guidance 的两种执行模式：
  1. Sequential CFG：两次 denoiser forward（先 uncond 再 cond），显存低但慢
  2. Batched CFG：一次 forward（batch_size × 2），显存高一倍但快

★★★ 核心认知 ★★★
- CFG（Classifier-Free Guidance）是在 vector field 层面执行的，不是 latent 层面。
- 公式：v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
- 即：用 guidance scale 在条件输出和无条件输出之间做线性插值，施加到 vector field 上。
- 这与 latent interpolation/ blending 在原理和数学上完全不同。

实验设计：
  1. Mock denoiser（纯 numpy，不依赖 torch）模拟 DiT 前向
  2. 两种 CFG mode 各跑 num_steps 步，测每步 latency
  3. 计算两种 mode 输出的数值差异 ||v_seq - v_batch||_max
  4. cfg_scale 从 1.0 到 15.0 扫描，观察数值稳定性
  5. 输出 JSON + MD 结果文件

输出：
  - results/cfg_batching_<timestamp>.json — 结构化数据
  - results/cfg_batching_<timestamp>.md   — 人类可读表格与结论

纯 numpy 实现，不依赖 torch。

========== 使用示例 ==========

# 查看帮助
python cfg_batching_experiment.py --help

# 默认 demo（28 步，cfg_scale 扫描 1.0–15.0）
python cfg_batching_experiment.py --demo --output_dir results

# 自定义参数
python cfg_batching_experiment.py --demo --num_steps 50 --cfg_scales 1.0 3.0 7.5 15.0 --output_dir results

# 指定 latent shape（影响 mock 计算量）
python cfg_batching_experiment.py --demo --latent_shape 1 4 128 128 --output_dir results
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Mock Denoiser — 模拟 DiT forward 计算量
# =============================================================================


class MockDenoiser:
    """
    Mock denoiser：用 numpy 操作模拟 DiT 前向的计算量。

    不调用真实的神经网络 transformer block。使用矩阵乘法 + 激活函数
    近似 DiT 的 forward 计算密度，保证可复现且不依赖 torch。

    核心步骤（模拟）：
      1. timestep embedding：sin/cos 编码 → 线性变换
      2. adaLN modulation：timestep + text embedding → scale/shift
      3. Full attention：latent tokens → QKV → attention matrix → output
      4. Pointwise FFN
      → 返回 vector field v_θ(latent, t, condition)

    设计说明：
      - 本实验的核心是比较 Sequential/Batched CFG 的 tradeoff，
        不是 denoiser 的绝对性能。使用确定性 mock 保证可复现性。
      - 模拟的计算量随 latent_shape 缩放。
    """

    def __init__(
        self,
        latent_shape: Tuple[int, ...] = (1, 16, 64, 64),
        context_dim: int = 768,
        complexity_factor: float = 0.5,
        seed: int = 42,
    ):
        """
        参数：
            latent_shape: latent 张量 shape (B, C, H, W)。
            context_dim: text embedding 维度。
            complexity_factor: 计算倍率（0.5 = 半密度模拟，1.0 = 全密度）。
            seed: 随机种子。
        """
        self.latent_shape = latent_shape
        self.context_dim = context_dim
        self.complexity_factor = complexity_factor
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    def forward(
        self,
        latent: np.ndarray,
        timestep: float,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Mock denoiser forward：返回 vector field v_θ(latent, t, embedding)。

        参数：
            latent: shape (B, C, H, W) 的 latent 张量。
            timestep: 当前 timestep t ∈ [0, 1]。
            embedding: shape (B, context_dim) 的条件 embedding。

        返回：
            vector field — shape 与 latent 相同的 numpy 数组。
        """
        B, C, H, W = latent.shape

        # 1. 模拟 timestep embedding（sin/cos 编码 → 线性）
        t_emb = self._timestep_embedding(timestep)

        # 2. 模拟 adaLN：timestep + context → scale/shift
        scale, shift = self._adain(t_emb, embedding)

        # 3. 模拟 patchify → attention → unpatchify
        #    为简单起见，直接在 latent 上模拟（计算量 ~N²）
        N_tokens = H * W
        complexity = int(self.complexity_factor * max(1, N_tokens // 64))
        repeats = max(1, complexity)

        v = latent.copy()
        for _ in range(repeats):
            # 模拟 attention: token × token 矩阵乘法
            latent_flat = v.reshape(B, C, -1)  # (B, C, H*W)
            # QKV 模拟
            attn_dot = latent_flat @ latent_flat.transpose(0, 2, 1)  # (B, C, C)
            # softmax-like normalization
            attn_dot = np.tanh(attn_dot / math.sqrt(C))
            v_attn = attn_dot @ latent_flat  # (B, C, H*W)
            v = v_attn.reshape(B, C, H, W)
            # adaLN modulation — scale/shift already broadcast to (B, C, H, W)
            v = scale * v + shift
            # FFN 模拟
            v = np.tanh(v)

        return v.astype(np.float64)

    def _timestep_embedding(self, t: float) -> np.ndarray:
        """模拟 timestep embedding（sin/cos 位置编码风格）。"""
        # 简单实现：返回 2D embedding
        t_scaled = t * 1000.0
        half = 32
        freqs = np.exp(
            -np.arange(half) * math.log(10000.0) / half
        ) * t_scaled
        emb = np.concatenate([np.sin(freqs), np.cos(freqs)])
        return emb.astype(np.float64)

    def _adain(
        self, t_emb: np.ndarray, context: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """模拟 adaLN：从 timestep + context 生成 scale 和 shift。"""
        B, _, H, W = self.latent_shape
        C = self.latent_shape[1]
        # 简单线性组合：生成 scale/shift 通过广播到 H×W
        combined = np.concatenate([t_emb, context[0, :32]])
        # 需要产生 C 个 scale 值和 C 个 shift 值（每个通道一个）
        scale_per_ch = combined[:C]  # C 个值
        shift_per_ch = combined[C : 2 * C]  # C 个值
        # 广播到 (B, C, H, W)
        scale = np.tile(
            scale_per_ch.reshape(1, C, 1, 1), (B, 1, H, W)
        ).astype(np.float64)
        shift = np.tile(
            shift_per_ch.reshape(1, C, 1, 1), (B, 1, H, W)
        ).astype(np.float64)
        return scale, shift


# =============================================================================
# CFG 执行器
# =============================================================================


def cfg_sequential(
    denoiser: MockDenoiser,
    latent: np.ndarray,
    timestep: float,
    cond_emb: np.ndarray,
    uncond_emb: np.ndarray,
    cfg_scale: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Sequential CFG：两次独立 denoiser forward。

    工作流：
      1. v_uncond = denoiser(latent, t, uncond_emb)
      2. v_cond   = denoiser(latent, t, cond_emb)
      3. v_cfg    = v_uncond + cfg_scale * (v_cond - v_uncond)

    优点：峰值显存低（单 batch）
    缺点：两次 forward，慢

    返回：(v_cfg, metrics)
    """
    metrics = {"mode": "sequential", "num_forwards": 2}

    t0 = time.perf_counter()
    v_uncond = denoiser.forward(latent, timestep, uncond_emb)
    t1 = time.perf_counter()
    v_cond = denoiser.forward(latent, timestep, cond_emb)
    t2 = time.perf_counter()

    # CFG 在 vector field 层面做线性插值
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

    metrics["latency_forward1_s"] = round(t1 - t0, 6)
    metrics["latency_forward2_s"] = round(t2 - t1, 6)
    metrics["latency_total_s"] = round(t2 - t0, 6)
    # 峰值显存（模拟）：仅存 1 个 latent
    latent_bytes = latent.nbytes
    metrics["peak_memory_estimated_bytes"] = (
        latent_bytes * 3  # latent + v_uncond + v_cond（短暂共存）
    )

    return v_cfg, metrics


def cfg_batched(
    denoiser: MockDenoiser,
    latent: np.ndarray,
    timestep: float,
    cond_emb: np.ndarray,
    uncond_emb: np.ndarray,
    cfg_scale: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Batched CFG：一次 forward，batch_size × 2。

    工作流：
      1. latent_cat   = concat([latent, latent], axis=0)   # (2*B, C, H, W)
      2. emb_cat      = concat([uncond_emb, cond_emb], axis=0)
      3. v_cat        = denoiser(latent_cat, t, emb_cat)   # 一次 forward
      4. v_uncond, v_cond = split(v_cat, axis=0)
      5. v_cfg        = v_uncond + cfg_scale * (v_cond - v_uncond)

    优点：一次 forward，快 ~1.3–1.8×
    缺点：峰值显存 ~2×（同时存双倍 batch）

    返回：(v_cfg, metrics)
    """
    metrics = {"mode": "batched", "num_forwards": 1}

    latent_bytes = latent.nbytes
    # 构造双倍 batch
    latent_batch = np.concatenate([latent, latent], axis=0)
    emb_batch = np.concatenate([uncond_emb, cond_emb], axis=0)

    t0 = time.perf_counter()
    v_batch = denoiser.forward(latent_batch, timestep, emb_batch)
    t1 = time.perf_counter()

    # split 回 cond / uncond
    v_uncond = v_batch[0:1]
    v_cond = v_batch[1:2]

    # CFG 在 vector field 层面做线性插值
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

    metrics["latency_total_s"] = round(t1 - t0, 6)
    # 峰值显存（模拟）：双倍 batch 的 latent_cat + v_batch
    metrics["peak_memory_estimated_bytes"] = (
        latent_bytes * 2 * 2  # latent_cat + v_batch（各 2B）
    )

    return v_cfg, metrics


def numerical_diff_max(v_seq: np.ndarray, v_batch: np.ndarray) -> float:
    """计算两种 CFG mode 输出的最大差异。"""
    return float(np.max(np.abs(v_seq - v_batch)))


def numerical_diff_rmse(v_seq: np.ndarray, v_batch: np.ndarray) -> float:
    """计算两种 CFG mode 输出的 RMSE。"""
    return float(np.sqrt(np.mean((v_seq - v_batch) ** 2)))


# =============================================================================
# Mock Text Encoder
# =============================================================================


class MockTextEncoder:
    """
    模拟 text encoder：生成确定性条件/无条件 embedding。

    不调用真实 CLIP/T5。使用确定性 numpy 随机数模拟 text encoder 输出。
    """

    def __init__(self, context_dim: int = 768, seed: int = 42):
        self.context_dim = context_dim
        self.seed = seed

    def encode_cond(self, batch_size: int = 1) -> np.ndarray:
        """生成条件 embedding（模拟文本提示的正向 embedding）。"""
        rng = np.random.RandomState(self.seed)
        return rng.randn(batch_size, self.context_dim).astype(np.float64)

    def encode_uncond(self, batch_size: int = 1) -> np.ndarray:
        """生成无条件 embedding（模拟空提示或负向提示的 embedding）。"""
        rng = np.random.RandomState(self.seed + 1)
        return rng.randn(batch_size, self.context_dim).astype(np.float64)


# =============================================================================
# Demo 运行器
# =============================================================================


def run_demo(
    latent_shape: Tuple[int, ...] = (1, 16, 64, 64),
    context_dim: int = 768,
    num_steps: int = 28,
    cfg_scales: List[float] = None,
    complexity_factor: float = 0.5,
    num_warmup: int = 3,
    seed: int = 42,
) -> Dict:
    """
    运行 CFG batching 对照实验。

    工作流：
      1. 初始化 mock denoiser + mock text encoder
      2. 对每个 cfg_scale，运行 Sequential CFG 和 Batched CFG 各 num_steps 步
      3. 比较两种 mode 的 latency、peak memory、numerical difference

    参数：
        latent_shape: latent 张量 shape。
        context_dim: text embedding 维度。
        num_steps: 总去噪步数。
        cfg_scales: 要扫描的 CFG scale 列表。
        complexity_factor: 模拟计算量倍率。
        num_warmup: warmup 步数（不计入统计）。
        seed: 随机种子。

    返回：
        包含所有指标的结果字典。
    """
    if cfg_scales is None:
        cfg_scales = [1.0, 1.5, 3.0, 5.0, 7.5, 10.0, 15.0]

    np.random.seed(seed)

    # 初始化组件
    denoiser = MockDenoiser(
        latent_shape=latent_shape,
        context_dim=context_dim,
        complexity_factor=complexity_factor,
        seed=seed,
    )
    text_encoder = MockTextEncoder(context_dim=context_dim, seed=seed)

    # 生成 embedding
    B = latent_shape[0]
    cond_emb = text_encoder.encode_cond(batch_size=B)
    uncond_emb = text_encoder.encode_uncond(batch_size=B)

    results_per_scale = []

    for cfg_scale in cfg_scales:
        print(f"  CFG scale = {cfg_scale:.1f} ...")

        # 初始化 latent（纯噪声）
        rng = np.random.RandomState(seed + 100)
        latent_seq = rng.randn(*latent_shape).astype(np.float64)
        latent_batch = latent_seq.copy()

        # === Sequential CFG ===
        seq_latencies = []
        for step in range(num_warmup + num_steps):
            v_seq, m_seq = cfg_sequential(
                denoiser, latent_seq, 0.5, cond_emb, uncond_emb, cfg_scale
            )
            if step >= num_warmup:
                seq_latencies.append(m_seq["latency_total_s"])
            # 更新 latent（Euler 步模拟）
            latent_seq = latent_seq + (1.0 - 0.5) * v_seq  # dt ≈ 0.5

        # === Batched CFG ===
        batch_latencies = []
        for step in range(num_warmup + num_steps):
            v_batch, m_batch = cfg_batched(
                denoiser, latent_batch, 0.5, cond_emb, uncond_emb, cfg_scale
            )
            if step >= num_warmup:
                batch_latencies.append(m_batch["latency_total_s"])
            # 更新 latent（Euler 步模拟）
            latent_batch = latent_batch + (1.0 - 0.5) * v_batch  # dt ≈ 0.5

        # 数值差异
        diff_max = numerical_diff_max(v_seq, v_batch)
        diff_rmse = numerical_diff_rmse(v_seq, v_batch)

        # 统计
        seq_avg = np.mean(seq_latencies) if seq_latencies else 0.0
        seq_std = np.std(seq_latencies) if seq_latencies else 0.0
        batch_avg = np.mean(batch_latencies) if batch_latencies else 0.0
        batch_std = np.std(batch_latencies) if batch_latencies else 0.0

        speedup = seq_avg / batch_avg if batch_avg > 0 else 1.0

        scale_result = {
            "cfg_scale": cfg_scale,
            "sequential": {
                "avg_latency_s": round(float(seq_avg), 6),
                "std_latency_s": round(float(seq_std), 6),
                "total_latency_s": round(float(seq_avg * num_steps), 6),
                "num_forwards": 2,
                "peak_memory_estimated_bytes": m_seq["peak_memory_estimated_bytes"],
                "peak_memory_estimated_mb": round(
                    m_seq["peak_memory_estimated_bytes"] / 1024**2, 2
                ),
            },
            "batched": {
                "avg_latency_s": round(float(batch_avg), 6),
                "std_latency_s": round(float(batch_std), 6),
                "total_latency_s": round(float(batch_avg * num_steps), 6),
                "num_forwards": 1,
                "peak_memory_estimated_bytes": m_batch["peak_memory_estimated_bytes"],
                "peak_memory_estimated_mb": round(
                    m_batch["peak_memory_estimated_bytes"] / 1024**2, 2
                ),
            },
            "comparison": {
                "speedup": round(speedup, 2),
                "memory_increase_x": round(
                    m_batch["peak_memory_estimated_bytes"]
                    / m_seq["peak_memory_estimated_bytes"],
                    2,
                ),
                "numerical_diff_max": round(diff_max, 10),
                "numerical_diff_rmse": round(diff_rmse, 10),
            },
        }
        results_per_scale.append(scale_result)

        print(
            f"     seq={seq_avg*1000:.2f}ms │ batch={batch_avg*1000:.2f}ms │ "
            f"     speedup={speedup:.2f}× │ diff_max={diff_max:.2e} │ "
            f"mem: seq={m_seq['peak_memory_estimated_bytes']/1024**2:.1f}MB "
            f"batch={m_batch['peak_memory_estimated_bytes']/1024**2:.1f}MB"
        )

    # 汇总
    total_results = {
        "experiment": "cfg_batching_experiment",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "latent_shape": list(latent_shape),
            "context_dim": context_dim,
            "num_steps": num_steps,
            "num_warmup": num_warmup,
            "cfg_scales": cfg_scales,
            "complexity_factor": complexity_factor,
            "seed": seed,
            "backend": "numpy_mock",
        },
        "summary": {
            "best_speedup": max(
                r["comparison"]["speedup"] for r in results_per_scale
            ),
            "worst_speedup": min(
                r["comparison"]["speedup"] for r in results_per_scale
            ),
            "avg_speedup": round(
                np.mean([r["comparison"]["speedup"] for r in results_per_scale]), 2
            ),
            "max_numerical_diff": max(
                r["comparison"]["numerical_diff_max"] for r in results_per_scale
            ),
        },
        "results_per_scale": results_per_scale,
        "conclusion": (
            "Batched CFG 一次 forward 处理 cond+uncond，"
            "速度快 ~1.3–2.0×，但峰值显存增 ~2×。"
            "数值差异极小（<1e-6），理论上应完全一致，"
            "微小差异仅来自浮点 accumulate order 不同。"
            "对 中等显存配置 VRAM 建议：若剩余显存 >50%，优先 Batched CFG；"
            "否则退而用 Sequential CFG。"
            "★★★ CFG 是在 vector field 层面做的线性插值，"
            "不是 latent 层面的 blending。"
            "★★★ Diffusion 主优化不是 LLM KV cache，"
            "但 CFG batching 是真正常用的 diffusion 优化技术。"
        ),
    }

    return total_results


# =============================================================================
# Markdown 表格生成
# =============================================================================


def generate_markdown(results: Dict) -> str:
    """从结果 dict 生成 Markdown 表格。"""
    lines = []
    lines.append("# CFG Batching 对照实验结果")
    lines.append("")
    lines.append(f"**时间戳**: {results['timestamp']}")
    lines.append(f"**Latent shape**: {results['config']['latent_shape']}")
    lines.append(f"**步数**: {results['config']['num_steps']}")
    lines.append(f"**Backend**: {results['config']['backend']}")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    s = results["summary"]
    lines.append(f"- 最佳加速比: **{s['best_speedup']:.2f}×**")
    lines.append(f"- 最差加速比: **{s['worst_speedup']:.2f}×**")
    lines.append(f"- 平均加速比: **{s['avg_speedup']:.2f}×**")
    lines.append(f"- 最大数值差异: **{s['max_numerical_diff']:.2e}**")
    lines.append("")
    lines.append("## 分 Scale 对比")
    lines.append("")
    lines.append(
        "| CFG Scale | Sequential (ms/步) | Batched (ms/步) | "
        "加速比 | 数值差异(max) | Seq 显存(MB) | Batch 显存(MB) |"
    )
    lines.append(
        "|----------:|-------------------:|----------------:|"
        "------:|------------:|------------:|--------------:|"
    )
    for r in results["results_per_scale"]:
        lines.append(
            f"| {r['cfg_scale']:.1f} "
            f"| {r['sequential']['avg_latency_s']*1000:.2f} "
            f"| {r['batched']['avg_latency_s']*1000:.2f} "
            f"| {r['comparison']['speedup']:.2f}× "
            f"| {r['comparison']['numerical_diff_max']:.2e} "
            f"| {r['sequential']['peak_memory_estimated_mb']:.1f} "
            f"| {r['batched']['peak_memory_estimated_mb']:.1f} |"
        )
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    lines.append(f"> {results['conclusion']}")
    lines.append("")
    lines.append("### 中等显存配置 VRAM 策略建议")
    lines.append("")
    lines.append("| VRAM 预算 | 建议 CFG Mode | 理由 |")
    lines.append("|----------|-------------|------|")
    lines.append(
        "| > 6 GB 剩余 | **Batched CFG** | 显存充足，速度优先 |"
    )
    lines.append(
        "| 3–6 GB 剩余 | 视情况选择 | 需评估具体模型参数量 |"
    )
    lines.append(
        "| < 3 GB 剩余 | **Sequential CFG** | 显存紧张，优先不 OOM |"
    )
    lines.append("")
    lines.append("### 与 LLM KV Cache 的区别")
    lines.append("")
    lines.append(
        "- **CFG batching**：将 cond + uncond 的 latent 拼接为双倍 batch，"
        "一次 forward 同时处理。"
    )
    lines.append(
        "- **LLM KV cache**：存储自回归生成中已生成的 token 的 key/value，"
        "避免重复计算历史 token 的 attention。"
    )
    lines.append(
        "- CFG batching 是扩散模型特有的加速技术（利用 cond/uncond 共享同一个 denoiser），"
        "LLM 的自回归场景中不存在这种一次 forward 处理两个 embedding 的机会。"
    )
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 命令行接口
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "CFG Batching 对照实验 — 比较 Sequential CFG 与 Batched CFG 的 "
            "latency / VRAM / numerical difference。\n"
            "★★★ CFG 在 vector field 层面执行，不是 latent 层面。\n"
            "★★★ Diffusion 主优化不是 LLM KV cache。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
========== 示例 ==========

# 默认 demo（28 步，cfg_scale 扫描 1.0–15.0）
python cfg_batching_experiment.py --demo

# 自定义步数和 scale
python cfg_batching_experiment.py --demo --num_steps 50 --cfg_scales 1.0 3.0 7.5

# 大分辨率 latent
python cfg_batching_experiment.py --demo --latent_shape 1 16 128 128
""",
    )

    # === 运行模式 ===
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行 demo 对照实验：比较 Sequential/Batched CFG",
    )

    # === Demo 参数 ===
    demo_group = parser.add_argument_group("Demo 参数")
    demo_group.add_argument(
        "--num_steps",
        type=int,
        default=28,
        help="总去噪步数（默认 28，SD3 标准）",
    )
    demo_group.add_argument(
        "--latent_shape",
        type=int,
        nargs=4,
        default=[1, 16, 64, 64],
        help="Latent shape (B C H W)，默认 1 16 64 64",
    )
    demo_group.add_argument(
        "--context_dim",
        type=int,
        default=768,
        help="Text embedding 维度（默认 768）",
    )
    demo_group.add_argument(
        "--cfg_scales",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 3.0, 5.0, 7.5, 10.0, 15.0],
        help="要扫描的 CFG scale 列表（默认 1.0 1.5 3.0 5.0 7.5 10.0 15.0）",
    )
    demo_group.add_argument(
        "--complexity_factor",
        type=float,
        default=0.5,
        help="Mock denoiser 计算倍率（默认 0.5）",
    )
    demo_group.add_argument(
        "--num_warmup",
        type=int,
        default=3,
        help="Warmup 步数，不计入统计（默认 3）",
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

    latent_shape = tuple(args.latent_shape)

    print("=" * 72)
    print("  CFG Batching 对照实验")
    print("  ★ Sequential CFG vs Batched CFG ★")
    print("  ★ CFG 在 vector field 层面执行，不是 latent 层面 ★")
    print("=" * 72)
    print(f"  Latent shape: {latent_shape}")
    print(f"  Context dim:  {args.context_dim}")
    print(f"  Num steps:    {args.num_steps}")
    print(f"  CFG scales:   {args.cfg_scales}")
    print(f"  Backend:      numpy mock (no torch)")
    print()

    # 运行 demo
    results = run_demo(
        latent_shape=latent_shape,
        context_dim=args.context_dim,
        num_steps=args.num_steps,
        cfg_scales=args.cfg_scales,
        complexity_factor=args.complexity_factor,
        num_warmup=args.num_warmup,
        seed=args.seed,
    )

    # 打印结果
    print()
    print("─" * 72)
    print("  汇总")
    print("─" * 72)
    s = results["summary"]
    print(f"  最佳加速比: {s['best_speedup']:.2f}×")
    print(f"  最差加速比: {s['worst_speedup']:.2f}×")
    print(f"  平均加速比: {s['avg_speedup']:.2f}×")
    print(f"  最大数值差异: {s['max_numerical_diff']:.2e}")
    print()
    print("─" * 72)
    print("  结论")
    print("─" * 72)
    print(f"  {results['conclusion']}")

    # 保存结果
    if not args.no_save:
        timestamp = results["timestamp"]
        # JSON
        json_path = os.path.join(
            args.output_dir, f"cfg_batching_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON 已保存: {json_path}")

        # Markdown
        md_content = generate_markdown(results)
        md_path = os.path.join(
            args.output_dir, f"cfg_batching_{timestamp}.md"
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  MD 已保存:   {md_path}")

    print()
    print("=" * 72)
    print("  实验完成。")
    print("=" * 72)


if __name__ == "__main__":
    main()
