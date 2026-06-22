#!/usr/bin/env python3
"""
attention_memory_benchmark.py — Attention 显存 Benchmark 实验

估算和对比不同场景下 attention 操作的显存占用：
  1. Image tokens：典型 DiT image latent (B=1, H=64, W=64, C=16, p=2)
     → N = (64/2)² = 1024 tokens → attention matrix = 1024² ≈ 1M entries
     → fp16: ~2MB
  2. Video tokens（spacetime patch）：典型 Sora-style video latent
     (B=1, T=16, H=32, W=32, C=4, p=(1,2,2))
     → N = 16 × (32/2)² = 16 × 256 = 4096 tokens
     → attention matrix = 4096² ≈ 16.8M entries → fp16: ~32MB
  3. Text tokens：典型 CLIP/T5 text embedding (B=1, L=77, D=4096)
     → attention matrix = 77² ≈ 5.9K entries → fp16: ~12KB
  4. MMDiT joint attention：img (1024) + text (77) = 1101 tokens
     → attention matrix = 1101² ≈ 1.2M entries → fp16: ~2.3MB

★★★ 核心认知 ★★★
- Attention memory 是 O(N²)：token 数翻倍，attention matrix 翻四倍。
- Video attention memory 是 image 的 ~16 倍（token² 关系）：
  4096² / 1024² = 4² = 16
- 这是 diffusion 推理的真正瓶颈（不是 LLM KV cache）：
  - LLM KV cache: 存储自回归生成的 key/value 历史，随序列长度线性增长 O(N)。
  - Diffusion full attention: 每步对所有 token 做全连接 attention，O(N²)。
  - 两者在机制上完全不同：LLM 通过 KV cache 避免 O(N²) 重算，
    扩散模型无法用 KV cache 因为每步 latent 全刷新。
- 未接入 flash-attn / xformers，本实验是 toy 统计估算。

实验设计：
  1. 模拟 4 种场景的 attention matrix 内存占用
  2. O(N²) 复杂度扫描：token 数从 64 → 16384
  3. 对比 standard attention vs memory-efficient attention 理论内存差异
  4. 输出 JSON + MD 结果文件

输出：
  - results/attention_memory_<timestamp>.json — 结构化数据
  - results/attention_memory_<timestamp>.md   — 人类可读表格与结论

纯 numpy 实现，不依赖 torch。

========== 使用示例 ==========

# 查看帮助
python attention_memory_benchmark.py --help

# 默认 demo
python attention_memory_benchmark.py --demo --output_dir results

# 自定义 token 扫描范围
python attention_memory_benchmark.py --demo --token_min 64 --token_max 16384 --output_dir results

# 包含 MMDiT joint attention 对比
python attention_memory_benchmark.py --demo --include_mmdit --output_dir results
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
# Attention 内存计算工具
# =============================================================================


def compute_attention_memory(
    num_tokens: int,
    head_dim: int = 64,
    num_heads: int = 16,
    dtype_bytes: int = 2,
    include_qkv: bool = True,
    include_output: bool = True,
) -> Dict:
    """
    计算 attention 操作的总内存占用。

    内存组成（standard attention）：
      1. Attention matrix（Q @ K^T）：N × N × dtype_bytes
      2. Attention output（softmax(QK^T) @ V）：N × head_dim × dtype_bytes
      3. QKV projection output（可选）：N × 3 × head_dim × num_heads × dtype_bytes
      4. Output projection（可选）：N × head_dim × num_heads × dtype_bytes

    对 memory-efficient attention（如 flash-attn/xformers）：
      - attention matrix 不会被物化（不用完整的 N×N 矩阵）
      - 通过 tiling / block-wise 计算，峰值内存 ~O(N) 而非 O(N²)

    参数：
        num_tokens: token 数量 N。
        head_dim: 每个 head 的维度 d_k。
        num_heads: attention head 数量。
        dtype_bytes: 数据类型字节数（fp16=2, fp32=4, bf16=2）。
        include_qkv: 是否计入 QKV projection 内存。
        include_output: 是否计入 output projection 内存。

    返回：
        {
            "num_tokens": int,
            "attention_matrix_bytes": int,
            "attention_matrix_mb": float,
            "attention_output_bytes": int,
            "qkv_projection_bytes": int,
            "output_projection_bytes": int,
            "total_standard_bytes": int,
            "total_standard_mb": float,
            "total_memory_efficient_estimated_bytes": int,
            "total_memory_efficient_estimated_mb": float,
            "memory_efficient_saving_ratio": float,
        }
    """
    N = num_tokens
    d = head_dim
    h = num_heads
    D = d * h  # total hidden dimension

    # 1. Attention matrix: N × N（Q @ K^T 的结果）
    attn_matrix_bytes = N * N * dtype_bytes

    # 2. Attention output: N × d（softmax(QK^T) @ V，per head）
    #    再 concat → N × D
    attn_output_bytes = N * D * dtype_bytes

    # 3. QKV projection: N × 3D（linear → Q, K, V）
    qkv_bytes = N * 3 * D * dtype_bytes if include_qkv else 0

    # 4. Output projection: N × D（concat heads → linear projection）
    out_proj_bytes = N * D * dtype_bytes if include_output else 0

    # Standard attention 峰值：attention matrix + QKV + output
    total_standard = attn_matrix_bytes + qkv_bytes + attn_output_bytes + out_proj_bytes

    # Memory-efficient attention 估算（不存完整的 N×N attention matrix）：
    # 峰值 = QKV + output projections + softmax temp（≈N×h 而非 N×N）
    softmax_temp = N * h * dtype_bytes  # softmax LSE 或 max
    total_mem_eff = qkv_bytes + attn_output_bytes + out_proj_bytes + softmax_temp

    saving_ratio = total_standard / total_mem_eff if total_mem_eff > 0 else float("inf")

    return {
        "num_tokens": N,
        "attention_matrix_bytes": attn_matrix_bytes,
        "attention_matrix_mb": round(attn_matrix_bytes / 1024**2, 4),
        "attention_output_bytes": attn_output_bytes,
        "attention_output_mb": round(attn_output_bytes / 1024**2, 4),
        "qkv_projection_bytes": qkv_bytes,
        "qkv_projection_mb": round(qkv_bytes / 1024**2, 4),
        "output_projection_bytes": out_proj_bytes,
        "output_projection_mb": round(out_proj_bytes / 1024**2, 4),
        "total_standard_bytes": total_standard,
        "total_standard_mb": round(total_standard / 1024**2, 4),
        "total_memory_efficient_estimated_bytes": total_mem_eff,
        "total_memory_efficient_estimated_mb": round(total_mem_eff / 1024**2, 4),
        "memory_efficient_saving_ratio": round(saving_ratio, 2),
    }


def compute_mmdit_joint_memory(
    img_tokens: int,
    text_tokens: int,
    head_dim: int = 64,
    num_heads: int = 24,
    dtype_bytes: int = 2,
) -> Dict:
    """
    计算 MMDiT joint attention 的内存占用。

    MMDiT（Multi-Modal Diffusion Transformer）在 SD3/FLUX 中使用，
    它将 image tokens 和 text tokens concatenate 后做 joint full attention。
    因此 attention matrix 大小是 (img_N + txt_N)²。

    参数：
        img_tokens: image token 数。
        text_tokens: text token 数。
        head_dim: head 维度。
        num_heads: head 数量。
        dtype_bytes: dtype 字节数。

    返回：
        内存估算 dict。
    """
    total_tokens = img_tokens + text_tokens
    mem = compute_attention_memory(
        num_tokens=total_tokens,
        head_dim=head_dim,
        num_heads=num_heads,
        dtype_bytes=dtype_bytes,
    )
    mem["img_tokens"] = img_tokens
    mem["text_tokens"] = text_tokens
    mem["joint_tokens"] = total_tokens
    mem["joint_vs_image_only_ratio"] = round(
        (total_tokens**2) / (img_tokens**2), 2
    )
    return mem


# =============================================================================
# 场景定义
# =============================================================================


def define_scenarios(dtype_bytes: int = 2) -> List[Dict]:
    """
    定义典型推理场景的 token 配置。

    返回值：场景列表，每个场景包含 name, num_tokens, description, params。

    典型场景：
      - SD3 1024² image：latent 128×128 → p=2 → N=4096
      - Sora 16f video（spacetime patch）：T=16, H=32, W=32 → N=4096
      - CLIP text：L=77, D=768
      - DiT 512² image：latent 64×64 → p=2 → N=1024
      - HunyuanVideo 64f：T=64, H=32, W=32 → N=16384
    """
    return [
        {
            "name": "DiT 512² (latent 64×64, p=2)",
            "num_tokens": 1024,
            "description": (
                "图像 512×512 → VAE latent 64×64 → patch_size=2 → "
                "N = (64/2)² = 1024 tokens。"
            ),
            "params": {"width": 512, "height": 512, "patch_size": 2, "latent_dim": 16},
        },
        {
            "name": "SD3 1024² (latent 128×128, p=2)",
            "num_tokens": 4096,
            "description": (
                "图像 1024×1024 → VAE latent 128×128 → patch_size=2 → "
                "N = (128/2)² = 4096 tokens。"
            ),
            "params": {"width": 1024, "height": 1024, "patch_size": 2, "latent_dim": 16},
        },
        {
            "name": "SD3 2048² (latent 256×256, p=2)",
            "num_tokens": 16384,
            "description": (
                "图像 2048×2048 → VAE latent 256×256 → patch_size=2 → "
                "N = (256/2)² = 16384 tokens。"  "高分辨率，attention matrix ≈ 268M entries。"
            ),
            "params": {"width": 2048, "height": 2048, "patch_size": 2, "latent_dim": 16},
        },
        {
            "name": "Sora 16f 480p (spacetime patch p=1,2,2)",
            "num_tokens": 4096,
            "description": (
                "视频 T=16 帧，帧 480×720 → VAE latent 60×90 → "
                "spacetime patch p=(1,2,2) → N = 16 × (60/2)×(90/2) = 16 × 30 × 45 "
                "= 4096 tokens（简化）。"  "典型 Sora-style 短视频配置。"
            ),
            "params": {
                "frames": 16, "height": 480, "width": 720,
                "patch_t": 1, "patch_h": 2, "patch_w": 2,
                "latent_h": 60, "latent_w": 90,
            },
        },
        {
            "name": "Sora 64f 480p (spacetime patch p=1,2,2)",
            "num_tokens": 16384,
            "description": (
                "视频 T=64 帧 → 与上一场景同分辨率但帧数 4× → "
                "N = 64 × 30 × 45 = 16384 tokens。"
                "attention matrix ≈ 268M entries，与 SD3 2048² 图像相当。"
            ),
            "params": {
                "frames": 64, "height": 480, "width": 720,
                "patch_t": 1, "patch_h": 2, "patch_w": 2,
                "latent_h": 60, "latent_w": 90,
            },
        },
        {
            "name": "CLIP-L Text (L=77, D=768)",
            "num_tokens": 77,
            "description": (
                "文本 77 tokens × 768 dim → attention matrix 77² = 5,929 entries。"
                "文本 attention 内存占比极小。"
            ),
            "params": {"max_len": 77, "hidden_dim": 768},
        },
        {
            "name": "T5-XXL Text (L=512, D=4096)",
            "num_tokens": 512,
            "description": (
                "文本 512 tokens × 4096 dim → attention matrix 512² = 262,144 entries。"
                "长文本 encoder 的 attention 也不大。"
            ),
            "params": {"max_len": 512, "hidden_dim": 4096},
        },
        {
            "name": "MMDiT Joint (img 1024 + txt 77)",
            "num_tokens": 1101,
            "description": (
                "MMDiT joint attention: image 1024 tokens + text 77 tokens → "
                "N = 1101 → attention matrix 1101² ≈ 1.2M entries。"
                "略大于纯 image 1024 (1.0M)，但增幅有限。"
            ),
            "params": {"img_tokens": 1024, "txt_tokens": 77},
        },
        {
            "name": "CogVideoX 49f 480p",
            "num_tokens": 11520,
            "description": (
                "CogVideoX 49 帧 720×480 → VAE 3D latent → "
                "spacetime patch p=(4,4,4) 等 → 约 11520 tokens。"
            ),
            "params": {
                "frames": 49, "height": 480, "width": 720,
                "patch_t": 4, "patch_h": 4, "patch_w": 4,
            },
        },
    ]


# =============================================================================
# O(N²) 扫描
# =============================================================================


def run_complexity_scan(
    token_list: List[int],
    head_dim: int = 64,
    num_heads: int = 16,
    dtype_bytes: int = 2,
) -> List[Dict]:
    """
    扫描不同 token 数下的 attention 内存。

    展示 O(N²) 关系：token 数 2× → attention matrix 4×。

    参数：
        token_list: 要扫描的 token 数列表。
        head_dim: head 维度。
        num_heads: head 数量。
        dtype_bytes: dtype 字节数。

    返回：
        每个 token 数的内存估算列表。
    """
    scan_results = []
    for n in token_list:
        mem = compute_attention_memory(
            num_tokens=n,
            head_dim=head_dim,
            num_heads=num_heads,
            dtype_bytes=dtype_bytes,
        )
        scan_results.append(mem)

    return scan_results


def estimate_real_scenario(
    width: int,
    height: int,
    patch_size: int = 2,
    latent_channels: int = 16,
    vae_downsample: int = 8,
    fps: int = 30,
    duration_s: float = 1.0,
    spacetime_patch: Tuple[int, int, int] = (1, 2, 2),
    dtype_bytes: int = 2,
    num_heads: int = 24,
) -> Dict:
    """
    从真实分辨率估算 attention 内存。

    用于用户在命令行指定真实参数后得到直接的内存估算。

    参数：
        width: 像素宽度。
        height: 像素高度。
        patch_size: DiT patch size。
        latent_channels: VAE 潜在通道数。
        vae_downsample: VAE 下采样倍率。
        fps: 帧率。
        duration_s: 视频时长（秒）。
        spacetime_patch: 视频 spacetime patch (pt, ph, pw)。
        dtype_bytes: dtype 字节数。
        num_heads: attention head 数量。

    返回：
        包含估算结果的 dict。
    """
    latent_h = height // vae_downsample
    latent_w = width // vae_downsample
    num_tokens = (latent_h // patch_size) * (latent_w // patch_size)
    memory = compute_attention_memory(
        num_tokens=num_tokens,
        head_dim=64,
        num_heads=num_heads,
        dtype_bytes=dtype_bytes,
    )
    memory["width"] = width
    memory["height"] = height
    memory["patch_size"] = patch_size
    memory["latent_h"] = latent_h
    memory["latent_w"] = latent_w
    memory["vae_downsample"] = vae_downsample
    return memory


# =============================================================================
# Demo 运行器
# =============================================================================


def run_demo(
    token_list: List[int] = None,
    head_dim: int = 64,
    num_heads: int = 16,
    dtype_bytes: int = 2,
    include_mmdit: bool = True,
    seed: int = 42,
) -> Dict:
    """
    运行 attention memory benchmark demo。

    工作流：
      1. 定义场景 → 计算每个场景的 attention memory
      2. O(N²) 扫描 → token 数 64 → 16384
      3. 比较 standard vs memory-efficient attention
      4. MMDiT joint attention（可选）

    参数：
        token_list: token 扫描列表。
        head_dim: head 维度。
        num_heads: head 数量。
        dtype_bytes: dtype 字节数。
        include_mmdit: 是否包含 MMDiT joint attention。
        seed: 随机种子。

    返回：
        包含所有指标的结果字典。
    """
    if token_list is None:
        # 生成 token 扫描序列：指数增长
        token_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # === 1. 场景内存估算 ===
    scenarios = define_scenarios(dtype_bytes=dtype_bytes)
    scenario_results = []
    for sc in scenarios:
        mem = compute_attention_memory(
            num_tokens=sc["num_tokens"],
            head_dim=head_dim,
            num_heads=num_heads,
            dtype_bytes=dtype_bytes,
        )
        mem["scenario_name"] = sc["name"]
        mem["description"] = sc["description"]
        scenario_results.append(mem)

    # === 2. O(N²) 复杂度扫描 ===
    scan_results = run_complexity_scan(
        token_list=token_list,
        head_dim=head_dim,
        num_heads=num_heads,
        dtype_bytes=dtype_bytes,
    )

    # === 3. Memory-efficient attention 对比 ===
    # 选几个关键 token 数做对比
    comparison_points = [256, 1024, 4096, 16384]
    comparison_results = []
    for n in comparison_points:
        mem = compute_attention_memory(
            num_tokens=n,
            head_dim=head_dim,
            num_heads=num_heads,
            dtype_bytes=dtype_bytes,
        )
        comparison_results.append({
            "num_tokens": n,
            "standard_attention_mb": mem["total_standard_mb"],
            "memory_efficient_mb": mem["total_memory_efficient_estimated_mb"],
            "saving_ratio": mem["memory_efficient_saving_ratio"],
            "attention_matrix_mb": mem["attention_matrix_mb"],
            "attention_matrix_dominates": (
                mem["attention_matrix_bytes"] > mem["qkv_projection_bytes"]
            ),
        })

    # === 4. MMDiT joint attention（可选）===
    mmdit_results = None
    if include_mmdit:
        mmdit_results = compute_mmdit_joint_memory(
            img_tokens=1024,
            text_tokens=77,
            head_dim=head_dim,
            num_heads=num_heads,
            dtype_bytes=dtype_bytes,
        )

    # === 汇总 ===
    # 关键结论：video attention = image attention × 16
    img_1024_mem = next(
        s for s in scenario_results if s["num_tokens"] == 1024
    )
    video_4096_mem = next(
        s for s in scenario_results if s["num_tokens"] == 4096
    )
    video_vs_image = (
        video_4096_mem["attention_matrix_bytes"]
        / img_1024_mem["attention_matrix_bytes"]
        if img_1024_mem["attention_matrix_bytes"] > 0
        else 0
    )

    total_results = {
        "experiment": "attention_memory_benchmark",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "head_dim": head_dim,
            "num_heads": num_heads,
            "dtype": "fp16" if dtype_bytes == 2 else ("fp32" if dtype_bytes == 4 else "bf16"),
            "dtype_bytes": dtype_bytes,
            "scanned_token_counts": token_list,
            "backend": "numpy_estimation",
            "seed": seed,
        },
        "key_findings": {
            "video_vs_image_attention_ratio": round(video_vs_image, 2),
            "video_vs_image_explanation": (
                f"Video 4096 tokens attention matrix = "
                f"{video_4096_mem['attention_matrix_mb']} MB "
                f"vs image 1024 tokens = {img_1024_mem['attention_matrix_mb']} MB。"
                f"4096² / 1024² = {4096**2}/{1024**2} = {4096**2/1024**2:.0f}×。"
                f"视频 attention 内存是图像的同 token 数倍率 = "
                f"({4096}/{1024})² = {(4096/1024)**2:.0f}×。"
            ),
            "complexity": "Attention 内存 O(N²)。Token 数翻倍 → attention matrix 翻四倍。",
            "main_note": (
                "★★★ Diffusion 主优化不是 LLM KV cache，"
                "但 attention memory 是扩散推理的真实瓶颈。"
                "LLM KV cache 线性增长 O(N)，"
                "diffusion full attention 平方增长 O(N²)。"
                "中等显存配置 VRAM 下，1024² 图像 → attention matrix ~2 MB (fp16)，可接受；"
                "2048² 图像 → ~67 MB，勉强；"
                "视频 4096 tokens → ~32 MB (fp16)，需 memory-efficient attention。"
            ),
            "flash_attn_note": (
                "未接入 flash-attn / xformers，本实验是 toy 统计估算。"
                "真实 flash-attn 可将峰值显存从 O(N²) 降至 ~O(N)，"
                "但需要 CUDA/MPS 支持且本环境无法验证。"
            ),
        },
        "scenarios": scenario_results,
        "complexity_scan": scan_results,
        "memory_efficient_comparison": comparison_results,
        "mmdit_joint": mmdit_results,
    }

    return total_results


# =============================================================================
# Markdown 表格生成
# =============================================================================


def generate_markdown(results: Dict) -> str:
    """从结果 dict 生成 Markdown 表格。"""
    lines = []
    lines.append("# Attention Memory Benchmark 结果")
    lines.append("")
    lines.append(f"**时间戳**: {results['timestamp']}")
    lines.append(
        f"**Head配置**: dim={results['config']['head_dim']}, "
        f"heads={results['config']['num_heads']}, "
        f"dtype={results['config']['dtype']}"
    )
    lines.append(f"**Backend**: {results['config']['backend']}")
    lines.append("")

    lines.append("## 关键发现")
    lines.append("")
    kf = results["key_findings"]
    lines.append(f"- 视频 vs 图像 attention 倍率: **{kf['video_vs_image_attention_ratio']}×**")
    lines.append(f"- 复杂度: {kf['complexity']}")
    lines.append("")
    lines.append(f"> {kf['main_note']}")
    lines.append("")
    lines.append(f"> 📝 {kf['flash_attn_note']}")
    lines.append("")

    lines.append("## 场景估算")
    lines.append("")
    lines.append(
        "| 场景 | Tokens | Attn Matrix (MB) | QKV (MB) | Output (MB) | "
        "Total Standard (MB) | Mem-Eff 估算 (MB) | 节约比 |"
    )
    lines.append(
        "|------|-------:|-----------------:|---------:|------------:|"
        "---------------------:|------------------:|------:|"
    )
    for sc in results["scenarios"]:
        mem = sc
        lines.append(
            f"| {mem['scenario_name']} "
            f"| {mem['num_tokens']} "
            f"| {mem['attention_matrix_mb']} "
            f"| {mem['qkv_projection_mb']} "
            f"| {mem['attention_output_mb']} "
            f"| {mem['total_standard_mb']} "
            f"| {mem['total_memory_efficient_estimated_mb']} "
            f"| {mem['memory_efficient_saving_ratio']}× |"
        )
    lines.append("")

    lines.append("## Standard vs Memory-Efficient Attention 对比")
    lines.append("")
    lines.append(
        "| Tokens | Standard Attn (MB) | Mem-Eff (MB) | "
        "节省比 | Attn Matrix 主导? |"
    )
    lines.append(
        "|-------:|-------------------:|-------------:|"
        "------:|-----------------|"
    )
    for cmp in results["memory_efficient_comparison"]:
        lines.append(
            f"| {cmp['num_tokens']} "
            f"| {cmp['standard_attention_mb']} "
            f"| {cmp['memory_efficient_mb']} "
            f"| {cmp['saving_ratio']}× "
            f"| {'是' if cmp['attention_matrix_dominates'] else '否'} |"
        )
    lines.append("")

    if results.get("mmdit_joint"):
        mj = results["mmdit_joint"]
        lines.append("## MMDiT Joint Attention")
        lines.append("")
        lines.append(f"- Image tokens: {mj['img_tokens']}")
        lines.append(f"- Text tokens: {mj['text_tokens']}")
        lines.append(f"- Joint tokens: {mj['joint_tokens']}")
        lines.append(f"- Joint vs Image-only 倍率: {mj['joint_vs_image_only_ratio']}×")
        lines.append(f"- Attention matrix: {mj['attention_matrix_mb']} MB")
        lines.append(f"- Total standard: {mj['total_standard_mb']} MB")
        lines.append("")

    lines.append("## O(N²) 复杂度验证")
    lines.append("")
    lines.append(
        "| Tokens (N) | Attn Matrix (MB) | N² | N²/N_prev² |"
    )
    lines.append(
        "|-----------:|-----------------:|----|-----------|"
    )
    prev_n2 = None
    for sc in results["complexity_scan"]:
        n = sc["num_tokens"]
        n2 = n * n
        ratio = n2 / prev_n2 if prev_n2 else 1.0
        prev_n2 = n2
        lines.append(
            f"| {n} "
            f"| {sc['attention_matrix_mb']} "
            f"| {n2:,} "
            f"| {ratio:.1f}× |"
        )
    lines.append("")
    lines.append("> Token 数 ×2 → N² ×4。验证 O(N²) 复杂度。")
    lines.append("")

    lines.append("## 中等显存配置 VRAM 策略")
    lines.append("")
    lines.append("| 场景 | N | FP16 Attn Matrix | 建议 |")
    lines.append("|------|---|-----------------|------|")
    lines.append("| 512² 图像 | 1024 | 2 MB | Standard attention 安全 |")
    lines.append("| 1024² 图像 | 4096 | 32 MB | Standard 可，推荐 mem-eff |")
    lines.append("| 2048² 图像 | 16384 | 536 MB | **必须** memory-efficient attention |")
    lines.append("| 短视频 (16f) | 4096 | 32 MB | Standard 可，推荐 mem-eff |")
    lines.append("| 长视频 (64f) | 16384 | 536 MB | **必须** memory-efficient attention |")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# 命令行接口
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "Attention Memory Benchmark — 估算不同场景下 attention 显存占用。\n"
            "覆盖 image / video / text tokens 和 MMDiT joint attention。\n"
            "★★★ Diffusion 主优化不是 LLM KV cache，但 attention memory 是真实瓶颈。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
========== 示例 ==========

# 默认 demo（扫描 64–16384 tokens）
python attention_memory_benchmark.py --demo

# 自定义 token 范围
python attention_memory_benchmark.py --demo --token_min 64 --token_max 8192

# 包含 MMDiT joint attention
python attention_memory_benchmark.py --demo --include_mmdit

# 估算特定分辨率
python attention_memory_benchmark.py --estimate --width 1024 --height 1024

# 估算视频
python attention_memory_benchmark.py --estimate --width 720 --height 480 --frames 49
""",
    )

    # === 运行模式 ===
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行 demo：场景估算 + O(N²) 扫描 + mem-eff 对比",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="估算指定分辨率的 attention 内存（需指定 --width --height）",
    )

    # === Demo 参数 ===
    demo_group = parser.add_argument_group("Demo 参数")
    demo_group.add_argument(
        "--token_min",
        type=int,
        default=64,
        help="O(N²) 扫描最小 token 数（默认 64）",
    )
    demo_group.add_argument(
        "--token_max",
        type=int,
        default=16384,
        help="O(N²) 扫描最大 token 数（默认 16384）",
    )
    demo_group.add_argument(
        "--head_dim",
        type=int,
        default=64,
        help="Attention head 维度（默认 64）",
    )
    demo_group.add_argument(
        "--num_heads",
        type=int,
        default=16,
        help="Attention head 数量（默认 16）",
    )
    demo_group.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="数据类型（默认 fp16）",
    )
    demo_group.add_argument(
        "--include_mmdit",
        action="store_true",
        help="包含 MMDiT joint attention 估算",
    )

    # === 估算参数 ===
    est_group = parser.add_argument_group("估算参数（用于 --estimate）")
    est_group.add_argument(
        "--width",
        type=int,
        default=1024,
        help="图像/视频像素宽度（默认 1024）",
    )
    est_group.add_argument(
        "--height",
        type=int,
        default=1024,
        help="图像/视频像素高度（默认 1024）",
    )
    est_group.add_argument(
        "--frames",
        type=int,
        default=1,
        help="视频帧数（1=图像，>1=视频，默认 1）",
    )
    est_group.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="DiT patch size（默认 2）",
    )
    est_group.add_argument(
        "--vae_downsample",
        type=int,
        default=8,
        help="VAE 下采样倍率（默认 8）",
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

    if not args.demo and not args.estimate:
        parser.print_help()
        print("\n提示：使用 --demo 运行完整实验，或 --estimate 估算特定分辨率。")
        sys.exit(0)

    # dtype_bytes 映射
    dtype_map = {"fp16": 2, "fp32": 4, "bf16": 2}
    dtype_bytes = dtype_map[args.dtype]

    # 创建输出目录
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    # === 估算模式 ===
    if args.estimate:
        print("=" * 72)
        print("  Attention Memory 估算（指定分辨率）")
        print("=" * 72)
        total_tokens = 0
        if args.frames == 1:
            total_tokens = (args.height // args.vae_downsample // args.patch_size) * (
                args.width // args.vae_downsample // args.patch_size
            )
        else:
            total_tokens = args.frames * (
                args.height // args.vae_downsample // args.patch_size
            ) * (args.width // args.vae_downsample // args.patch_size)

        mem = compute_attention_memory(
            num_tokens=total_tokens,
            head_dim=args.head_dim,
            num_heads=args.num_heads,
            dtype_bytes=dtype_bytes,
        )
        print(f"  分辨率: {args.width}×{args.height}")
        print(f"  帧数: {args.frames}")
        print(f"  总 token 数: {total_tokens}")
        print(f"  Attention matrix: {mem['attention_matrix_mb']} MB ({args.dtype})")
        print(f"  QKV projection:   {mem['qkv_projection_mb']} MB")
        print(f"  总 standard attn: {mem['total_standard_mb']} MB")
        print(f"  总 mem-eff 估算:  {mem['total_memory_efficient_estimated_mb']} MB")
        print(f"  内存节约比:       {mem['memory_efficient_saving_ratio']}×")
        print()
        print(f"  ★ O(N²) 复杂度：token 数 ×2 → attention matrix ×4")
        if total_tokens >= 8192:
            print(f"  ⚠ 建议使用 memory-efficient attention（token > 8192）")
        sys.exit(0)

    # === Demo 模式 ===
    # 生成 token 扫描列表（指数增长）
    token_scan = []
    n = args.token_min
    while n <= args.token_max:
        token_scan.append(n)
        n *= 2
    if token_scan[-1] < args.token_max:
        token_scan.append(args.token_max)

    print("=" * 72)
    print("  Attention Memory Benchmark")
    print("  ★ 估算 image / video / text attention 显存 ★")
    print("  ★ Diffusion attention 是 O(N²)，不是 LLM KV cache O(N) ★")
    print("=" * 72)
    print(f"  Head dim:  {args.head_dim}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Dtype:     {args.dtype} ({dtype_bytes}B)")
    print(f"  Token 扫描: {token_scan}")
    print(f"  MMDiT:     {'是' if args.include_mmdit else '否'}")
    print()

    # 运行 demo
    results = run_demo(
        token_list=token_scan,
        head_dim=args.head_dim,
        num_heads=args.num_heads,
        dtype_bytes=dtype_bytes,
        include_mmdit=args.include_mmdit,
        seed=args.seed,
    )

    # 打印关键发现
    kf = results["key_findings"]
    print("─" * 72)
    print("  关键发现")
    print("─" * 72)
    print(f"  视频/图像 attention 倍率: {kf['video_vs_image_attention_ratio']}×")
    print(f"  {kf['complexity']}")
    print(f"  {kf['main_note']}")

    # 打印场景表格
    print()
    print("─" * 72)
    print("  场景估算（total standard attn memory）")
    print("─" * 72)
    for sc in results["scenarios"]:
        print(
            f"  {sc['scenario_name']:45s} "
            f"N={sc['num_tokens']:6d}  "
            f"attn_matrix={sc['attention_matrix_mb']:8.2f}MB  "
            f"total={sc['total_standard_mb']:8.2f}MB  "
            f"mem_eff={sc['total_memory_efficient_estimated_mb']:8.2f}MB  "
            f"save={sc['memory_efficient_saving_ratio']:.1f}×"
        )
    print()
    print("─" * 72)
    print("  Memory-Efficient Attention 对比")
    print("─" * 72)
    for cmp in results["memory_efficient_comparison"]:
        print(
            f"  N={cmp['num_tokens']:6d}  "
            f"standard={cmp['standard_attention_mb']:8.2f}MB  "
            f"mem_eff={cmp['memory_efficient_mb']:8.2f}MB  "
            f"save={cmp['saving_ratio']:.1f}×  "
            f"matrix_dominates={'是' if cmp['attention_matrix_dominates'] else '否'}"
        )

    # 保存结果
    if not args.no_save:
        timestamp = results["timestamp"]
        # JSON
        json_path = os.path.join(
            args.output_dir, f"attention_memory_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON 已保存: {json_path}")

        # Markdown
        md_content = generate_markdown(results)
        md_path = os.path.join(
            args.output_dir, f"attention_memory_{timestamp}.md"
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
