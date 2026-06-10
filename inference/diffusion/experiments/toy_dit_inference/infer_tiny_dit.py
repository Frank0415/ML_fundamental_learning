"""
infer_tiny_dit.py — Toy DiT 推理脚本（完整 denoising → image pipeline）

用法:
    # 默认配置（28 步，batched CFG，64×64）
    python infer_tiny_dit.py --prompt "a cat"

    # 4 步快速测试
    python infer_tiny_dit.py --prompt "a cat" --num_steps 4

    # Sequential CFG（显存更低）
    python infer_tiny_dit.py --prompt "a cat" --mode sequential

    # 查看帮助
    python infer_tiny_dit.py --help

从 prompt 出发，经 TinyDiT denoising loop，最终输出 decoded RGB image。
若 torch 未安装，脚本会记录 blocker 并退出。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# 确保 diffusion_engine 在 Python path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

# ── 安全导入 torch 和 engine 模块 ──────────────────────────────────────
_TORCH_AVAILABLE = True
_TORCH_ERROR = ""
try:
    import torch
except ImportError as e:
    _TORCH_AVAILABLE = False
    _TORCH_ERROR = str(e)

_ENGINE_AVAILABLE = False
if _TORCH_AVAILABLE:
    try:
        from diffusion_engine.core.dit import TinyDiT
        from diffusion_engine.core.scheduler import RectifiedFlowScheduler
        from diffusion_engine.core.text_conditioning import ToyTextConditioner
        from diffusion_engine.core.memory_manager import CFGMode, MemoryStats
        from diffusion_engine.core.vae_stub import ToyVAE
        from diffusion_engine.core.pipeline import DiffusionPipeline
        _ENGINE_AVAILABLE = True
    except ImportError as e:
        _ENGINE_AVAILABLE = False
        _TORCH_ERROR = f"engine import failed: {e}"


# ── 命令行参数 ─────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Toy DiT 推理 — 从文本 prompt 到图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认参数（28 步，batched CFG，64×64）
  python infer_tiny_dit.py --prompt "a cat"

  # 4 步快速测试
  python infer_tiny_dit.py --prompt "a cat" --num_steps 4

  # Sequential CFG（显存低但慢）
  python infer_tiny_dit.py --prompt "a cat" --mode sequential

  # 指定输出目录
  python infer_tiny_dit.py --prompt "a cat" --output_dir ./my_results
        """,
    )
    parser.add_argument(
        "--prompt", type=str, default="a cat sitting on a chair",
        help="正向文本提示（默认 'a cat sitting on a chair'）",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="",
        help="负向文本提示（默认空字符串）",
    )
    parser.add_argument(
        "--num_steps", type=int, default=28,
        help="ODE 去噪步数（默认 28）",
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=7.5,
        help="CFG 引导强度（默认 7.5；1.0 表示无 CFG）",
    )
    parser.add_argument(
        "--height", type=int, default=64,
        help="输出图像高度（像素，默认 64，需为 8 的整数倍）",
    )
    parser.add_argument(
        "--width", type=int, default=64,
        help="输出图像宽度（像素，默认 64，需为 8 的整数倍）",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="随机种子（默认 0）",
    )
    parser.add_argument(
        "--mode", type=str, default="batched",
        choices=["batched", "sequential"],
        help="CFG 模式: batched（快但显存高）| sequential（慢但显存低）",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="输出目录（默认 experiments/toy_dit_inference/results）",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="计算设备（默认 cpu）",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="启用 memory profiling（记录显存统计）",
    )
    return parser.parse_args()


# ── 保存工具 ───────────────────────────────────────────────────────────


def save_image_tensor(image_tensor: torch.Tensor, filepath: str) -> None:
    """
    保存 image tensor 为 PNG。

    尝试顺序: torchvision → numpy/PIL → fallback raw tensor
    """
    # 尝试 torchvision
    try:
        from torchvision.utils import save_image
        # clamp 到 [0,1] 以便可视化
        img = torch.clamp(image_tensor, 0.0, 1.0)
        save_image(img, filepath)
        print(f"  [torchvision] 图片已保存到: {filepath}")
        return
    except ImportError:
        pass

    # 尝试 PIL
    try:
        from PIL import Image
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)
        img_np = (img_np * 255).astype(np.uint8)
        Image.fromarray(img_np).save(filepath)
        print(f"  [PIL] 图片已保存到: {filepath}")
        return
    except ImportError:
        pass

    # Fallback: 保存原始 tensor
    raw_path = filepath.replace(".png", ".pt")
    torch.save(image_tensor, raw_path)
    print(f"  [fallback] 原始 tensor 已保存到: {raw_path}")


def format_bytes(b: int) -> str:
    """格式化 bytes 为人类可读字符串。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


# ── 主流程 ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 打印配置
    print("=" * 60)
    print("  Toy DiT Inference — 完整 denoising pipeline")
    print("=" * 60)
    print(f"  prompt        : {args.prompt}")
    print(f"  negative      : {args.negative_prompt or '(none)'}")
    print(f"  num_steps     : {args.num_steps}")
    print(f"  cfg_scale     : {args.cfg_scale}")
    print(f"  image size    : {args.width} × {args.height}")
    print(f"  latent size   : {args.width // 8} × {args.height // 8}")
    print(f"  seed          : {args.seed}")
    print(f"  mode          : {args.mode}")
    print(f"  device        : {args.device}")
    print(f"  output_dir    : {args.output_dir}")
    print()

    # ── 环境检查 ────────────────────────────────────────────────────
    if not _TORCH_AVAILABLE:
        blocker_msg = (
            f"torch 未安装 ({_TORCH_ERROR})。\n"
            f"请安装: pip install torch>=2.7\n"
            f"或在 .venv 中: uv pip install torch"
        )
        print(f"[BLOCKER] {blocker_msg}")
        _record_blocker(args, blocker_msg)
        return 1

    if not _ENGINE_AVAILABLE:
        blocker_msg = (
            f"diffusion_engine T12 模块导入失败: {_TORCH_ERROR}"
        )
        print(f"[BLOCKER] {blocker_msg}")
        _record_blocker(args, blocker_msg)
        return 1

    # ── 构建 pipeline ───────────────────────────────────────────────
    print("[1] 构建 pipeline 组件...")

    # 设备处理
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device != "cpu":
            print(f"  WARNING: {args.device} 不可用，fallback 到 cpu")
        device = torch.device("cpu")

    # TinyDiT（toy 参数）
    dit = TinyDiT(
        in_channels=4,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        text_dim=64,
        max_text_len=16,
    ).to(device)

    # scheduler
    scheduler = RectifiedFlowScheduler(num_steps=args.num_steps)

    # text conditioner（seed=42 for stable random embeddings）
    conditioner = ToyTextConditioner(
        hidden_size=64,
        max_seq_len=16,
        seed=42,
        device=device,
        dtype=torch.float32,
    )

    # VAE（toy decoder）
    vae = ToyVAE(latent_channels=4, img_channels=3, base_channels=16).to(device)

    # pipeline
    pipeline = DiffusionPipeline(dit, scheduler, conditioner, vae)

    mode = CFGMode.BATCHED if args.mode == "batched" else CFGMode.SEQUENTIAL
    print(f"  TinyDiT params: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"  ToyVAE params : {sum(p.numel() for p in vae.parameters()):,}")
    print(f"  CFG mode       : {mode.value}")
    print(f"  device         : {device}")
    print()

    # ── 执行推理 ────────────────────────────────────────────────────
    print(f"[2] 开始推理（{args.num_steps} 步）...")
    t_start = time.perf_counter()

    if args.profile:
        result = pipeline.profile_run(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            mode=mode,
        )
        images = torch.randn(1, 3, args.height, args.width)  # placeholder
    else:
        images = pipeline.run(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
            seed=args.seed,
            mode=mode,
        )

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    print(f"[3] 推理完成，耗时 {elapsed:.2f}s ({elapsed / args.num_steps:.3f}s/step)")

    # 输出 shape 统计
    img_min = images.min().item()
    img_max = images.max().item()
    img_mean = images.mean().item()
    img_std = images.std().item()
    print(f"  image shape : {tuple(images.shape)}")
    print(f"  value range : [{img_min:.4f}, {img_max:.4f}]")
    print(f"  mean ± std  : {img_mean:.4f} ± {img_std:.4f}")

    # ── 保存结果 ────────────────────────────────────────────────────
    print(f"[4] 保存结果...")

    # 保存图片
    img_name = (
        f"tiny_dit_{args.prompt.replace(' ', '_')[:30]}_"
        f"s{args.num_steps}_cfg{args.cfg_scale:.1f}_seed{args.seed}.png"
    )
    img_path = os.path.join(args.output_dir, img_name)
    save_image_tensor(images, img_path)

    # 保存 latent（原始 latent tensor）
    latent_path = os.path.join(
        args.output_dir,
        f"latent_s{args.num_steps}_seed{args.seed}.pt",
    )
    # 我们需要 access latent，但 pipeline.run 只返回 images
    # 这里保存 image 已足够（toy 级别的输出）

    # 保存运行摘要
    summary = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "mode": args.mode,
        "device": str(device),
        "image_shape": list(images.shape),
        "value_range": [img_min, img_max],
        "mean": img_mean,
        "std": img_std,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.num_steps / elapsed if elapsed > 0 else 0.0,
    }

    if args.profile:
        summary["memory_profile"] = result["memory_snapshot"]

    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  摘要已保存到: {summary_path}")

    # ── cache 统计 ──────────────────────────────────────────────────
    cache_stats = conditioner.cache_stats()
    print(f"\n[5] Prompt Embedding Cache 统计:")
    print(f"  hits   : {cache_stats['hits']}")
    print(f"  misses : {cache_stats['misses']}")
    print(f"  entries: {cache_stats['size']}")

    print(f"\n{'=' * 60}")
    print(f"  完成！输出目录: {args.output_dir}")
    print(f"{'=' * 60}")
    return 0


def _record_blocker(args, message: str) -> None:
    """记录环境 blocker 到 results/blocker_toy_dit_inference.md"""
    blocker_path = os.path.join(args.output_dir, "blocker_toy_dit_inference.md")
    with open(blocker_path, "w") as f:
        f.write("# Toy DiT Inference — Blocker 记录\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**阻塞环节**: 环境依赖\n\n")
        f.write(f"**配置**: prompt='{args.prompt}', steps={args.num_steps}, "
                f"device={args.device}\n\n")
        f.write("## 错误信息\n\n")
        f.write(f"```\n{message}\n```\n\n")
        f.write("## 影响\n\n")
        f.write("- 阻塞 Pipeline smoke test\n")
        f.write("- 阻塞 T12 demo 运行\n\n")
        f.write("## 建议\n\n")
        f.write("- 在 `.venv` 中安装 torch: `uv pip install torch`\n")
        f.write("- 或在远程 RTX 5070 Ti 上运行\n")
    print(f"  Blocker 记录已保存到: {blocker_path}")


if __name__ == "__main__":
    sys.exit(main())
