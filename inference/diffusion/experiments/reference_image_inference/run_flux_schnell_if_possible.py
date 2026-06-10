#!/usr/bin/env python3
"""run_flux_schnell_if_possible.py — 使用 HuggingFace Diffusers 运行 FLUX.1-schnell 文生图推理。

**前置条件**（运行前必须完成）：
  - Python 3.13+（通过 uv 管理）
  - torch 已安装：`uv sync`
  - diffusers 已安装：`uv sync`
  - HF token 已配置：`huggingface-cli login`
  - 已接受 BFL 许可：访问 https://huggingface.co/black-forest-labs/FLUX.1-schnell
    并点击 "Agree and access repository"

**FLUX.1-schnell 是 gated 模型（Apache 2.0，但需授权），必须先到 HF 接受 license！**

**关键特性**：
  - 4 步蒸馏（schnell = 德语 "fast"）：推荐只用 4 步。
  - 不使用 CFG（cfg_scale ≈ 1.0）。
  - VRAM 约 10GB（偏紧），推荐 sequential CPU offload。
  - 社区 Q4 量化可大幅降低 VRAM，若 fp16 OOM 优先尝试 GGUF 量化路径。

**Fallback 路径**（若本脚本失败）：
  1. 确认 HF login + license accepted
  2. 降 resolution：--height 768 --width 768
  3. 尝试社区 Q4 量化路径（需额外安装 bitsandbytes）
  4. 仍失败 → 记录 blocker，尝试 Sana（run_sana_if_possible.py）

**不要**在 dev host (Mac M5) 上跑此脚本。
**不要**把 diffusers 的调用包装成 diffusion_engine/ 的成果。
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

torch = None
diffusers = None


def _ensure_deps() -> None:
    global torch, diffusers
    missing = []
    try:
        import torch as _torch
        torch = _torch
    except ImportError:
        missing.append("torch")
    try:
        import diffusers as _diffusers
        diffusers = _diffusers
    except ImportError:
        missing.append("diffusers")
    if missing:
        print(f"[BLOCKER] 缺少依赖: {', '.join(missing)}")
        print("  请运行: uv sync")
        sys.exit(1)


def check_cuda_available() -> bool:
    if torch.cuda.is_available():
        return True
    return False


# ---------------------------------------------------------------------------
# 1. 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FLUX.1-schnell 文生图推理（HuggingFace Diffusers）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
前置条件:
  FLUX.1-schnell 是 gated 模型，需要:
    1. 注册 HF 账号: https://huggingface.co/join
    2. 创建 access token: https://huggingface.co/settings/tokens
    3. huggingface-cli login
    4. 访问 https://huggingface.co/black-forest-labs/FLUX.1-schnell
       点击 "Agree and access repository"

模型下载 (~23GB，请预留磁盘空间):
  huggingface-cli download black-forest-labs/FLUX.1-schnell

关键特性:
  - schnell 仅需 4 步（distilled）
  - 不使用 CFG（cfg_scale ≈ 1.0）
  - VRAM 约 10GB（12GB 偏紧，推荐 sequential CPU offload）

Fallback（若 OOM 或失败）:
  1. 降 resolution: --height 768 --width 768
  2. 尝试社区 GGUF Q4 量化路径（需 bitsandbytes）
  3. 确认 HF token 已配置 + license 已接受

设备要求:
  需要 CUDA GPU（RTX 5070 Ti 12GB）。MPS/CPU 不支持。
""",
    )

    # 必填参数
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="正向 prompt（必填）。例如：'一只柴犬在樱花树下'",
    )

    # 模型参数
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="HF 模型 ID。默认 FLUX.1-schnell（4 步蒸馏，Apache 2.0）",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="负向 prompt。FLUX schnell 不使用 CFG，通常留空",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="去噪步数（默认 4）。schnell 推荐 4 步，不要设超过 8",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale（默认 1.0）。schnell 不使用 CFG（设为 1.0 = 不使用）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="输出高度（默认 1024）。降级可设 768",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="输出宽度（默认 1024）。降级可设 768",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（默认 0）。设为 -1 使用随机种子",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="推理精度（默认 fp16）",
    )

    # FLUX specific
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        default=True,
        help="启用 model CPU offload（默认启用）",
    )
    parser.add_argument(
        "--no-enable_cpu_offload",
        action="store_false",
        dest="enable_cpu_offload",
        help="禁用 CPU offload。12GB 设备不建议禁用",
    )
    parser.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        default=True,
        help="启用 VAE slicing（默认启用）",
    )
    parser.add_argument(
        "--no-enable_vae_slicing",
        action="store_false",
        dest="enable_vae_slicing",
        help="禁用 VAE slicing",
    )
    parser.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        default=True,
        help="启用 VAE tiling（默认启用）",
    )
    parser.add_argument(
        "--no-enable_vae_tiling",
        action="store_false",
        dest="enable_vae_tiling",
        help="禁用 VAE tiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出目录（默认 results/）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. 主推理逻辑
# ---------------------------------------------------------------------------
def run_flux_schnell_inference(args: argparse.Namespace) -> dict:
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"[INFO] 模型 ID: {args.model_id}")
    print(f"[INFO] Prompt: {args.prompt}")
    print(f"[INFO] Resolution: {args.height}x{args.width}")
    print(f"[INFO] Steps: {args.num_steps}, CFG: {args.cfg_scale}")
    print(f"[INFO] Dtype: {args.dtype}, Seed: {args.seed}")
    print(f"[INFO] CPU offload: {args.enable_cpu_offload}")
    print(f"[INFO] VAE slicing: {args.enable_vae_slicing}")
    print(f"[INFO] VAE tiling: {args.enable_vae_tiling}")

    if not check_cuda_available():
        print("[WARN] CUDA 不可用。FLUX pipeline 在 MPS/CPU 上无法运行。")
        print("[WARN] 建议在远程 RTX 5070 Ti 上执行本脚本。")
        return {
            "status": "blocker",
            "output_path": None,
            "peak_vram_mb": 0.0,
            "elapsed_s": 0.0,
            "error": "CUDA 不可用。需 RTX 5070 Ti 远程执行。",
        }

    seed = args.seed if args.seed >= 0 else int(time.time())
    generator = torch.Generator(device="cuda").manual_seed(seed)

    t_start = time.time()

    try:
        # 加载 pipeline
        print(f"[INFO] 加载 pipeline: {args.model_id} ...")
        print("[INFO] FLUX 模型 ~23GB 压缩包，加载可能需要几分钟，请耐心等待...")

        pipeline = diffusers.FluxPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
        )
        pipeline = pipeline.to("cuda")

        # 启用优化
        if args.enable_cpu_offload:
            print("[INFO] 启用 model CPU offload ...")
            pipeline.enable_model_cpu_offload()
        if args.enable_vae_slicing:
            print("[INFO] 启用 VAE slicing ...")
            pipeline.enable_vae_slicing()
        if args.enable_vae_tiling:
            print("[INFO] 启用 VAE tiling ...")
            pipeline.enable_vae_tiling()

        # 执行推理
        print(f"[INFO] 开始推理（{args.num_steps} 步）...")
        t_infer_start = time.time()

        image = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            num_inference_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

        t_infer_end = time.time()
        t_end = time.time()

        # 保存图片
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"flux_schnell_{timestamp}_{seed}.png")
        image.save(output_path)

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        elapsed_total = t_end - t_start
        elapsed_infer = t_infer_end - t_infer_start

        print(f"\n[SUCCESS] 推理完成!")
        print(f"  输出: {output_path}")
        print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")
        print(f"  推理耗时: {elapsed_infer:.1f}s（总耗时: {elapsed_total:.1f}s）")

        return {
            "status": "success",
            "output_path": output_path,
            "peak_vram_mb": peak_vram_mb,
            "elapsed_s": elapsed_total,
            "elapsed_infer_s": elapsed_infer,
            "error": None,
        }

    except Exception as exc:
        t_end = time.time()
        elapsed = t_end - t_start
        err_msg = f"{type(exc).__name__}: {exc}"

        print(f"\n[FAILED] 推理失败 ({elapsed:.1f}s)")
        print(f"  错误: {err_msg}")

        if "out of memory" in str(exc).lower():
            print("[HINT] OOM — FLUX schnell 在 12GB 上偏紧。尝试降级:")
            print("  1. --height 768 --width 768")
            print("  2. --num_steps 4（已是最小值）")
            print("  3. 尝试社区 GGUF Q4 量化路径")
            print("  4. 尝试 run_sana_if_possible.py（Sana 更省显存）")
        elif "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
            print("[HINT] 权限问题 — FLUX.1-schnell 是 gated 模型（Apache 2.0）:")
            print("  1. 确认 huggingface-cli login 已执行")
            print("  2. 访问 https://huggingface.co/black-forest-labs/FLUX.1-schnell")
            print("  3. 点击 'Agree and access repository'")
            print("  4. 等待 5 分钟后重试")
        elif "RepositoryNotFoundError" in type(exc).__name__:
            print("[HINT] 模型未找到 — 可能是未接受许可:")
            print("  请先到 HF 页面 accept license 再运行。")
        else:
            traceback.print_exc()

        return {
            "status": "blocker",
            "output_path": None,
            "peak_vram_mb": 0.0,
            "elapsed_s": elapsed,
            "error": err_msg,
        }


# ---------------------------------------------------------------------------
# 3. 主入口
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    _ensure_deps()
    result = run_flux_schnell_inference(args)

    summary_path = os.path.join(args.output_dir, "flux_schnell_result_summary.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"FLUX.1-schnell Inference Result\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Resolution: {args.height}x{args.width}\n")
        f.write(f"Steps: {args.num_steps}, CFG: {args.cfg_scale}\n")
        f.write(f"Dtype: {args.dtype}, Seed: {args.seed}\n")
        f.write(f"Status: {result['status']}\n")
        if result["output_path"]:
            f.write(f"Output: {result['output_path']}\n")
        f.write(f"Peak VRAM: {result['peak_vram_mb']:.1f} MB\n")
        f.write(f"Elapsed: {result['elapsed_s']:.1f}s\n")
        if result["error"]:
            f.write(f"Error: {result['error']}\n")
    print(f"[INFO] 结果摘要已写入: {summary_path}")

    if result["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
