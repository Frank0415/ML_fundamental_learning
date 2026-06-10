#!/usr/bin/env python3
"""run_sd3_medium_if_possible.py — 使用 HuggingFace Diffusers 运行 SD3 Medium 文生图推理。

**前置条件**（运行前必须完成）：
  - Python 3.13+（通过 uv 管理）
  - torch 已安装：`uv sync`
  - diffusers 已安装：`uv sync`
  - HF token 已配置：`huggingface-cli login`
  - 已接受 Stability AI 许可：访问 https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
    并点击 "Agree and access repository"

**SD3 Medium 是 gated 模型，必须先到 HF 接受 license！**

**核心策略：关闭 T5（no-T5 路径）**
  SD3 Medium 有三个文本编码器：CLIP-L、CLIP-G、T5-XXL。
  T5-XXL 单独占用约 11GB VRAM（> 12GB 总预算）。
  本脚本默认 `--no_t5`（关闭 T5），将 VRAM 从 ~15GB 降到 ~4.3GB。

**Fallback 路径**（若本脚本失败）：
  1. 确认 HF login + license accepted
  2. 降 resolution：--height 768 --width 768
  3. 减 steps：--num_steps 15
  4. 启用 sequential CPU offload（替代 model CPU offload）
  5. 仍失败 → 记录 blocker，尝试 Sana（run_sana_if_possible.py）

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
        description="SD3 Medium 文生图推理（HuggingFace Diffusers）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
前置条件:
  SD3 Medium 是 gated 模型，需要:
    1. 注册 HF 账号: https://huggingface.co/join
    2. 创建 access token: https://huggingface.co/settings/tokens
    3. huggingface-cli login
    4. 访问 https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
       点击 "Agree and access repository"

模型下载:
  huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers

核心策略 (no-T5):
  默认关闭 T5-XXL 文本编码器（--no_t5）。
  T5-XXL 单独占约 11GB VRAM，在 12GB 设备上必须关闭。

Fallback（若 OOM 或失败）:
  1. 降 resolution: --height 768 --width 768
  2. 减 steps: --num_steps 15
  3. 启用更多 offload: 在脚本内设置 enable_sequential_cpu_offload
  4. 确认 HF token 已配置 + license 已接受

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
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="HF 模型 ID。默认 SD3 Medium（no-T5 约 4.3GB VRAM）",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="负向 prompt（默认空）",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=28,
        help="去噪步数（默认 28）。SD3 推荐 28~50。降级可减到 15",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.5,
        help="CFG scale（默认 4.5）。SD3 推荐 3.5~7.0",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="输出高度（默认 1024）。建议为 64 的倍数。降级可设 768",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="输出宽度（默认 1024）。建议为 64 的倍数。降级可设 768",
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
        help="推理精度（默认 fp16）。CUDA 推荐 fp16",
    )

    # SD3 specific
    parser.add_argument(
        "--no_t5",
        action="store_true",
        default=True,
        help="关闭 T5-XXL 文本编码器（默认关闭）。12GB 设备必须关闭 T5",
    )
    parser.add_argument(
        "--use_t5",
        action="store_false",
        dest="no_t5",
        help="启用 T5-XXL（不推荐：12GB 设备几乎必然 OOM）",
    )
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
        help="禁用 CPU offload",
    )
    parser.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        default=True,
        help="启用 VAE slicing（默认启用）。降低 VAE decode VRAM",
    )
    parser.add_argument(
        "--no-enable_vae_slicing",
        action="store_false",
        dest="enable_vae_slicing",
        help="禁用 VAE slicing",
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
def run_sd3_medium_inference(args: argparse.Namespace) -> dict:
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
    print(f"[INFO] T5-XXL: {'关闭' if args.no_t5 else '启用（谨慎！12GB 可能 OOM）'}")
    print(f"[INFO] CPU offload: {args.enable_cpu_offload}")
    print(f"[INFO] VAE slicing: {args.enable_vae_slicing}")

    if not check_cuda_available():
        print("[WARN] CUDA 不可用。SD3 pipeline 在 MPS/CPU 上无法运行。")
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
        # SD3 的 from_pretrained 加载所有三个 text encoder。
        # no-T5 路径：加载后删除 text_encoder_3
        print(f"[INFO] 加载 pipeline: {args.model_id} ...")
        pipeline = diffusers.StableDiffusion3Pipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
        )

        # 关闭 T5
        if args.no_t5 and hasattr(pipeline, "text_encoder_3"):
            print("[INFO] 关闭 T5-XXL（text_encoder_3）...")
            pipeline.text_encoder_3 = None
            if hasattr(pipeline, "tokenizer_3"):
                pipeline.tokenizer_3 = None

        pipeline = pipeline.to("cuda")

        # 启用优化
        if args.enable_cpu_offload:
            print("[INFO] 启用 model CPU offload ...")
            pipeline.enable_model_cpu_offload()
        if args.enable_vae_slicing:
            print("[INFO] 启用 VAE slicing ...")
            pipeline.enable_vae_slicing()

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
        output_path = os.path.join(args.output_dir, f"sd3_medium_{timestamp}_{seed}.png")
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
            print("[HINT] OOM — 尝试降级:")
            print("  1. --height 768 --width 768")
            print("  2. --num_steps 15")
            print("  3. 确认 --no_t5 已启用（默认启用）")
            print("  4. 尝试 run_sana_if_possible.py（Sana 更省显存）")
        elif "401" in str(exc) or "403" in str(exc) or "gated" in str(exc).lower():
            print("[HINT] 权限问题 — SD3 Medium 是 gated 模型:")
            print("  1. 确认 huggingface-cli login 已执行")
            print("  2. 访问 https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers")
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
    result = run_sd3_medium_inference(args)

    summary_path = os.path.join(args.output_dir, "sd3_medium_result_summary.txt")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"SD3 Medium Inference Result\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Resolution: {args.height}x{args.width}\n")
        f.write(f"Steps: {args.num_steps}, CFG: {args.cfg_scale}\n")
        f.write(f"Dtype: {args.dtype}, Seed: {args.seed}\n")
        f.write(f"T5: {'disabled' if args.no_t5 else 'enabled'}\n")
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
