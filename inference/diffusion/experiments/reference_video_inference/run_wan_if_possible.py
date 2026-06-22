#!/usr/bin/env python3
"""
run_wan_if_possible.py — Wan2.1-T2V-1.3B 视频生成（T15）

尝试在 中等显存配置 VRAM 下使用 diffusers.WanPipeline 生成短视频。
默认小规格：16 帧、256×256、8 步、bf16、cpu_offload。

优先级：T15 第三选模型（1.3B 轻量参数，但 480p×81 帧需 ~10GB，中等显存极限）。

前置条件：
  1. Python 3.13+ 环境已通过 `uv sync` 安装依赖
  2. HF token 已配置（huggingface-cli login）
  3. 已在 https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B 接受许可协议
  4. 可用的 CUDA GPU 上有 ≥10GB 空闲 VRAM

不强制真的能跑 — 缺失依赖或无法加载模型时会记录 blocker 并 exit。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime


def get_timestamp() -> str:
    """返回短时间戳，用于文件名。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def check_dependencies() -> list[str]:
    """检查必要依赖，返回缺失的包名列表。"""
    missing = []
    for pkg in ["torch", "diffusers", "imageio"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def infer_wan(args: argparse.Namespace) -> int:
    """执行 Wan2.1 推理。返回 0 表示成功，非 0 表示失败。"""

    # ---- 依赖检查 ----
    missing = check_dependencies()
    if missing:
        blocker_msg = f"Blocker: 缺少依赖包 {missing}。请运行 `uv sync` 安装。"
        print(f"[BLOCKER] {blocker_msg}", file=sys.stderr)
        _write_blocker(
            output_dir=args.output_dir,
            model_name="Wan",
            model_id=args.model_id,
            reason=f"依赖缺失: {missing}",
            details=blocker_msg,
        )
        return 1

    import torch
    import diffusers

    # ---- CUDA 检查 ----
    if not torch.cuda.is_available():
        blocker_msg = "Blocker: CUDA 不可用。本脚本需要在 NVIDIA GPU 上运行。Mac M5 不支持 CUDA。"
        print(f"[BLOCKER] {blocker_msg}", file=sys.stderr)
        _write_blocker(
            output_dir=args.output_dir,
            model_name="Wan",
            model_id=args.model_id,
            reason="CUDA 不可用",
            details=blocker_msg,
        )
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"[INFO] 设备: {device_name}")
    total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"[INFO] GPU 总显存: {total_mem:.1f} GB")

    # ---- dtype 映射 ----
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # ---- 构建提示信息 ----
    print(f"[INFO] 模型 ID: {args.model_id}")
    print(f"[INFO] 参数: dtype={args.dtype}, res={args.width}x{args.height}, "
          f"frames={args.num_frames}, steps={args.num_steps}, "
          f"cfg_scale={args.cfg_scale}, seed={args.seed}")
    print(f"[INFO] offload: cpu_offload={args.enable_cpu_offload}, "
          f"vae_tiling={args.enable_vae_tiling}")
    print(f"[INFO] 输出目录: {args.output_dir}")

    # ---- 加载 Pipeline ----
    print("[STEP 1/3] 加载 WanPipeline ...")
    t_load_start = time.perf_counter()

    # Wan2.1 在 diffusers 中的 pipeline 类名为 WanPipeline
    # 如果不存在，尝试 WanVideoPipeline 或直接查 diffusers
    pipeline_cls = None
    for cls_name in ["WanPipeline", "WanVideoPipeline"]:
        if hasattr(diffusers, cls_name):
            pipeline_cls = getattr(diffusers, cls_name)
            break

    if pipeline_cls is None:
        blocker_msg = (
            "模型加载失败: 当前 diffusers 版本中未找到 WanPipeline 或 WanVideoPipeline。"
            " 请确认 diffusers >= 0.32.0 已安装，或检查 HF 文档中 Wan 的 pipeline 类名。"
        )
        print(f"[BLOCKER] {blocker_msg}", file=sys.stderr)
        _write_blocker(
            output_dir=args.output_dir,
            model_name="Wan",
            model_id=args.model_id,
            reason="pipeline 类未找到",
            details=blocker_msg,
        )
        return 1

    try:
        pipe = pipeline_cls.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
        )
    except Exception as e:
        blocker_msg = f"模型加载失败: {type(e).__name__}: {e}"
        print(f"[BLOCKER] {blocker_msg}", file=sys.stderr)
        _write_blocker(
            output_dir=args.output_dir,
            model_name="Wan",
            model_id=args.model_id,
            reason=f"模型加载失败: {type(e).__name__}",
            details=f"{blocker_msg}\n\n{traceback.format_exc()}",
        )
        return 1

    t_load_end = time.perf_counter()
    print(f"[INFO] 模型加载耗时: {t_load_end - t_load_start:.1f}s")

    if args.enable_vae_tiling:
        try:
            pipe.vae.enable_tiling()
            print("[INFO] VAE tiling 已启用")
        except AttributeError:
            print("[WARN] 此版本的 Wan pipeline 不支持 VAE tiling")

    if args.enable_cpu_offload:
        try:
            pipe.enable_model_cpu_offload()
            print("[INFO] model_cpu_offload 已启用")
        except Exception as e:
            print(f"[WARN] cpu_offload 失败: {e}，fallback 到 GPU 加载")
            pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cuda")

    # ---- 推理 ----
    print("[STEP 2/3] 开始去噪推理 ...")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    t_infer_start = time.perf_counter()

    try:
        # Wan pipeline 的 API 可能与其他 pipeline 略有不同
        output = pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
            generator=generator,
        )
        video_frames = output.frames[0]
    except torch.cuda.OutOfMemoryError as e:
        blocker_msg = (
            f"OOM: CUDA 显存不足。"
            f"尝试配置: {args.width}x{args.height}, {args.num_frames} 帧, "
            f"{args.num_steps} 步, dtype={args.dtype}, "
            f"cpu_offload={args.enable_cpu_offload}。"
            f"请按 README 降级路径（Level 1 → Level 2 → Level 3）逐步尝试。"
        )
        print(f"[BLOCKER] {blocker_msg}", file=sys.stderr)
        _write_blocker(
            output_dir=args.output_dir,
            model_name="Wan",
            model_id=args.model_id,
            reason="OOM",
            details=f"{blocker_msg}\n\n{traceback.format_exc()}",
        )
        return 1
    except Exception as e:
        blocker_msg = f"推理失败: {type(e).__name__}: {e}"
        print(f"[BLOCKER] {blocker_msg}", file=sys.stderr)
        _write_blocker(
            output_dir=args.output_dir,
            model_name="Wan",
            model_id=args.model_id,
            reason=f"推理失败: {type(e).__name__}",
            details=f"{blocker_msg}\n\n{traceback.format_exc()}",
        )
        return 1

    t_infer_end = time.perf_counter()
    infer_time = t_infer_end - t_infer_start
    print(f"[INFO] 推理耗时: {infer_time:.1f}s")

    # ---- 峰值 VRAM ----
    peak_vram_bytes = torch.cuda.max_memory_allocated()
    peak_vram_gb = peak_vram_bytes / (1024**3)
    print(f"[INFO] 峰值 VRAM: {peak_vram_gb:.2f} GB ({peak_vram_bytes:,} bytes)")

    # ---- 保存视频 ----
    print("[STEP 3/3] 保存输出视频 ...")
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = get_timestamp()
    output_path = os.path.join(
        args.output_dir,
        f"wan_{args.num_frames}f_{args.height}p_s{args.seed}_{timestamp}.mp4",
    )

    try:
        diffusers.utils.export_to_video(video_frames, output_path, fps=8)
    except Exception:
        import imageio
        import numpy as np

        frames_pil = []
        for frame in video_frames:
            if hasattr(frame, "save"):
                import io

                buf = io.BytesIO()
                frame.save(buf, format="PNG")
                buf.seek(0)
                from PIL import Image as PILImage

                frames_pil.append(PILImage.open(buf))
            else:
                import torchvision.transforms.functional as TF

                frames_pil.append(TF.to_pil_image(frame))

        imageio.mimsave(output_path, [np.array(f) for f in frames_pil], fps=8)

    print(f"[OK] 视频已保存: {output_path}")

    # ---- 保存 profile JSON ----
    profile_data = {
        "model": "Wan2.1-T2V-1.3B",
        "model_id": args.model_id,
        "device": device_name,
        "dtype": args.dtype,
        "resolution": f"{args.width}x{args.height}",
        "num_frames": args.num_frames,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "cpu_offload": args.enable_cpu_offload,
        "vae_tiling": args.enable_vae_tiling,
        "load_time_s": round(t_load_end - t_load_start, 2),
        "infer_time_s": round(infer_time, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "peak_vram_bytes": peak_vram_bytes,
        "output_path": output_path,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
    }

    profile_path = os.path.join(args.output_dir, "wan_profiling.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Profile 已保存: {profile_path}")

    torch.cuda.reset_peak_memory_stats()
    return 0


def _write_blocker(
    output_dir: str,
    model_name: str,
    model_id: str,
    reason: str,
    details: str,
) -> None:
    """将 blocker 信息写入 results/blocker_<model>.md。"""
    os.makedirs(output_dir, exist_ok=True)

    safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
    blocker_path = os.path.join(output_dir, f"blocker_{safe_name}.md")

    content = f"""# Reference Video Inference — Blocker ({model_name})

**日期**：{datetime.now().strftime("%Y-%m-%d")}
**模型**：{model_id}
**设备**：可用的 CUDA GPU（中等显存配置 VRAM）（预期设备，实际可能为其他）
**执行者**：T15 系统尝试

## 失败原因
{reason}

## 详细日志
```
{details[:2000]}
```

## 结论
该模型在本次尝试中因上述原因未能完成推理。请检查前置条件（README 第 3 节）后重试。

## 对后续的建议
- 若依赖缺失：`uv sync` 安装依赖
- 若授权未通过：检查 HF token 和 Wan2.1 许可协议接受状态
- 若 CUDA 不可用：在远程 CUDA GPU 上运行
- 若 OOM：Wan2.1 在中等显存配置下是极限操作，可能需要 480p 配置 + 全部 offload
"""
    with open(blocker_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] Blocker 记录已保存: {blocker_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B — 在 中等显存配置 VRAM 下尝试视频生成。"
        " 默认小规格：16 帧 × 256×256 × 8 步。"
        " 失败则记录 blocker 并 exit(1)。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
前置条件（运行前必须完成）:
  1. `uv sync` 安装依赖（torch, diffusers, imageio）
  2. `huggingface-cli login` 配置 HF token
  3. 在 https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B 接受许可协议
  4. 可用的 CUDA GPU 上有 ≥10GB 空闲 VRAM

降级路径 (OOM 时逐级尝试):
  Level 0 (默认): --num_frames 16 --height 256 --width 256 --num_steps 8 --dtype bf16 --enable_cpu_offload
  Level 1:        --num_frames 12 --height 240 --width 240 --num_steps 6 --dtype fp16 --enable_cpu_offload
  Level 2:        --num_frames 8  --height 192 --width 192 --num_steps 4 --dtype fp16 --enable_cpu_offload
  Level 3:        记录 blocker，停止尝试

中等显存配置 现实路径:
  - Wan2.1-1.3B 参数量最小（1.3B），但 3D VAE 和 text encoder 额外显存大。
  - 480p × 5s（81 帧）→ VAE latent 约 (1,16,21,60,104) → 大量 token。
  - 在中等显存配置下，默认小规格 (16f×256²) 约 8-10GB，属于极限操作。
  - 必须开启 cpu_offload，推荐 bfloat16 降低显存。
  - Wan 是 4 个模型中对 中等显存配置 最不友好的（小而密架构 → 中间激活大）。

Wan2.1 特有参数:
  - cfg_scale: 官方推荐 5.0
  - 常规配置: 50 步、81 帧、832×480
  - 小规格配置: 8 步、16 帧、256×256（默认）

示例:
  # 最小尝试
  python run_wan_if_possible.py --prompt "一只白猫在草地上缓步走向镜头"

  # 指定所有参数
  python run_wan_if_possible.py \\
    --prompt "一只白猫在草地上缓步走向镜头" \\
    --num_frames 16 --height 256 --width 256 --num_steps 8 \\
    --cfg_scale 5.0 --seed 42 --dtype bf16 \\
    --enable_cpu_offload --enable_vae_tiling \\
    --output_dir results/
""",
    )

    # 运行参数
    parser.add_argument("--prompt", type=str, required=True, help="视频生成提示词（必填）")
    parser.add_argument("--num_frames", type=int, default=16, help="生成帧数（默认 16）")
    parser.add_argument("--height", type=int, default=256, help="视频高度（默认 256）")
    parser.add_argument("--width", type=int, default=256, help="视频宽度（默认 256）")
    parser.add_argument("--num_steps", type=int, default=8, help="推理步数（默认 8，常规 50）")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale（默认 5.0）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                        help="模型 dtype（默认 bf16）")

    # 优化开关
    parser.add_argument("--enable_cpu_offload", action="store_true", default=True,
                        help="启用 model_cpu_offload（默认 True）")
    parser.add_argument("--no_cpu_offload", action="store_false", dest="enable_cpu_offload",
                        help="禁用 cpu_offload")
    parser.add_argument("--enable_vae_tiling", action="store_true", default=True,
                        help="启用 VAE tiling（默认 True）")
    parser.add_argument("--no_vae_tiling", action="store_false", dest="enable_vae_tiling",
                        help="禁用 VAE tiling")

    # 模型与输出
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B",
                        help="HF 模型 ID（默认 Wan-AI/Wan2.1-T2V-1.3B）")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="输出目录（默认 results/）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ret = infer_wan(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
