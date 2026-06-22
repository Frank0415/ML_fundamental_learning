#!/usr/bin/env python3
"""
profile_video_memory.py — 视频推理 VRAM 预估与 Profile 工具（T15）

对指定模型/脚本进行 VRAM 估算。默认 --dry_run（不加载实体模型），
只基于模型公开参数和 latent shape 计算理论显存占用。

视频 VRAM 估算比图像复杂：
  - latent 多了 T 维（latent 大小 = C × T × H × W × bytes_per_element）
  - DiT attention 的 activation 内存 = N_tokens²（N_tokens = T × (H/patch) × (W/patch)）
  - 视频 VAE 的 3D conv 比 2D conv 多一层时间维度的卷积激活

公式（理论峰值 VRAM，不含框架开销）:
  VRAM_total ≈ weights + latent_buffer + attention_activation + vae_buffer + text_encoder
  其中:
    weights = model_params × bytes_per_param （fp16 下 ×2, bf16 下 ×2）
    latent_buffer = C × T × H_latent × W_latent × 2 × bytes_per_element (×2 含 CFG 双 forward)
    attention_activation = N_tokens² × num_heads × num_layers × bytes_per_element（粗略）
    vae_buffer ≈ C_out × T_out × H_pixel × W_pixel × 4 × bytes_per_element (×4 用于中间激活)
    text_encoder ≈ text_encoder_params × bytes_per_param

输出：results/<name>_profile.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime


# ---- 模型规格表（公开参数，非实测）----
MODEL_SPECS = {
    "ltx": {
        "name": "LTX-Video 2B distilled",
        "model_id": "Lightricks/LTX-Video",
        "params_billion": 2.0,
        "vae_spatial_compress": 8,
        "vae_temporal_compress": 1,
        "latent_channels": 128,  # LTX-Video 使用高通道 latent（128ch）
        "patch_size_t": 1,
        "patch_size_h": 2,
        "patch_size_w": 2,
        "num_heads": 24,
        "num_layers": 28,
        "text_encoder_params_billion": 0.3,
    },
    "cogvideox": {
        "name": "CogVideoX-2B",
        "model_id": "THUDM/CogVideoX-2b",
        "params_billion": 2.0,
        "vae_spatial_compress": 8,
        "vae_temporal_compress": 4,
        "latent_channels": 16,
        "patch_size_t": 1,
        "patch_size_h": 2,
        "patch_size_w": 2,
        "num_heads": 30,
        "num_layers": 30,
        "text_encoder_params_billion": 0.3,
    },
    "wan": {
        "name": "Wan2.1-T2V-1.3B",
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "params_billion": 1.3,
        "vae_spatial_compress": 8,
        "vae_temporal_compress": 4,
        "latent_channels": 16,
        "patch_size_t": 1,
        "patch_size_h": 2,
        "patch_size_w": 2,
        "num_heads": 12,
        "num_layers": 24,
        "text_encoder_params_billion": 1.5,  # Wan 用 T5 做 text encoder，约 1.5B
    },
}


def estimate_vram(
    spec: dict,
    num_frames: int,
    height: int,
    width: int,
    dtype: str,
    cfg_enabled: bool,
) -> dict:
    """
    基于模型规格和用户参数估算视频推理 VRAM。

    返回一个字典，包含各项细分明细和总和。
    """

    bytes_per_param = 2 if dtype in ("fp16", "bf16") else 4

    # ---- 1. 权重显存 ----
    weights_gb = spec["params_billion"] * 1e9 * bytes_per_param / (1024**3)

    # ---- 2. Text Encoder 显存 ----
    text_enc_gb = spec["text_encoder_params_billion"] * 1e9 * bytes_per_param / (1024**3)

    # ---- 3. Latent Shape 计算 ----
    # VAE 空间压缩后的分辨率
    h_latent = height // spec["vae_spatial_compress"]
    w_latent = width // spec["vae_spatial_compress"]
    # VAE 时间压缩后的帧数
    t_latent = num_frames // spec["vae_temporal_compress"]
    if t_latent < 1:
        t_latent = 1  # 最小 1 帧 latent

    # DiT patch 后的 token 数
    n_tokens_h = math.ceil(h_latent / spec["patch_size_h"])
    n_tokens_w = math.ceil(w_latent / spec["patch_size_w"])
    n_tokens_t = math.ceil(t_latent / spec["patch_size_t"])
    n_tokens_total = n_tokens_t * n_tokens_h * n_tokens_w

    # Latent buffer（CFG 双 forward 需要 ×2）
    latent_elements = spec["latent_channels"] * t_latent * h_latent * w_latent
    cfg_multiplier = 2 if cfg_enabled else 1
    latent_gb = latent_elements * cfg_multiplier * bytes_per_param / (1024**3)

    # ---- 4. Attention Activation（粗略估计）----
    # self-attention: O(N² × num_heads × num_layers × bytes)
    # 这是峰值 activation，通常在第一个 attention block 后开始
    activation_per_head = n_tokens_total * n_tokens_total * bytes_per_param
    activation_all_heads = activation_per_head * spec["num_heads"]
    # 粗略：每层保留前一层 activation，最终峰值约 2× 同时存活
    attn_activation_gb = activation_all_heads * 2 / (1024**3)

    # ---- 5. VAE Buffer ----
    # 视频 VAE decoder 需要还原到像素空间
    # 中间激活 ≈ C_out × T_out × H_pix × W_pix × 4 (中间 feature maps)
    vae_buffer_elements = 4 * num_frames * height * width * 4  # ×4 通道 ×4 buffer
    vae_buffer_gb = vae_buffer_elements * bytes_per_param / (1024**3)

    # ---- 6. 框架开销（约 10%）----
    subtotal = weights_gb + text_enc_gb + latent_gb + attn_activation_gb + vae_buffer_gb
    overhead_gb = subtotal * 0.10

    total_gb = subtotal + overhead_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "text_encoder_gb": round(text_enc_gb, 2),
        "latent_gb": round(latent_gb, 2),
        "attention_activation_gb": round(attn_activation_gb, 2),
        "vae_buffer_gb": round(vae_buffer_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_gb": round(total_gb, 2),
        "budget_12gb": round(total_gb, 2),
        "budget_remaining_gb": round(10.2 - total_gb, 2),
        "feasible_on_12gb": total_gb <= 10.2,
        # 中间计算信息
        "latent_shape": f"(B,{spec['latent_channels']},{t_latent},{h_latent},{w_latent})",
        "n_tokens_total": n_tokens_total,
        "cfg_double_forward": cfg_enabled,
        "dtype_bytes_per_param": bytes_per_param,
    }


def run_dry_profile(script_name: str, output_dir: str) -> int:
    """干跑 VRAM 预估（不加载模型）。"""

    spec = MODEL_SPECS.get(script_name)
    if spec is None:
        print(f"[ERROR] 未知脚本名: {script_name}。可选: {list(MODEL_SPECS.keys())}",
              file=sys.stderr)
        return 1

    # ---- 默认小规格估算 ----
    small_spec = estimate_vram(
        spec=spec,
        num_frames=16, height=256, width=256, dtype="bf16", cfg_enabled=True,
    )

    # ---- 极限降级规格估算 ----
    minimal_spec = estimate_vram(
        spec=spec,
        num_frames=8, height=192, width=192, dtype="fp16", cfg_enabled=False,
    )

    # ---- 官方常规规格估算 ----
    regular_configs = {
        "ltx": (121, 768, 512, "bf16"),
        "cogvideox": (49, 720, 480, "fp16"),
        "wan": (81, 832, 480, "bf16"),
    }
    reg_frames, reg_h, reg_w, reg_dtype = regular_configs.get(script_name, (16, 256, 256, "bf16"))
    regular_spec = estimate_vram(
        spec=spec,
        num_frames=reg_frames, height=reg_h, width=reg_w, dtype=reg_dtype,
        cfg_enabled=True,
    )

    profile_data = {
        "model": spec["name"],
        "model_id": spec["model_id"],
        "params_billion": spec["params_billion"],
        "profiling_method": "dry_run (公式估算，非实测)",
        "timestamp": datetime.now().isoformat(),
        "small_config": {
            "description": "默认小规格 (T15 推荐)",
            "num_frames": 16, "height": 256, "width": 256,
            "dtype": "bf16", "cfg_enabled": True,
            "vram_estimate": small_spec,
        },
        "minimal_config": {
            "description": "极限降级 (Level 2)",
            "num_frames": 8, "height": 192, "width": 192,
            "dtype": "fp16", "cfg_enabled": False,
            "vram_estimate": minimal_spec,
        },
        "regular_config": {
            "description": f"官方常规规格 ({spec['name']} 推荐)",
            "num_frames": reg_frames, "height": reg_h, "width": reg_w,
            "dtype": reg_dtype, "cfg_enabled": True,
            "vram_estimate": regular_spec,
        },
        "vram_formula_notes": {
            "weights": "params × bytes_per_param",
            "latent_buffer": "C×T×H_latent×W_latent ×2(CFG) × bytes_per_elem",
            "attention_activation": "N_tokens² × num_heads × 2 × bytes_per_param (粗略)",
            "vae_buffer": "4 × T × H_pixel × W_pixel × 4_channels × bytes_per_param",
            "text_encoder": "text_encoder_params × bytes_per_param",
            "overhead": "10% of subtotal for framework + CUDA context",
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, f"{script_name}_profiling.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Profile 已保存: {profile_path}")

    # ---- 人类可读摘要 ----
    print(f"\n{'='*60}")
    print(f"  {spec['name']} VRAM 预估（dry-run，公式估算）")
    print(f"{'='*60}")
    print(f"  参数规模: {spec['params_billion']}B, "
          f"VAE: {spec['vae_spatial_compress']}×空间/{spec['vae_temporal_compress']}×时间压缩")
    print(f"{'='*60}")
    for cfg_name, cfg_data in [
        ("小规格 (16f×256²)", small_spec),
        ("极限降级 (8f×192²)", minimal_spec),
        ("常规规格", regular_spec),
    ]:
        print(f"\n  [{cfg_name}]")
        print(f"    权重:        {cfg_data['weights_gb']:.2f} GB")
        print(f"    文本编码器:  {cfg_data['text_encoder_gb']:.2f} GB")
        print(f"    Latent:      {cfg_data['latent_gb']:.2f} GB")
        print(f"    Attn 激活:   {cfg_data['attention_activation_gb']:.2f} GB")
        print(f"    VAE buffer:  {cfg_data['vae_buffer_gb']:.2f} GB")
        print(f"    开销:        {cfg_data['overhead_gb']:.2f} GB")
        print(f"    ─────────────────────────")
        print(f"    总计:        {cfg_data['total_gb']:.2f} GB")
        print(f"    中等显存配置 可行性: {'✅ 可行' if cfg_data['feasible_on_12gb'] else '❌ 超预算'}")
        print(f"    剩余预算:    {cfg_data['budget_remaining_gb']:+.2f} GB")
        print(f"    Latent shape:{cfg_data['latent_shape']}")
        print(f"    DiT tokens:  {cfg_data['n_tokens_total']}")

    print(f"\n  ⚠️ 注意：以上为公式估算，实际 VRAM 可能因框架开销、")
    print(f"     attention 实现差异（flash-attn / sdpa / vanilla）而有 ±30% 偏差。")
    print(f"     CPU offload 可将 weights 部分转移到 CPU，大幅降低 GPU VRAM。")

    return 0


def run_real_profile(script_name: str, output_dir: str) -> int:
    """真实 profile（需要加载模型，目前为占位实现）。"""

    spec = MODEL_SPECS.get(script_name)
    if spec is None:
        print(f"[ERROR] 未知脚本名: {script_name}。可选: {list(MODEL_SPECS.keys())}",
              file=sys.stderr)
        return 1

    print(f"[INFO] 真实 profile 模式未实现。请使用 --dry_run 进行公式估算。")
    print(f"[INFO] 或在对应的 run_{script_name}_if_possible.py 中启用推理后查看 profiling.json。")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="视频推理 VRAM 预估工具。默认 --dry_run 模式，"
        "基于模型公开参数和 latent shape 公式估算显存占用。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式说明:
  --dry_run (默认): 公式估算，不加载实体模型。输出各配置（小规格/极限/常规）的 VRAM 分解。
  非 --dry_run: 真实 profile，需要加载模型并运行一次推理（当前为占位实现）。

VRAM 估算公式（视频特有，含 T 维度）:
  - weights: params × bytes_per_param (fp16→2, fp32→4)
  - latent_buffer: C × T_latent × H_latent × W_latent × 2(CFG) × bytes
    (T 维度使 video latent 比 image latent 大 T_latent 倍)
  - attention_activation: N_tokens² × num_heads × 2 × bytes
    (N_tokens = T_patches × H_patches × W_patches，视频 token 数是图像的 T_patches 倍)
  - vae_buffer: 4 × T_frames × H_pixel × W_pixel × 4ch × bytes
  - text_encoder: params × bytes_per_param
  - overhead: 10% 框架 + CUDA context 开销

示例:
  # 干跑所有三个模型的 VRAM 预估
  python profile_video_memory.py --script ltx --dry_run
  python profile_video_memory.py --script cogvideox --dry_run
  python profile_video_memory.py --script wan --dry_run

  # 指定输出目录
  python profile_video_memory.py --script ltx --dry_run --output_dir results/
""",
    )

    parser.add_argument(
        "--script", type=str, required=True,
        choices=list(MODEL_SPECS.keys()),
        help=f"目标模型脚本。可选: {list(MODEL_SPECS.keys())}",
    )
    parser.add_argument(
        "--pipeline_class", type=str, default=None,
        help="pipeline 类名字符串（备用，当前未实现。请使用 --script）",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=True,
        help="干跑模式，仅做公式估算（默认 True）",
    )
    parser.add_argument(
        "--no_dry_run", action="store_false", dest="dry_run",
        help="真实 profile（需加载模型，占位实现）",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/",
        help="输出目录（默认 results/）",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        return run_dry_profile(args.script, args.output_dir)
    else:
        return run_real_profile(args.script, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
