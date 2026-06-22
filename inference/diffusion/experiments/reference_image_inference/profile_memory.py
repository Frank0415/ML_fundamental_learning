#!/usr/bin/env python3
"""profile_memory.py — 显存预估与 profiling 工具（默认 dry-run 模式）。

**功能**：
  - 根据模型、resolution、steps、dtype 预估 VRAM 需求。
  - 默认 dry-run（--dry-run 默认 True），只打印预测结果，不跑真实模型。
  - 若 --no-dry-run 且环境就绪（CUDA + diffusers），可调用相应脚本实测。

**输出**：
  - stdout：预估/实测表格 + 降级建议
  - JSON：results/<name>_profile.json

**VRAM 预估公式**（基于官方文档与社区经验，仅供参考）：
  - Sana-0.6B:  base ~2.5GB + (resolution² / 1024²) * 0.002GB + steps * 0.05GB
  - SD3-Medium (no-T5): base ~3.5GB + (resolution² / 1024²) * 0.003GB + steps * 0.08GB
  - FLUX.1-schnell: base ~8GB + (resolution² / 1024²) * 0.004GB + steps * 0.1GB
  以上公式为 rough estimate，实际值取决于 batch size、attention 实现等。

**降级建议表**（中等显存配置 预算，85% 有效 = 10.2GB）：
  | 预估 VRAM | 颜色 | 建议 |
  |-----------|------|------|
  | < 8 GB    | 🟢   | yes — 安全可跑 |
  | 8~10 GB   | 🟡   | maybe — 开启 offload 后可尝试 |
  | > 10 GB   | 🔴   | no — 需降级或切换模型 |
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# 0. 模型元数据
# ---------------------------------------------------------------------------
MODEL_META = {
    "sana": {
        "name": "Sana-0.6B",
        "model_id": "Efficient-Large-Model/Sana_600M_1024px_diffusers",
        "base_vram_gb": 2.5,
        "res_coeff": 0.002,   # GB per (Mpx)
        "step_coeff": 0.05,   # GB per step
        "gated": False,
        "license": "Apache 2.0",
        "min_steps": 10,
        "max_steps": 50,
        "recommended_steps": 20,
        "recommended_res": 1024,
        "recommended_dtype": "bf16",
    },
    "sd3": {
        "name": "SD3-Medium (no-T5)",
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "base_vram_gb": 3.5,
        "res_coeff": 0.003,
        "step_coeff": 0.08,
        "gated": True,
        "license": "Stability AI Community License",
        "min_steps": 15,
        "max_steps": 50,
        "recommended_steps": 28,
        "recommended_res": 1024,
        "recommended_dtype": "fp16",
        "note": "必须关闭 T5-XXL（--no_t5）",
    },
    "flux": {
        "name": "FLUX.1-schnell",
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "base_vram_gb": 8.0,
        "res_coeff": 0.004,
        "step_coeff": 0.1,
        "gated": True,
        "license": "Apache 2.0 (gated)",
        "min_steps": 4,
        "max_steps": 8,
        "recommended_steps": 4,
        "recommended_res": 1024,
        "recommended_dtype": "fp16",
        "note": "在中等显存配置下偏紧，推荐 sequential CPU offload 或 GGUF Q4 量化",
    },
}

VRAM_BUDGET_GB = 10.2  # 中等显存配置 × 0.85
GPU_NAME = "可用的 CUDA GPU"
GPU_TOTAL_GB = 12.0


# ---------------------------------------------------------------------------
# 1. 预估 VRAM
# ---------------------------------------------------------------------------
def estimate_vram(model_key: str, resolution: int, steps: int, dtype: str) -> dict:
    """基于经验公式预估 VRAM。返回 dict 含 estimated_gb / color / verdict。"""
    meta = MODEL_META[model_key]
    mpx = (resolution ** 2) / (1024 ** 2)  # megapixels relative to 1024²
    estimated = meta["base_vram_gb"] + meta["res_coeff"] * mpx + meta["step_coeff"] * steps

    # dtype 修正
    if dtype == "fp32":
        estimated *= 1.6
    elif dtype == "bf16":
        estimated *= 0.95

    verdict = ""
    color = ""
    if estimated < 8.0:
        color = "🟢"
        verdict = "yes — 安全可跑"
    elif estimated < VRAM_BUDGET_GB:
        color = "🟡"
        verdict = "maybe — 开启 offload 后可尝试"
    else:
        color = "🔴"
        verdict = "no — 需降级或切换模型"

    return {
        "model": meta["name"],
        "model_id": meta["model_id"],
        "resolution": resolution,
        "steps": steps,
        "dtype": dtype,
        "estimated_vram_gb": round(estimated, 2),
        "vram_budget_gb": VRAM_BUDGET_GB,
        "color": color,
        "verdict": verdict,
    }


def get_downgrade_advice(model_key: str, current_res: int, current_steps: int) -> list[dict]:
    """生成逐级降级建议。"""
    meta = MODEL_META[model_key]
    advice = []
    max_res = current_res
    max_steps = current_steps

    # Level 1: 降 resolution（但不过 512）
    new_res = max_res // 2
    while new_res >= 512:
        est = estimate_vram(model_key, new_res, current_steps, meta["recommended_dtype"])
        advice.append({
            "level": len(advice) + 1,
            "action": f"降 resolution: {max_res} → {new_res}",
            "estimated_vram_gb": est["estimated_vram_gb"],
            "color": est["color"],
        })
        max_res = new_res
        new_res = max_res // 2

    # Level 2: 减 steps（但不过 meta["min_steps"]）
    new_steps = max(current_steps // 2, meta["min_steps"])
    if new_steps < current_steps:
        est = estimate_vram(model_key, current_res, new_steps, meta["recommended_dtype"])
        advice.append({
            "level": len(advice) + 1,
            "action": f"减 steps: {current_steps} → {new_steps}",
            "estimated_vram_gb": est["estimated_vram_gb"],
            "color": est["color"],
        })

    # Level 3: 启用 offload（无法量化预估，提示即可）
    advice.append({
        "level": len(advice) + 1,
        "action": "启用 CPU offload / sequential offload / VAE tiling",
        "estimated_vram_gb": "取决于 offload 效果",
        "color": "🟡",
    })

    # Level 4: 切更小模型
    if model_key == "flux":
        advice.append({
            "level": len(advice) + 1,
            "action": "切模型: FLUX → SD3 Medium (no-T5) → Sana-0.6B",
            "estimated_vram_gb": "显著降低",
            "color": "🟢",
        })
    elif model_key == "sd3":
        advice.append({
            "level": len(advice) + 1,
            "action": "切模型: SD3 → Sana-0.6B",
            "estimated_vram_gb": "~2.5GB base",
            "color": "🟢",
        })
    elif model_key == "sana":
        advice.append({
            "level": len(advice) + 1,
            "action": "切模型: Sana-1.6B → Sana-0.6B（--model_id Efficient-Large-Model/Sana_600M_1024px_diffusers）",
            "estimated_vram_gb": "~2.5GB base",
            "color": "🟢",
        })

    return advice


# ---------------------------------------------------------------------------
# 2. 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="显存预估与 profiling 工具（默认 dry-run）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # dry-run 预估（默认，安全）
  python profile_memory.py --script sana --resolution 1024 --steps 20

  # 三个模型逐一预估
  python profile_memory.py --script sana,sd3,flux --resolution 1024

  # 指定 prompt + resolution 格式
  python profile_memory.py --script sd3 --prompt "一只柴犬" --resolution 1024x1024

  # 实测（需要 CUDA + diffusers 就绪）
  python profile_memory.py --script sana --no-dry-run --prompt "测试 prompt"

降级建议:
  若预估为 🔴，脚本会输出逐级降级计划（降 resolution → 减 steps → 开 offload → 换模型）。
""",
    )

    parser.add_argument(
        "--script",
        type=str,
        default="sana",
        help="模型选择: sana / sd3 / flux。逗号分隔可一次跑多个（默认 sana）",
    )
    parser.add_argument(
        "--pipeline_class",
        type=str,
        default="",
        help="（可选）直接指定 pipeline class 字符串，替代 --script",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="一只柴犬在樱花树下",
        help="测试 prompt（默认 '一只柴犬在樱花树下'）",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1024x1024",
        help="分辨率，格式 WxH（默认 1024x1024）。单独数字视为正方形",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=0,
        help="去噪步数（默认 0 = 使用各模型推荐值）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="recommended",
        choices=["fp32", "fp16", "bf16", "recommended"],
        help="推理精度（默认 recommended = 各模型默认值）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=True,
        help="dry-run 模式（默认启用）。只预估，不跑真实模型",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="实测模式。需要 CUDA + diffusers 就绪",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="输出目录（默认 results/）",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 3. 主逻辑
# ---------------------------------------------------------------------------
def parse_resolution(res_str: str) -> int:
    """解析 resolution 字符串为单边长度（取 max(W,H) 近似估算）。"""
    if "x" in res_str.lower():
        parts = res_str.lower().split("x")
        return max(int(parts[0]), int(parts[1]))
    return int(res_str)


def main() -> None:
    args = parse_args()

    # 解析 resolution
    resolution = parse_resolution(args.resolution)

    # 解析 model keys
    if args.pipeline_class:
        model_keys = [args.pipeline_class]
    else:
        model_keys = [k.strip() for k in args.script.split(",")]
        for k in model_keys:
            if k not in MODEL_META:
                print(f"[ERROR] 未知模型 '{k}'。可选: {list(MODEL_META.keys())}")
                sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"显存预估工具 — {GPU_NAME} ({GPU_TOTAL_GB:.0f}GB, 有效预算 {VRAM_BUDGET_GB:.1f}GB)")
    print(f"模式: {'dry-run（仅预估）' if args.dry_run else '实测（需 CUDA + diffusers）'}")
    print(f"Prompt: {args.prompt}")
    print(f"Resolution: {resolution}×{resolution}")
    print(f"{'=' * 60}\n")

    all_results = []

    for model_key in model_keys:
        meta = MODEL_META[model_key]

        # 确定 steps 和 dtype
        steps = args.num_steps if args.num_steps > 0 else meta["recommended_steps"]
        dtype = meta["recommended_dtype"] if args.dtype == "recommended" else args.dtype

        print(f"--- {meta['name']} ---")
        print(f"  Model ID: {meta['model_id']}")
        print(f"  License: {meta['license']}{' (gated — 需 HF token + accept license)' if meta['gated'] else ' (开放，无需授权)'}")
        print(f"  Dtype: {dtype}")
        print(f"  Steps: {steps}")
        print(f"  Resolution: {resolution}×{resolution}")

        # 预估 VRAM
        est = estimate_vram(model_key, resolution, steps, dtype)
        print(f"  预估 VRAM: {est['estimated_vram_gb']:.2f} GB / {VRAM_BUDGET_GB:.1f} GB")
        print(f"  判定: {est['color']} {est['verdict']}")

        result_entry = {
            "model": meta["name"],
            "model_id": meta["model_id"],
            "gated": meta["gated"],
            "license": meta["license"],
            "prompt": args.prompt,
            "resolution": resolution,
            "steps": steps,
            "dtype": dtype,
            "dry_run": args.dry_run,
            "estimated_vram_gb": est["estimated_vram_gb"],
            "vram_budget_gb": VRAM_BUDGET_GB,
            "verdict": est["verdict"],
        }

        # 降级建议
        if est["color"] == "🔴" or est["color"] == "🟡":
            print(f"\n  降级建议（{meta['name']}）：")
            downgrades = get_downgrade_advice(model_key, resolution, steps)
            result_entry["downgrade_advice"] = downgrades
            for dg in downgrades:
                print(f"    Level {dg['level']}: {dg['action']}")
                print(f"      → 预估: {dg['estimated_vram_gb']} [{dg['color']}]")
        else:
            print(f"  无需降级。")
            result_entry["downgrade_advice"] = []

        # 实测（若启用）
        if not args.dry_run:
            script_map = {
                "sana": "run_sana_if_possible.py",
                "sd3": "run_sd3_medium_if_possible.py",
                "flux": "run_flux_schnell_if_possible.py",
            }
            script = script_map.get(model_key)
            if script:
                print(f"\n  [实测模式] 将调用 {script}（需 CUDA + diffusers 就绪）")
                print(f"  命令: python {script} --prompt '{args.prompt}' "
                      f"--height {resolution} --width {resolution} --num_steps {steps} "
                      f"--dtype {dtype} --output_dir {args.output_dir}")
                print(f"  [TODO] 自动调用脚本的功能尚待实现。请手动执行上述命令。")
                result_entry["measured"] = None
            else:
                print(f"\n  [实测模式] 未知脚本映射: {model_key}")

        all_results.append(result_entry)
        print()

    # 写入 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"profile_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果已写入: {json_path}")

    # 总结表
    print(f"\n{'=' * 60}")
    print(f"总结（{GPU_NAME}, {VRAM_BUDGET_GB:.1f}GB 有效预算）")
    print(f"{'=' * 60}")
    print(f"{'模型':<22} {'Res':<6} {'Steps':<6} {'Dtype':<6} {'预估VRAM':<10} {'判定'}")
    print(f"{'-' * 60}")
    for r in all_results:
        color = "🟢" if r["estimated_vram_gb"] < 8.0 else ("🟡" if r["estimated_vram_gb"] < VRAM_BUDGET_GB else "🔴")
        print(f"{r['model']:<22} {r['resolution']:<6} {r['steps']:<6} {r['dtype']:<6} "
              f"{r['estimated_vram_gb']:<8.2f}GB   {color} {r['verdict']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
