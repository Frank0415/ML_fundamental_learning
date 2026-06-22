#!/usr/bin/env python3
"""run_qwen_vl_reference.py — Task 10: 中等显存配置 约束下的小型 VLM reference 矩阵。

4 候选模型，按主路径 → 稳定 fallback → 先进对照 → 轻量对照顺序尝试。
任何单个模型失败不中断脚本；4 个全失败时执行降级 smoke（tokenizer-only）。

模型枚举硬编码，与 plan 完全一致：
  Qwen/Qwen3-VL-4B-Instruct
  Qwen/Qwen2.5-VL-3B-Instruct
  OpenGVLab/InternVL3_5-4B
  HuggingFaceTB/SmolVLM2-2.2B-Instruct

用法：
  python run_qwen_vl_reference.py \
    --image sample_images/demo.jpg \
    --prompt "请描述这张图片。" \
    --max-new-tokens 64
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 4 model candidates — hard-coded, not extensible
MODELS = [
    {
        "key": "qwen3_vl_4b",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "role": "primary",
        "description": "Qwen3-VL-4B-Instruct — 主路径",
    },
    {
        "key": "qwen2_5_vl_3b",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "role": "stable_fallback",
        "description": "Qwen2.5-VL-3B-Instruct — 稳定 fallback",
    },
    {
        "key": "internvl3_5_4b",
        "model_id": "OpenGVLab/InternVL3_5-4B",
        "role": "advanced_comparison",
        "description": "InternVL3.5-4B — 先进对照",
    },
    {
        "key": "smolvlm2_2b",
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "role": "lightweight_comparison",
        "description": "SmolVLM2-2.2B — 轻量对照",
    },
]

# Smoke model (tokenizer-only, guaranteed lightweight)
SMOKE_MODEL_ID = "Qwen/Qwen3-0.6B"

DEFAULT_PROMPTS = [
    "请描述这张图片。",
    "图片里有什么文字？",
]
DEFAULT_MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_info() -> dict:
    """Collect GPU availability / memory info."""
    import torch
    info: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    try:
        info["mps_available"] = torch.backends.mps.is_available()
    except Exception:
        info["mps_available"] = False
    try:
        if torch.cuda.is_available():
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_mem / (1024**3)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_mem_gb": round(mem_gb, 2),
                })
    except Exception:
        pass
    return info


def _image_info(image_path: str) -> dict:
    """Collect image metadata."""
    from PIL import Image
    img = Image.open(image_path)
    return {
        "path": str(image_path),
        "size": list(img.size),
        "mode": img.mode,
        "format": img.format,
    }


def _make_messages(image_path: str, prompt: str):
    """Build multimodal chat messages for a VLM processor."""
    # Use a generic multimodal content block format.
    # Most VLMs (Qwen, InternVL, SmolVLM/Idefics) accept this structure
    # via their chat template.  The image value is a path string; the processor
    # will load it internally when apply_chat_template(tokenize=True).
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_reference(model_key: str, model_id: str, image_path: str,
                  prompts: list[str], max_new_tokens: int) -> dict:
    """Attempt to load *model_id*, answer *prompts*, and return structured result.

    Returns a dict with status "ok" or "fail".  Never raises — all errors are
    caught and recorded.
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    start = time.time()
    result: dict[str, object] = {
        "model_key": model_key,
        "model_id": model_id,
        "status": "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_info": _gpu_info(),
        "image_info": _image_info(image_path),
        "max_new_tokens": max_new_tokens,
    }

    try:
        # ---- Step 1: Load model ----
        print(f"  加载模型: {model_id} ...")
        model_load_start = time.time()
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        model_load_sec = round(time.time() - model_load_start, 1)
        result["model_load_sec"] = model_load_sec
        print(f"    模型加载完成 ({model_load_sec}s)")

        # ---- Step 2: Load processor ----
        processor_load_start = time.time()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor_load_sec = round(time.time() - processor_load_start, 1)
        result["processor_load_sec"] = processor_load_sec
        result["processor_class"] = type(processor).__name__
        print(f"    处理器加载完成 ({processor_load_sec}s) — {type(processor).__name__}")

        # ---- Step 3: Generate for each prompt ----
        outputs = []
        for prompt in prompts:
            prompt_start = time.time()
            print(f"    提示词: {prompt[:60]}...")

            messages = _make_messages(image_path, prompt)

            # Build tokenized inputs via chat template
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception:
                # Fallback: some processors don't support apply_chat_template
                # with images or accept a different signature.
                img = Image.open(image_path).convert("RGB")
                inputs = processor(text=[prompt], images=img, return_tensors="pt")

            # Move to model device
            if hasattr(inputs, "to"):
                inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            # Decode only the new tokens
            if hasattr(inputs, "input_ids"):
                input_len = inputs.input_ids.shape[1]  # type: ignore[union-attr]
                new_ids = generated_ids[0, input_len:]
            else:
                new_ids = generated_ids[0]

            output_text = processor.batch_decode(
                [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            prompt_sec = round(time.time() - prompt_start, 2)
            output_entry = {
                "prompt": prompt,
                "output_text": output_text.strip(),
                "generation_sec": prompt_sec,
                "output_tokens": int(new_ids.shape[0]),
            }
            outputs.append(output_entry)
            print(f"      输出 ({prompt_sec}s, {new_ids.shape[0]} tokens): {output_text.strip()[:120]}")

        result["outputs"] = outputs

        # ---- Step 4: Record GPU memory if available ----
        if torch.cuda.is_available():
            result["gpu_memory"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
            }

        result["status"] = "ok"
        result["total_sec"] = round(time.time() - start, 1)
        print(f"    ✅ 成功 ({result['total_sec']}s)")

    except ImportError as e:
        result["error_type"] = "ImportError"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"    ❌ ImportError: {e}")
    except OSError as e:
        result["error_type"] = "OSError"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"    ❌ OSError: {e}")
    except MemoryError as e:
        result["error_type"] = "OutOfMemoryError"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"    ❌ OutOfMemoryError: {e}")
    except ValueError as e:
        result["error_type"] = "ValueError"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"    ❌ ValueError: {e}")
    except Exception as e:
        # Catch-all: classify by name
        ename = type(e).__name__
        if "oom" in ename.lower() or "memory" in ename.lower():
            result["error_type"] = f"OutOfMemoryError({ename})"
        else:
            result["error_type"] = ename
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"    ❌ {ename}: {e}")

    finally:
        result["total_sec"] = round(time.time() - start, 1)
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def run_smoke() -> dict:
    """Degradation smoke: load a tiny tokenizer-only model to prove the env works."""
    import torch
    from transformers import AutoTokenizer

    start = time.time()
    result: dict[str, object] = {
        "model_id": SMOKE_MODEL_ID,
        "status": "fail",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_info": _gpu_info(),
        "description": "降级 smoke — tokenizer-only（不加载 VLM 权重）",
    }

    try:
        print(f"\n  🔥 降级 smoke: 加载 tokenizer {SMOKE_MODEL_ID} ...")
        tokenizer = AutoTokenizer.from_pretrained(SMOKE_MODEL_ID)
        result["tokenizer_class"] = type(tokenizer).__name__
        result["vocab_size"] = tokenizer.vocab_size

        # Probe: encode a simple Chinese/English string
        test_str = "请描述这张图片。 What is in the image?"
        tokens = tokenizer.encode(test_str)
        decoded = tokenizer.decode(tokens)
        result["smoke_test"] = {
            "input": test_str,
            "num_tokens": len(tokens),
            "decoded": decoded,
        }
        result["status"] = "ok"
        print(f"    ✅ tokenizer smoke 通过 (vocab={tokenizer.vocab_size}, tokens={len(tokens)})")
    except Exception as e:
        result["error_type"] = type(e).__name__
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"    ❌ tokenizer smoke 失败: {e}")

    result["total_sec"] = round(time.time() - start, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Task 10: VLM reference 矩阵 — 4 候选模型 + 降级 smoke"
    )
    parser.add_argument(
        "--image",
        default=str(_SCRIPT_DIR / "sample_images" / "demo.jpg"),
        help="输入图片路径",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="提示词（可多次指定；默认两个中文问题）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"最大生成 token 数（默认 {DEFAULT_MAX_NEW_TOKENS}）",
    )
    parser.add_argument(
        "--model-key",
        choices=[m["key"] for m in MODELS],
        default=None,
        help="仅尝试指定模型（默认尝试全部 4 个）",
    )
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    if not os.path.isfile(args.image):
        print(f"错误: 图片不存在 — {args.image}")
        sys.exit(1)

    # Try the 4 models
    models_to_try = MODELS if args.model_key is None else [m for m in MODELS if m["key"] == args.model_key]

    print("=" * 60)
    print("Task 10: VLM Reference 矩阵")
    print(f"图片: {args.image}")
    print(f"提示词: {prompts}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"模型候选数: {len(models_to_try)}")
    print("=" * 60)

    gpu = _gpu_info()
    print(f"\nGPU 信息: CUDA={gpu.get('cuda_available')}, MPS={gpu.get('mps_available')}, "
          f"device_count={gpu.get('device_count')}")

    ok_models = []
    fail_models = []

    for m in models_to_try:
        print(f"\n{'─' * 50}")
        print(f"▶ {m['description']}")
        print(f"  model_id: {m['model_id']}")
        print(f"  role: {m['role']}")

        result = run_reference(
            model_key=m["key"],
            model_id=m["model_id"],
            image_path=args.image,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )

        # Save result
        suffix = "ok" if result["status"] == "ok" else "fail"
        json_path = RESULTS_DIR / f"{m['key']}.{suffix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  📄 结果已保存: {json_path}")

        if result["status"] == "ok":
            ok_models.append(m)
        else:
            fail_models.append(m)

    # ---- Gate: success path or degradation path ----
    print(f"\n{'=' * 60}")
    print(f"Gate 结果: 成功 {len(ok_models)} / 失败 {len(fail_models)}")

    smoke_result = None
    if not ok_models:
        print(f"\n⚠️  全部 {len(fail_models)} 个模型加载失败，执行降级 smoke ...")
        smoke_result = run_smoke()

        smoke_path = RESULTS_DIR / "processor_only_smoke.json"
        with open(smoke_path, "w", encoding="utf-8") as f:
            json.dump(smoke_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  📄 降级 smoke 结果已保存: {smoke_path}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("最终摘要:")
    for m in ok_models:
        print(f"  ✅ {m['key']} ({m['role']})")
    for m in fail_models:
        print(f"  ❌ {m['key']} ({m['role']})")
    if smoke_result:
        print(f"  🔥 smoke: {'✅' if smoke_result['status'] == 'ok' else '❌'}")

    overall = "ok" if ok_models else ("degraded" if smoke_result and smoke_result["status"] == "ok" else "fail")
    print(f"\n  OVERALL: {overall}")
    print(f"{'=' * 60}")

    # Exit code reflects whether at least one model succeeded (or smoke passed)
    if ok_models or (smoke_result and smoke_result["status"] == "ok"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
