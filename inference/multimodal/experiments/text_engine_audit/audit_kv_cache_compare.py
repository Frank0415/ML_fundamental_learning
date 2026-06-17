#!/usr/bin/env python3
"""audit_kv_cache_compare.py — Verify KVCache-driven prefill+decode matches full recompute.

Compares logits from:
  1. full_compute(seq_len)  — one forward pass over all tokens
  2. cached_prefill_then_decode(seq_len) — prefill (write cache) + decode 1 token (read cache)

Modes:
  --mode smoke              Single seq_len=8 quick check
  --mode compare            Multi seq_len comparison (--seq-lens 1 8 64 512)
  --mode error_cases        Edge-case error handling tests

No HF weights required — uses random init Qwen3Config(**QWEN3_0_6B).
"""

import argparse
import json
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import Qwen3Config

# Ensure minivLLM is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]  # multimodal/
_MINIVLLM_ROOT = _PROJECT_ROOT / "minivLLM"
if str(_MINIVLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIVLLM_ROOT))

from minivllm.model.qwen3 import Qwen3
from minivllm.core.kv_cache import KVCache

RESULTS_DIR = _SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QWEN3_0_6B = dict(
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=3072,
    vocab_size=151936,
    max_position_embeddings=4096 * 32,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000,
    hidden_act="silu",
    tie_word_embeddings=True,
)

ATOL = 1e-5
RTOL = 1e-4


def full_compute(model: Qwen3, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """One-shot forward: returns logits for ALL positions (last position used for comparison)."""
    hidden = model(input_ids, positions)
    logits = model.compute_logits(hidden)
    return logits


def cached_prefill_then_decode(model: Qwen3, seq_len: int, device: torch.device) -> torch.Tensor:
    """Prefill + single-token decode using KV cache. Returns last-token logits."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, QWEN3_0_6B["vocab_size"], (seq_len,), device=device)

    num_layers = QWEN3_0_6B["num_hidden_layers"]
    num_kv_heads = QWEN3_0_6B["num_key_value_heads"]
    # Get actual head_dim from the model (Qwen3Config has head_dim=128, not 64)
    head_dim = model.model.layers[0].attn.head_dim
    # Use a practical max_seq_len for the test (avoid allocating 131K positions)
    max_seq_len = max(seq_len, 1024)

    kv_cache = KVCache(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    if seq_len == 1:
        # Single token: just prefill (no separate decode needed)
        positions = torch.arange(1, device=device)
        hidden = model(input_ids[:1], positions, kv_cache=kv_cache, is_prefill=True)
        logits = model.compute_logits(hidden)
        return logits[-1:]  # shape [1, vocab_size]

    # Prefill: tokens 0..seq_len-2
    prefill_ids = input_ids[:seq_len - 1]
    prefill_positions = torch.arange(seq_len - 1, device=device)
    with torch.no_grad():
        _ = model(prefill_ids, prefill_positions, kv_cache=kv_cache, is_prefill=True)

    # Decode: single token at position seq_len-1
    decode_id = input_ids[seq_len - 1:seq_len]  # shape [1]
    decode_positions = torch.tensor([seq_len - 1], device=device)
    with torch.no_grad():
        hidden = model(decode_id, decode_positions, kv_cache=kv_cache, is_prefill=False)
        logits = model.compute_logits(hidden)
    return logits  # shape [1, vocab_size]


def run_compare(seq_lens: list[int], device: torch.device) -> list[dict]:
    """Compare full compute vs cached prefill+decode for each seq_len."""
    cfg = Qwen3Config(**QWEN3_0_6B)
    model = Qwen3(cfg).to(device).eval()

    results = []
    for sl in seq_lens:
        model.zero_grad()
        torch.manual_seed(42)
        input_ids = torch.randint(0, QWEN3_0_6B["vocab_size"], (sl,), device=device)
        positions = torch.arange(sl, device=device)

        with torch.no_grad():
            logits_full = full_compute(model, input_ids, positions)
            logits_last_full = logits_full[-1:]  # last token

        logits_last_cached = cached_prefill_then_decode(model, sl, device)

        diff = (logits_last_full - logits_last_cached).abs()
        max_abs_diff = diff.max().item()
        cos_sim = F.cosine_similarity(
            logits_last_full.float().flatten(),
            logits_last_cached.float().flatten(),
            dim=0,
        ).item()

        passed = torch.allclose(logits_last_full, logits_last_cached, atol=ATOL, rtol=RTOL)
        # Also report per-element top diffs for diagnostics
        top_diffs = []
        if not passed:
            flat_diff = diff.flatten()
            topk_vals, topk_idx = flat_diff.topk(min(10, flat_diff.numel()))
            for v, i in zip(topk_vals.tolist(), topk_idx.tolist()):
                top_diffs.append({"idx": i, "abs_diff": round(v, 8),
                                  "full_val": round(logits_last_full.flatten()[i].item(), 8),
                                  "cached_val": round(logits_last_cached.flatten()[i].item(), 8)})

        result = {
            "seq_len": sl,
            "max_abs_diff": round(max_abs_diff, 8),
            "cosine_sim": round(cos_sim, 10),
            "passed": passed,
            "threshold": {"atol": ATOL, "rtol": RTOL},
        }
        if top_diffs:
            result["top_diffs"] = top_diffs

        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  seq_len={sl:>4}: max|diff|={max_abs_diff:.2e}  cos_sim={cos_sim:.10f}  {status}")
        if top_diffs:
            print(f"           top element diffs: {top_diffs[:3]}")

    return results


def run_error_cases(device: torch.device) -> list[dict]:
    """Test edge-case error handling in KVCache."""
    num_layers = 4
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 64

    results = []
    cache = KVCache(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    # Test 1: out-of-bounds write (position too large)
    k = torch.randn(1, num_kv_heads, head_dim, device=device)
    v = torch.randn(1, num_kv_heads, head_dim, device=device)
    try:
        cache.write(0, max_seq_len, k, v)
        results.append({"test": "write_OOB_position", "expected": "IndexError", "got": "no_error", "pass": False})
    except IndexError:
        results.append({"test": "write_OOB_position", "expected": "IndexError", "got": "IndexError", "pass": True})
    except Exception as e:
        results.append({"test": "write_OOB_position", "expected": "IndexError", "got": type(e).__name__, "pass": False})

    # Test 2: out-of-bounds layer_idx
    try:
        cache.read(num_layers, 1)
        results.append({"test": "read_OOB_layer", "expected": "IndexError", "got": "no_error", "pass": False})
    except IndexError:
        results.append({"test": "read_OOB_layer", "expected": "IndexError", "got": "IndexError", "pass": True})
    except Exception as e:
        results.append({"test": "read_OOB_layer", "expected": "IndexError", "got": type(e).__name__, "pass": False})

    # Test 3: write with wrong K shape
    k_bad = torch.randn(1, num_kv_heads + 1, head_dim, device=device)
    try:
        cache.write(0, 0, k_bad, v)
        results.append({"test": "write_bad_shape", "expected": "ValueError/IndexError", "got": "no_error", "pass": False})
    except (ValueError, IndexError):
        results.append({"test": "write_bad_shape", "expected": "ValueError/IndexError", "got": "Error", "pass": True})
    except Exception as e:
        results.append({"test": "write_bad_shape", "expected": "ValueError/IndexError", "got": type(e).__name__, "pass": False})

    # Test 4: read with negative end_pos
    try:
        cache.read(0, -1)
        results.append({"test": "read_negative_end", "expected": "IndexError", "got": "no_error", "pass": False})
    except IndexError:
        results.append({"test": "read_negative_end", "expected": "IndexError", "got": "IndexError", "pass": True})
    except Exception as e:
        results.append({"test": "read_negative_end", "expected": "IndexError", "got": type(e).__name__, "pass": False})

    # Test 5: read end_pos beyond max_seq_len
    try:
        cache.read(0, max_seq_len + 1)
        results.append({"test": "read_OOB_end", "expected": "IndexError", "got": "no_error", "pass": False})
    except IndexError:
        results.append({"test": "read_OOB_end", "expected": "IndexError", "got": "IndexError", "pass": True})
    except Exception as e:
        results.append({"test": "read_OOB_end", "expected": "IndexError", "got": type(e).__name__, "pass": False})

    # Test 6: write negative position
    try:
        cache.write(0, -1, k, v)
        results.append({"test": "write_negative_pos", "expected": "IndexError", "got": "no_error", "pass": False})
    except IndexError:
        results.append({"test": "write_negative_pos", "expected": "IndexError", "got": "IndexError", "pass": True})
    except Exception as e:
        results.append({"test": "write_negative_pos", "expected": "IndexError", "got": type(e).__name__, "pass": False})

    # Test 7: empty read (end_pos=0)
    k_empty, v_empty = cache.read(0, 0)
    results.append({
        "test": "read_empty",
        "expected": "shape[0]==0",
        "got": f"k_shape={list(k_empty.shape)} v_shape={list(v_empty.shape)}",
        "pass": k_empty.shape[0] == 0 and v_empty.shape[0] == 0,
    })

    # Test 8: reset clears cache
    k_write = torch.randn(1, num_kv_heads, head_dim, device=device)
    v_write = torch.randn(1, num_kv_heads, head_dim, device=device)
    cache.write(0, 0, k_write, v_write)
    cache.reset()
    k_read, v_read = cache.read(0, 1)
    is_zero = (k_read.abs().sum().item() == 0.0) and (v_read.abs().sum().item() == 0.0)
    results.append({
        "test": "reset_clears",
        "expected": "all_zero",
        "got": f"k_sum={k_read.abs().sum().item():.2e} v_sum={v_read.abs().sum().item():.2e}",
        "pass": is_zero,
    })

    # Test 9: write at multiple positions (batch write)
    cache.reset()
    k_multi = torch.randn(3, num_kv_heads, head_dim, device=device)
    v_multi = torch.randn(3, num_kv_heads, head_dim, device=device)
    positions = torch.tensor([0, 2, 4], device=device)
    cache.write(0, positions, k_multi, v_multi)
    k_read = cache.read(0, 5)[0]
    match_0 = torch.allclose(k_read[0], k_multi[0], atol=1e-7)
    match_2 = torch.allclose(k_read[2], k_multi[1], atol=1e-7)
    match_4 = torch.allclose(k_read[4], k_multi[2], atol=1e-7)
    results.append({
        "test": "batch_write_read",
        "expected": "positions_0_2_4_match",
        "got": f"pos0={match_0} pos2={match_2} pos4={match_4}",
        "pass": match_0 and match_2 and match_4,
    })

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['test']}: {status}  (expected={r['expected']}, got={r['got']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="KV Cache comparison audit")
    parser.add_argument("--mode", choices=["smoke", "compare", "error_cases"], default="smoke")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[8])
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override num_hidden_layers (for fast smoke testing)")
    args = parser.parse_args()

    # Initialize distributed (required by Qwen3Attn.__init__)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    if args.num_layers is not None:
        QWEN3_0_6B["num_hidden_layers"] = args.num_layers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {QWEN3_0_6B['num_hidden_layers']} layers, "
          f"hidden={QWEN3_0_6B['hidden_size']}, "
          f"heads={QWEN3_0_6B['num_attention_heads']}, "
          f"kv_heads={QWEN3_0_6B['num_key_value_heads']}")
    print(f"Thresholds: atol={ATOL}, rtol={RTOL}")

    report = {"threshold": {"atol": ATOL, "rtol": RTOL}, "mode": args.mode}

    if args.mode in ("smoke", "compare"):
        seq_lens = args.seq_lens if args.mode == "compare" else [8]
        print(f"\n{' Compare Mode ':-^50}")
        print(f"seq_lens: {seq_lens}")
        results = run_compare(seq_lens, device)
        report["compare_results"] = results

        all_passed = all(r["passed"] for r in results)
        print(f"\n{' Summary ':-^50}")
        print(f"  Total: {len(results)} tests, All passed: {all_passed}")
        if not all_passed:
            failed = [r["seq_len"] for r in results if not r["passed"]]
            print(f"  FAILED seq_lens: {failed}")

        # Write JSON
        json_path = RESULTS_DIR / "kv_cache_compare.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  JSON report: {json_path}")

        # Write TXT summary
        txt_path = RESULTS_DIR / "kv_cache_compare.txt"
        with open(txt_path, "w") as f:
            f.write("KV Cache Compare Results\n")
            f.write("========================\n\n")
            f.write(f"Model: {QWEN3_0_6B['num_hidden_layers']} layers\n")
            f.write(f"Threshold: atol={ATOL}, rtol={RTOL}\n\n")
            for r in results:
                status = "PASS" if r["passed"] else "FAIL"
                f.write(f"seq_len={r['seq_len']:>4}: max|diff|={r['max_abs_diff']:.2e}  "
                        f"cos_sim={r['cosine_sim']:.10f}  {status}\n")
            f.write(f"\nAll passed: {all_passed}\n")
        print(f"  TXT  report: {txt_path}")

        if not all_passed:
            sys.exit(1)

    elif args.mode == "error_cases":
        print(f"\n{' Error Cases Mode ':-^50}")
        results = run_error_cases(device)
        report["error_results"] = results
        all_passed = all(r["pass"] for r in results)
        print(f"\n{' Summary ':-^50}")
        print(f"  Total: {len(results)} tests, All passed: {all_passed}")

        json_path = RESULTS_DIR / "kv_cache_error_cases.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  JSON report: {json_path}")

        if not all_passed:
            sys.exit(1)

    print(f"\n{' ALL DONE ':-^50}")


if __name__ == "__main__":
    main()
