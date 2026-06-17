#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen3Config

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
AUDIT_DIR = PROJECT_DIR / "experiments" / "text_engine_audit"
MINIVLLM_DIR = PROJECT_DIR / "minivLLM"
for path in (EXPERIMENT_DIR, AUDIT_DIR, MINIVLLM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_kv_cache_compare as task5
from block_manager import BlockManager
from lifecycle import decode_append, free_request, prefill_allocate
from paged_kv_cache import PagedKVCache
from request_state import RequestState
from minivllm.model.qwen3 import Qwen3

RESULTS_DIR = EXPERIMENT_DIR / "results"
EVIDENCE_DIR = PROJECT_DIR / ".omo" / "evidence"
ATOL = 1e-5
RTOL = 1e-4


class PagedKVCacheAdapter:
    def __init__(self, paged_cache: PagedKVCache, request: RequestState) -> None:
        self.paged_cache = paged_cache
        self.request = request

    def write(self, layer_idx: int, positions: int | torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        positions_tensor = self._normalize_positions(positions, k.device)
        if positions_tensor.numel() != k.shape[0] or k.shape != v.shape:
            raise ValueError("positions, k, and v token counts must match")
        for row, position in enumerate(positions_tensor.tolist()):
            block_id, offset = self.request.block_table.token_to_block_offset(position)
            self.paged_cache.write_kv(block_id, layer_idx, 0, offset, k[row])
            self.paged_cache.write_kv(block_id, layer_idx, 1, offset, v[row])

    def read(self, layer_idx: int, end_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.paged_cache.gather_kv_for_attention(
            self.request.block_table.physical_block_ids,
            layer_idx,
            0,
            end_pos,
        )

    @staticmethod
    def _normalize_positions(positions: int | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(positions, int):
            return torch.tensor([positions], device=device, dtype=torch.long)
        return positions.to(device=device, dtype=torch.long)


def paged_prefill_then_decode(model: Qwen3, seq_len: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, dict]:
    torch.manual_seed(42)
    input_ids = torch.randint(0, task5.QWEN3_0_6B["vocab_size"], (seq_len,), device=device)
    num_layers = task5.QWEN3_0_6B["num_hidden_layers"]
    num_kv_heads = task5.QWEN3_0_6B["num_key_value_heads"]
    head_dim = model.model.layers[0].attn.head_dim
    total_blocks = max(4, (seq_len + block_size - 1) // block_size + 2)
    block_mgr = BlockManager(total_blocks=total_blocks, block_size=block_size)
    request = RequestState("compare", max_tokens=seq_len, block_size=block_size)
    paged_cache = PagedKVCache(
        total_blocks=total_blocks,
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    adapter = PagedKVCacheAdapter(paged_cache, request)

    if seq_len == 1:
        prefill_allocate(block_mgr, request, prompt_len=1)
        positions = torch.arange(1, device=device)
        with torch.no_grad():
            hidden = model(input_ids[:1], positions, kv_cache=adapter, is_prefill=True)
            logits = model.compute_logits(hidden)[-1:]
        stats = block_mgr.stats()
        free_request(block_mgr, request)
        return logits, stats

    prefill_allocate(block_mgr, request, prompt_len=seq_len - 1)
    with torch.no_grad():
        prefill_ids = input_ids[: seq_len - 1]
        prefill_positions = torch.arange(seq_len - 1, device=device)
        _ = model(prefill_ids, prefill_positions, kv_cache=adapter, is_prefill=True)

    decode_append(block_mgr, request)
    decode_id = input_ids[seq_len - 1 : seq_len]
    decode_positions = torch.tensor([seq_len - 1], device=device)
    with torch.no_grad():
        hidden = model(decode_id, decode_positions, kv_cache=adapter, is_prefill=False)
        logits = model.compute_logits(hidden)
    stats = block_mgr.stats()
    free_request(block_mgr, request)
    return logits, stats


def run_compare(block_sizes: list[int], seq_lens: list[int], device: torch.device) -> list[dict]:
    cfg = Qwen3Config(**task5.QWEN3_0_6B)
    model = Qwen3(cfg).to(device).eval()
    results = []
    for block_size in block_sizes:
        for seq_len in seq_lens:
            logits_contiguous = task5.cached_prefill_then_decode(model, seq_len, device)
            logits_paged, stats = paged_prefill_then_decode(model, seq_len, block_size, device)
            diff = (logits_contiguous - logits_paged).abs()
            max_abs_diff = diff.max().item()
            cos_sim = F.cosine_similarity(logits_contiguous.float().flatten(), logits_paged.float().flatten(), dim=0).item()
            passed = torch.allclose(logits_contiguous, logits_paged, atol=ATOL, rtol=RTOL)
            assert passed, f"paged logits mismatch block_size={block_size} seq_len={seq_len} max_abs_diff={max_abs_diff}"
            result = {
                "block_size": block_size,
                "seq_len": seq_len,
                "max_abs_diff": round(max_abs_diff, 8),
                "cosine_sim": round(cos_sim, 10),
                "passed": passed,
                "stats": stats,
                "threshold": {"atol": ATOL, "rtol": RTOL},
            }
            results.append(result)
            print(f"block_size={block_size:>2} seq_len={seq_len:>4}: max|diff|={max_abs_diff:.2e} cos={cos_sim:.10f} PASS stats={stats}")
    return results


def write_reports(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "contiguous_vs_paged.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["Task 6 contiguous vs paged compare", "===================================", ""]
    for item in results:
        lines.append(
            f"block_size={item['block_size']:>2} seq_len={item['seq_len']:>4}: "
            f"max|diff|={item['max_abs_diff']:.2e} cos={item['cosine_sim']:.10f} PASS stats={item['stats']}"
        )
    lines.append(f"\nAll passed: {all(item['passed'] for item in results)}")
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / "contiguous_vs_paged.txt").write_text(text, encoding="utf-8")
    (EVIDENCE_DIR / "task-6-paged-compare.txt").write_text(text, encoding="utf-8")


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    results = run_compare(block_sizes=[16, 32], seq_lens=[8, 64, 512], device=device)
    write_reports(results)


if __name__ == "__main__":
    main()
