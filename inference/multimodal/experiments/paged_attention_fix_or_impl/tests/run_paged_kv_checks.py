#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = EXPERIMENT_DIR.parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from block_manager import BlockManager
from lifecycle import decode_append, free_request, prefill_allocate
from paged_kv_cache import PagedKVCache
from request_state import RequestState

RESULTS_DIR = EXPERIMENT_DIR / "results"
EVIDENCE_DIR = PROJECT_DIR / ".omo" / "evidence"


def test_single_request_short_prompt(block_size: int) -> dict:
    block_mgr = BlockManager(total_blocks=8, block_size=block_size)
    request = RequestState("short", max_tokens=block_size * 2, block_size=block_size)
    prefill_allocate(block_mgr, request, prompt_len=max(1, block_size // 2))
    passed = request.block_table.num_blocks == 1 and request.seq_len == max(1, block_size // 2)
    return _result("single_request_short_prompt", block_size, passed, block_mgr.stats())


def test_single_request_cross_block(block_size: int) -> dict:
    block_mgr = BlockManager(total_blocks=8, block_size=block_size)
    request = RequestState("cross", max_tokens=block_size * 4, block_size=block_size)
    prompt_len = block_size * 2
    prefill_allocate(block_mgr, request, prompt_len=prompt_len)
    passed = request.block_table.num_blocks == 2 and block_mgr.fragmentation_ratio() <= 0.10
    return _result("single_request_cross_block", block_size, passed, block_mgr.stats())


def test_multi_request_different_lengths(block_size: int) -> dict:
    block_mgr = BlockManager(total_blocks=16, block_size=block_size)
    req_a = RequestState("a", max_tokens=block_size * 3, block_size=block_size)
    req_b = RequestState("b", max_tokens=block_size * 3, block_size=block_size)
    prefill_allocate(block_mgr, req_a, prompt_len=block_size)
    prefill_allocate(block_mgr, req_b, prompt_len=block_size * 2)
    ids_unique = set(req_a.block_table.physical_block_ids).isdisjoint(req_b.block_table.physical_block_ids)
    passed = ids_unique and block_mgr.stats()["used_tokens"] == block_size * 3
    return _result("multi_request_different_lengths", block_size, passed, block_mgr.stats())


def test_decode_append_token(block_size: int) -> dict:
    block_mgr = BlockManager(total_blocks=8, block_size=block_size)
    request = RequestState("decode", max_tokens=block_size + 2, block_size=block_size)
    prefill_allocate(block_mgr, request, prompt_len=block_size)
    new_block_id = decode_append(block_mgr, request)
    passed = new_block_id is not None and request.seq_len == block_size + 1
    return _result("decode_append_token", block_size, passed, block_mgr.stats())


def test_free_request_releases_blocks(block_size: int) -> dict:
    block_mgr = BlockManager(total_blocks=8, block_size=block_size)
    request = RequestState("free", max_tokens=block_size * 2, block_size=block_size)
    prefill_allocate(block_mgr, request, prompt_len=block_size + 1)
    free_request(block_mgr, request)
    stats = block_mgr.stats()
    passed = stats["allocated_blocks"] == 0 and stats["free_blocks"] == 8 and request.seq_len == 0
    return _result("free_request_releases_blocks", block_size, passed, stats)


def test_block_reuse_and_gather(block_size: int) -> dict:
    block_mgr = BlockManager(total_blocks=4, block_size=block_size)
    first = RequestState("first", max_tokens=block_size, block_size=block_size)
    prefill_allocate(block_mgr, first, prompt_len=block_size)
    first_block = first.block_table.physical_block_ids[0]
    free_request(block_mgr, first)

    second = RequestState("second", max_tokens=block_size, block_size=block_size)
    prefill_allocate(block_mgr, second, prompt_len=block_size)
    cache = PagedKVCache(4, 1, 2, 4, block_size, dtype=torch.float32)
    data_k = torch.arange(block_size * 2 * 4, dtype=torch.float32).view(block_size, 2, 4)
    data_v = data_k + 1000
    cache.write_kv(second.block_table.physical_block_ids[0], 0, 0, 0, data_k)
    cache.write_kv(second.block_table.physical_block_ids[0], 0, 1, 0, data_v)
    gathered_k, gathered_v = cache.gather_kv_for_attention(second.block_table.physical_block_ids, 0, 0, block_size)
    passed = (
        second.block_table.physical_block_ids[0] == first_block
        and torch.equal(gathered_k, data_k)
        and torch.equal(gathered_v, data_v)
    )
    return _result("block_reuse_and_gather", block_size, passed, block_mgr.stats())


def _result(name: str, block_size: int, passed: bool, stats: dict[str, int | float]) -> dict:
    return {"test": name, "block_size": block_size, "passed": passed, "stats": stats}


def run_for_block_size(block_size: int) -> list[dict]:
    checks = [
        test_single_request_short_prompt,
        test_single_request_cross_block,
        test_multi_request_different_lengths,
        test_decode_append_token,
        test_free_request_releases_blocks,
        test_block_reuse_and_gather,
    ]
    return [check(block_size) for check in checks]


def write_reports(results: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "paged_kv_checks.json"
    txt_path = RESULTS_DIR / "paged_kv_checks.txt"
    evidence_path = EVIDENCE_DIR / "task-6-paged-tests.txt"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = ["Task 6 paged KV checks", "======================", ""]
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"block_size={item['block_size']:>2} {item['test']}: {status} stats={item['stats']}")
    lines.append(f"\nAll passed: {all(item['passed'] for item in results)}")
    text = "\n".join(lines) + "\n"
    txt_path.write_text(text, encoding="utf-8")
    evidence_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paged KV correctness checks")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    results = []
    for block_size in args.block_sizes:
        results.extend(run_for_block_size(block_size))
    write_reports(results)
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"block_size={item['block_size']:>2} {item['test']}: {status} stats={item['stats']}")
    if not all(item["passed"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
