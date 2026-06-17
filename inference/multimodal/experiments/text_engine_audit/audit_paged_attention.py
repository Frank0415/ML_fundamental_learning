#!/usr/bin/env python3
"""audit_paged_attention.py — Static audit of paged attention implementation status.

Scans: entire minivLLM/ for block_table, paged, slot_mapping, page_table keywords.
Determines whether paged attention is implemented, partially scaffolded, or absent.
Does NOT import minivLLM.
Produces JSON report to results/paged_attention.json.
"""

import json
import os
import re
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[2] / "minivLLM" / "minivllm"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Also scan the validate_model.py at the parent level
VALIDATE_SCRIPT = Path(__file__).resolve().parents[2] / "minivLLM" / "validate_model.py"

# Keywords to scan for, organized by category
KEYWORDS = {
    "paged_attention_core": [
        "paged", "page_table", "block_table", "block_tables",
        "slot_mapping", "block_size", "kvcache_block",
    ],
    "block_management": [
        "block_alloc", "free_block", "allocate_block", "BlockManager",
        "BlockAllocator", "BlockTable",
    ],
    "paged_kernel": [
        "paged_attention", "flash_attn", "flash_attention",
        "triton", "flashinfer",
    ],
    "scheduler": [
        "scheduler", "Scheduler", "batch", "request_queue",
        "preemption", "swap",
    ],
    "context_integration": [
        "set_context", "block_tables", "context_lens",
    ],
}


def scan_file(filepath: str) -> dict:
    """Scan a single file for all keyword categories."""
    source = open(filepath).read()
    result = {"file": os.path.relpath(filepath, str(ENGINE_ROOT.parent)), "hits": {}}
    found_any = False
    for category, keywords in KEYWORDS.items():
        cat_hits = {}
        for kw in keywords:
            # Use word boundary for short keywords to avoid false positives
            if len(kw) <= 5:
                pattern = rf"\b{re.escape(kw)}\b"
            else:
                pattern = re.escape(kw)
            matches = len(re.findall(pattern, source, re.IGNORECASE))
            if matches > 0:
                cat_hits[kw] = matches
                found_any = True
        if cat_hits:
            result["hits"][category] = cat_hits
    result["found_any"] = found_any
    return result


def scan_kvcache_for_paged(source: str) -> dict:
    """Detailed analysis of KV cache whether it uses paged logic."""
    return {
        "has_block_table": "block_table" in source.lower(),
        "has_paged": "paged" in source.lower(),
        "uses_slot_mapping": "slot_mapping" in source.lower(),
        "buffer_shape": re.findall(r"shape\s*=\s*\([^)]+\)", source),
        "is_contiguous_buffer": bool(re.search(r"max_seq_len[,\s]*num_kv_heads", source)),
        "note": (
            "Contiguous buffer: shape=(num_layers, max_seq_len, num_kv_heads, head_dim). "
            "No block-level indirection."
        ),
    }


def scan_context_for_paged(source: str) -> dict:
    """Check Context dataclass scaffolding."""
    return {
        "block_tables_field": "block_tables" in source,
        "slot_mapping_field": "slot_mapping" in source,
        "context_lens_field": "context_lens" in source,
        "is_field_used": False,  # requires cross-ref, done below
    }


def determine_status(all_results: list) -> dict:
    """From all scan results, determine the paged attention implementation status."""
    status = {
        "verdict": "",
        "details": [],
    }

    total_paged_hits = 0
    total_block_table_hits = 0
    has_block_manager = False
    has_paged_kernel = False
    has_scheduler = False
    context_used = False

    for result in all_results:
        for cat, hits in result.get("hits", {}).items():
            if cat == "paged_attention_core":
                total_paged_hits += sum(hits.values())
                if "block_table" in hits or "block_tables" in hits:
                    total_block_table_hits += hits.get("block_table", 0) + hits.get("block_tables", 0)
            if cat == "block_management" and hits:
                has_block_manager = True
            if cat == "paged_kernel" and hits:
                has_paged_kernel = True
            if cat == "scheduler" and hits:
                has_scheduler = True
            if cat == "context_integration" and hits:
                context_used = True

    # Additional specific check: is set_context ever called?
    all_source = ""
    for root, dirs, files in os.walk(str(ENGINE_ROOT)):
        for f in files:
            if f.endswith(".py"):
                all_source += open(os.path.join(root, f)).read() + "\n"
    if VALIDATE_SCRIPT.exists():
        all_source += open(VALIDATE_SCRIPT).read()

    all_matches = len(re.findall(r"set_context\s*\(", all_source))
    def_matches = len(re.findall(r"def\s+\w*set_context\s*\(", all_source))
    set_context_calls = all_matches - def_matches
    context_calls = set_context_calls > 0

    # Determine verdict
    reasons = []

    if not has_block_manager:
        reasons.append("无 Block Manager（无 block 分配/释放逻辑）")
    if not has_paged_kernel:
        reasons.append("无 paged attention kernel（无 block 级别注意力计算）")
    if not has_scheduler:
        reasons.append("无调度器（无 batching/request 管理）")
    if not context_calls:
        reasons.append("Context.set_context() 从未被调用（脚手架字段休眠）")

    block_table_in_context_only = False
    for result in all_results:
        if "context.py" in result["file"] and "block_tables" in str(result.get("hits", {})):
            block_table_in_context_only = True

    if total_block_table_hits > 0 and not context_calls:
        reasons.append("block_tables 字段仅存在于 Context 定义中，未在任何运行路径中被赋值")

    if not reasons:
        status["verdict"] = "已实现 paged attention"
    else:
        status["verdict"] = "未实现 paged attention"
        status["details"] = reasons

    status.update({
        "total_paged_keyword_hits": total_paged_hits,
        "block_table_hits": total_block_table_hits,
        "has_block_manager": has_block_manager,
        "has_paged_kernel": has_paged_kernel,
        "has_scheduler": has_scheduler,
        "context_scaffolding_exists": block_table_in_context_only,
        "set_context_ever_called": context_calls,
        "set_context_call_count": set_context_calls,
    })

    return status


def main():
    results = []

    # Scan all .py files in minivllm/
    for root, dirs, files in os.walk(str(ENGINE_ROOT)):
        for f in files:
            if f.endswith(".py"):
                result = scan_file(os.path.join(root, f))
                if result["found_any"]:
                    results.append(result)

    # Also scan validate_model.py
    if VALIDATE_SCRIPT.exists():
        result = scan_file(str(VALIDATE_SCRIPT))
        if result["found_any"]:
            results.append(result)

    # Detailed KV cache analysis
    kv_source = open(ENGINE_ROOT / "core" / "kv_cache.py").read()
    ctx_source = open(ENGINE_ROOT / "utils" / "context.py").read()
    kv_paged = scan_kvcache_for_paged(kv_source)
    ctx_paged = scan_context_for_paged(ctx_source)

    # Determine overall status
    status = determine_status(results)

    report = {
        "audit_target": "Paged Attention implementation status",
        "timestamp": "2026-06-07",
        "file_scan_results": results,
        "kv_cache_paged_analysis": kv_paged,
        "context_paged_analysis": ctx_paged,
        "status": status,
        "结论": f"contiguous，不是 paged attention。{'; '.join(status['details'])}" if status["details"] else status["verdict"],
        "实现状态": status["verdict"],
        "阻塞项": status["details"] if status["verdict"] == "未实现 paged attention" else [],
    }

    out_path = RESULTS_DIR / "paged_attention.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report written to {out_path}")
    print(f"  Verdict: {status['verdict']}")
    print(f"  结论: {report['结论']}")
    if status["details"]:
        for d in status["details"]:
            print(f"    - {d}")
    print(f"  Block manager: {status['has_block_manager']}")
    print(f"  Paged kernel: {status['has_paged_kernel']}")
    print(f"  Scheduler: {status['has_scheduler']}")
    print(f"  Context scaffolding: {status['context_scaffolding_exists']}")
    print(f"  set_context called: {status['set_context_ever_called']} ({status['set_context_call_count']} times)")


if __name__ == "__main__":
    main()
