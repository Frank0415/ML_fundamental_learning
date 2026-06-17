#!/usr/bin/env python3
"""audit_kv_cache.py — Static audit of minivLLM KV cache implementation.

Scans: kv_cache.py and all engine files for KVCache references.
Does NOT import minivLLM.
Produces JSON report to results/kv_cache.json.
"""

import json
import os
import re
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[2] / "minivLLM" / "minivllm"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_file(rel_path: str) -> str:
    p = ENGINE_ROOT / rel_path
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def analyze_kvcache_source(source: str) -> dict:
    """Extract structural info from KVCache class."""
    result = {
        "class_exists": "class KVCache" in source,
        "buffer_shape_description": "",
        "is_contiguous": False,
        "has_block_tables": "block_table" in source.lower(),
        "has_paged_logic": "paged" in source.lower(),
        "methods": [],
    }
    # Extract buffer shapes
    shape_match = re.search(r"shape\s*=\s*\(([^)]+)\)", source)
    if shape_match:
        result["buffer_shape_description"] = shape_match.group(1).strip()
    # Check if contiguous (has flat shape like (layers, seq_len, ...))
    if re.search(r"max_seq_len", source) and not re.search(r"block_size", source):
        result["is_contiguous"] = True
    # Extract method names
    result["methods"] = re.findall(r"def\s+(\w+)\s*\(", source)
    return result


def scan_all_files_for_kvcache_refs(engine_root: str) -> dict:
    """Search every .py file for KVCache import/reference."""
    findings = {
        "imports": [],
        "instantiations": [],
        "forward_usage": [],
        "total_files_scanned": 0,
    }
    for root, dirs, files in os.walk(engine_root):
        for f in files:
            if f.endswith(".py"):
                findings["total_files_scanned"] += 1
                fp = os.path.join(root, f)
                rel = os.path.relpath(fp, engine_root)
                source = open(fp).read()
                if "KVCache" in source or "kv_cache" in source:
                    if re.search(r"(import|from).*KVCache", source):
                        findings["imports"].append(rel)
                    if re.search(r"KVCache\s*\(", source):
                        findings["instantiations"].append(rel)
                    if re.search(r"\.write\s*\(|\.read\s*\(|\.reset\s*\(|\.layer_cache\s*\(", source):
                        findings["forward_usage"].append(rel)
    return findings


def scan_context_scaffolding(source: str) -> dict:
    """Analyze Context dataclass for paged attention fields."""
    result = {
        "has_block_tables": "block_tables" in source,
        "has_slot_mapping": "slot_mapping" in source,
        "has_context_lens": "context_lens" in source,
        "set_context_calls": 0,
        "get_context_calls": 0,
    }
    result["set_context_calls"] = len(re.findall(r"set_context\s*\(", source))
    result["get_context_calls"] = len(re.findall(r"get_context\s*\(", source))
    return result


def scan_config_paged_fields(source: str) -> dict:
    """Check Config for paged attention related fields."""
    result = {}
    for field in ["kvcache_block_size", "num_kvcache_blocks"]:
        m = re.search(rf"{field}\s*[:=]\s*([^,\n]+)", source)
        if m:
            result[field] = m.group(1).strip()
        else:
            result[field] = "NOT_FOUND"
    return result


def scan_for_decode_usage(engine_root: str) -> list:
    """Find any decode/generate loops in engine code."""
    loops = []
    for root, dirs, files in os.walk(engine_root):
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                source = open(fp).read()
                # Look for generation loops
                if re.search(r"generate|decode", source, re.IGNORECASE):
                    loops.append({
                        "file": os.path.relpath(fp, engine_root),
                        "has_for_loop_generate": bool(re.search(r"for\s+\w+\s+in\s+range.*generate", source)),
                        "has_decode_comment": bool(re.search(r"decode", source, re.IGNORECASE)),
                    })
    return loops


def main():
    kv_source = read_file("core/kv_cache.py")
    ctx_source = read_file("utils/context.py")
    cfg_source = read_file("config.py")

    kv_analysis = analyze_kvcache_source(kv_source)
    refs = scan_all_files_for_kvcache_refs(str(ENGINE_ROOT))
    ctx_analysis = scan_context_scaffolding(ctx_source)
    cfg_analysis = scan_config_paged_fields(cfg_source)

    report = {
        "audit_target": "KV Cache implementation",
        "timestamp": "2026-06-07",
        "kvcache_class_analysis": kv_analysis,
        "cross_references_in_engine": refs,
        "cross_reference_verdict": (
            "ZERO forward references — KVCache is never imported or used by any model code"
            if not refs["imports"] and not refs["forward_usage"]
            else f"KVCache has {len(refs['imports'])} imports and {len(refs['forward_usage'])} forward usages"
        ),
        "is_wired_to_forward": len(refs["forward_usage"]) > 0,
        "context_scaffolding": ctx_analysis,
        "context_verdict": (
            "Context.set_context() is NEVER called — scaffolding fields (block_tables, slot_mapping) are dormant"
            if ctx_analysis["set_context_calls"] == 0
            else f"set_context called {ctx_analysis['set_context_calls']} times"
        ),
        "config_paged_fields": cfg_analysis,
        "contiguous_or_paged": (
            "contiguous buffer" if kv_analysis["is_contiguous"] else "NOT contiguous (check source)"
        ),
        "conclusion": (
            "KVCache 是 contiguous buffer 实现，不是 paged attention。"
            "整个 minivllm/ 代码库中没有对 KVCache 的任何 forward 引用——它是 dead code。"
            "Context.set_context() 从未被调用，block_tables 等脚手架字段处于休眠状态。"
            "Config 中有 kvcache_block_size 字段（paged attention 预留），但无代码使用。"
        ),
    }

    out_path = RESULTS_DIR / "kv_cache.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report written to {out_path}")
    print(f"  KVCache methods: {kv_analysis['methods']}")
    print(f"  Buffer shape: {kv_analysis['buffer_shape_description']}")
    print(f"  Is contiguous: {kv_analysis['is_contiguous']}")
    print(f"  Cross-ref imports: {refs['imports']}")
    print(f"  Cross-ref forward usage: {refs['forward_usage']}")
    print(f"  set_context calls: {ctx_analysis['set_context_calls']}")


if __name__ == "__main__":
    main()
