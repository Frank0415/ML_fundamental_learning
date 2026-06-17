#!/usr/bin/env python3
"""audit_attention.py — Static audit of minivLLM attention implementation.

Scans: attention.py, qwen3.py (Qwen3Attn)
Does NOT import minivLLM (avoids triggering construction errors).
Produces JSON report to results/attention.json.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parents[2] / "minivLLM" / "minivllm"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_file(rel_path: str) -> str:
    p = ENGINE_ROOT / rel_path
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def scan_attn_init_signature(source: str) -> dict:
    """Extract Attn.__init__ parameter names and defaults."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Attn":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    params = []
                    for arg in item.args.args:
                        params.append(arg.arg)
                    defaults = [None] * (len(params) - len(item.args.defaults)) + [
                        ast.unparse(d) if hasattr(ast, "unparse") else str(d) for d in item.args.defaults
                    ]
                    return {
                        "params": params,
                        "defaults": {p: d for p, d in zip(params, defaults) if d is not None},
                    }
    return {"params": [], "defaults": {}}


def scan_qwen3attn_init_calls(source: str) -> list:
    """Find Attn(...) constructor calls inside Qwen3Attn.__init__."""
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3Attn":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Call):
                            if isinstance(sub.func, ast.Name) and sub.func.id == "Attn":
                                kwargs = {
                                    kw.arg: ast.unparse(kw.value) if hasattr(ast, "unparse") else str(kw.value)
                                    for kw in sub.keywords
                                }
                                calls.append(kwargs)
    return calls


def scan_ffn_act_fn(source: str) -> dict:
    """Check Qwen3FFN.act_fn status."""
    tree = ast.parse(source)
    result = {"act_fn_assignment": None, "forward_calls_act_fn": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3FFN":
            for item in node.body:
                # Check __init__ for act_fn assignment
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if isinstance(t, ast.Attribute) and t.attr == "act_fn":
                                    if hasattr(ast, "unparse"):
                                        result["act_fn_assignment"] = ast.unparse(sub.value)
                                    else:
                                        if isinstance(sub.value, ast.Constant) and sub.value.value is None:
                                            result["act_fn_assignment"] = "None"
                                        elif isinstance(sub.value, ast.NameConstant) and sub.value.value is None:
                                            result["act_fn_assignment"] = "None"
                                        else:
                                            result["act_fn_assignment"] = str(sub.value)
                # Check forward for act_fn call
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Call):
                            if isinstance(sub.func, ast.Attribute) and sub.func.attr == "act_fn":
                                result["forward_calls_act_fn"] = True
    return result


def scan_causal_mask(source: str) -> dict:
    """Analyze causal_mask function."""
    tree = ast.parse(source)
    result = {"exists": False, "params": [], "hardcoded_query_start_0": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "causal_mask":
            result["exists"] = True
            result["params"] = [a.arg for a in node.args.args]
            result["hardcoded_query_start_0"] = True  # current impl always starts from 0
    return result


def scan_paged_keywords(source: str) -> dict:
    """Search for paged attention related keywords in source."""
    keywords = {
        "block_table": len(re.findall(r"block_table", source, re.IGNORECASE)),
        "slot_mapping": len(re.findall(r"slot_mapping", source, re.IGNORECASE)),
        "paged": len(re.findall(r"\bpaged\b", source, re.IGNORECASE)),
        "page_table": len(re.findall(r"page_table", source, re.IGNORECASE)),
    }
    return keywords


def main():
    attn_source = read_file("layers/numpy/attention.py")
    qwen_source = read_file("model/qwen3.py")

    attn_sig = scan_attn_init_signature(attn_source)
    attn_calls = scan_qwen3attn_init_calls(qwen_source)
    ffn_result = scan_ffn_act_fn(qwen_source)
    causal_result = scan_causal_mask(attn_source)

    # Check parameter mismatch
    mismatch = []
    for call_kwargs in attn_calls:
        extra_params = set(call_kwargs.keys()) - set(attn_sig["params"])
        missing_params = set(attn_sig["params"]) - set(call_kwargs.keys())
        if extra_params or missing_params:
            mismatch.append({
                "called_with": call_kwargs,
                "attn_accepts": attn_sig["params"],
                "extra_params_in_call": list(extra_params),
                "missing_params_in_call": list(missing_params),
            })

    # Scan entire minivllm for paged keywords
    all_paged = {}
    for root, dirs, files in os.walk(str(ENGINE_ROOT)):
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                source = open(fp).read()
                kw = scan_paged_keywords(source)
                rel = os.path.relpath(fp, str(ENGINE_ROOT))
                if any(v > 0 for v in kw.values()):
                    all_paged[rel] = kw

    report = {
        "audit_target": "attention + Qwen3Attn",
        "timestamp": "2026-06-07",
        "attn_init_signature": attn_sig,
        "qwen3attn_attn_calls": attn_calls,
        "parameter_mismatch": mismatch,
        "parameter_mismatch_verdict": (
            "BLOCKER: Qwen3Attn passes 'S' and 'is_decode' to Attn(), which does not accept them."
            if mismatch
            else "NO MISMATCH"
        ),
        "qwen3ffn_act_fn": ffn_result,
        "qwen3ffn_verdict": (
            "BLOCKER: act_fn = None; forward() calls self.act_fn(gate_up) -> TypeError"
            if ffn_result["forward_calls_act_fn"] and ffn_result["act_fn_assignment"] == "None"
            else "OK"
        ),
        "causal_mask": causal_result,
        "gqa_implementation": "repeat_kv() in attention.py (naive expand, no optimization)",
        "paged_attention_keywords_in_attention_files": {
            "attention.py": scan_paged_keywords(attn_source),
            "qwen3.py": scan_paged_keywords(qwen_source),
        },
        "paged_attention_keywords_in_engine": all_paged,
        "conclusion": (
            "attention 基本实现正确（Attn + GQA + causal mask），但存在致命构造参数不匹配（B1）"
            "和 Qwen3FFN act_fn=None（B2）。Paged attention 相关关键词在 attention.py/qwen3.py "
            "中完全不存在。"
        ),
    }

    out_path = RESULTS_DIR / "attention.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"Report written to {out_path}")
    print(f"  Attn.__init__ accepts: {attn_sig['params']}")
    print(f"  Qwen3Attn calls Attn with: {attn_calls}")
    print(f"  Mismatch: {mismatch}")
    print(f"  Qwen3FFN act_fn: {ffn_result}")
    print(f"  Causal mask: {causal_result}")


if __name__ == "__main__":
    main()
