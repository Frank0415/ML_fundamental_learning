#!/usr/bin/env python3
"""
Case 1: same_text_same_image — 相同文本 + 相同图片, 应命中。

预期:
- 策略 A: true_hit (text 相同)
- 策略 B: true_hit (text + image 相同)
- 策略 C: true_hit (全量元数据相同)
"""
from __future__ import annotations

import json
import os
import sys

# 确保可 import 上级模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_kv_cache_management.mm_cache_simulator import (
    CacheSimulator, MultimodalRequest, _read_sample_bytes, _make_synthetic_bytes,
)

CASE_NAME = "same_text_same_image"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_case_data():
    """构造同一个请求的两次副本 (完全一致)。"""
    img = _read_sample_bytes()  # 真实 demo.jpg
    text = [101, 102, 103, 104, 105]  # 模拟 token IDs

    r1 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img],
        original_sizes=[(512, 512)],
        resized_sizes=[(336, 336)],
        patch_grids=[(12, 12)],
        num_visual_tokens_list=[256],
    )
    r2 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img],
        original_sizes=[(512, 512)],
        resized_sizes=[(336, 336)],
        patch_grids=[(12, 12)],
        num_visual_tokens_list=[256],
    )
    return r1, r2


def run():
    r1, r2 = build_case_data()
    results = {}
    all_stats = {}

    for strat in ("A", "B", "C"):
        sim = CacheSimulator(strategy=strat)
        sim.insert(r1, prefill_tokens=len(r1.text_token_ids) + 256, kv_blocks_used=22)
        verdict = sim.query(r2)
        stats = sim.stats()
        results[strat] = verdict
        all_stats[strat] = stats

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  Case: {CASE_NAME}")
    print(f"{'='*60}")
    for strat in ("A", "B", "C"):
        v = results[strat]
        s = all_stats[strat]
        print(f"  策略 {strat}: verdict={v['verdict']:12s}  "
              f"true_hits={s['true_hits']}  false_hits={s['false_hits']}  "
              f"safe_misses={s['safe_misses']}")

    # 写 JSON
    out = {
        "case": CASE_NAME,
        "description": "same text, same image — expected: all strategies should hit",
        "per_strategy": {s: {"verdict": results[s]["verdict"], **all_stats[s]}
                         for s in ("A", "B", "C")},
    }
    json_path = os.path.join(RESULTS_DIR, f"{CASE_NAME}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON → {json_path}")

    # 生成 HTML
    html = _build_html(CASE_NAME, out)
    html_path = os.path.join(RESULTS_DIR, f"{CASE_NAME}_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML → {html_path}")

    return out


def _build_html(case: str, data: dict) -> str:
    ps = data["per_strategy"]
    rows = ""
    for s in ("A", "B", "C"):
        st = ps[s]
        v = st["verdict"]
        vc = "#4caf50" if v == "true_hit" else ("#f44336" if "false" in v else "#ff9800")
        rows += f"""
        <tr>
          <td><strong>策略 {s}</strong></td>
          <td style="color:{vc};font-weight:bold">{v}</td>
          <td>{st['true_hits']}</td>
          <td>{st['false_hits']}</td>
          <td>{st['safe_misses']}</td>
          <td>{st['prefill_tokens_saved']}</td>
          <td>{st['kv_blocks_reused']}</td>
          <td>{st['memory_saved_bytes']}</td>
        </tr>"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{case} — KV Cache Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#333;padding:32px}}
h1{{font-size:24px;margin-bottom:8px}}
.sub{{color:#666;margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08)}}
th{{background:#1a1a2e;color:#fff;padding:12px 16px;text-align:left;font-size:13px}}
td{{padding:10px 16px;border-bottom:1px solid #eee;font-size:14px}}
tr:last-child td{{border-bottom:none}}
.key-finding{{margin-top:24px;padding:16px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px}}
.key-finding h3{{margin-bottom:8px}}
.pass{{color:#4caf50;font-weight:bold}}
.fail{{color:#f44336;font-weight:bold}}
</style></head>
<body>
<h1>📊 {case}</h1>
<p class="sub">描述: {data['description']}</p>
<table><thead><tr>
  <th>策略</th><th>判定</th><th>true_hits</th><th>false_hits</th>
  <th>safe_misses</th><th>prefill_tokens_saved</th><th>kv_blocks_reused</th><th>memory_saved</th>
</tr></thead><tbody>{rows}</tbody></table>
<div class="key-finding">
  <h3>🔍 关键发现</h3>
  <p>当文本与图像完全相同时, 三种策略均正确命中。这是多模态 KV cache 的最理想场景——前缀 cache 安全复用, 无 false hit 风险。</p>
  <p>但请参阅 <strong>same_text_different_image</strong> 场景, text-only prefix cache 在多模态下存在严重 false hit 风险。</p>
</div>
</body></html>"""


if __name__ == "__main__":
    run()
