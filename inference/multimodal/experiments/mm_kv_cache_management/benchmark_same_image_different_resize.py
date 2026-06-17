#!/usr/bin/env python3
"""
Case 4: same_image_different_resize — 同一图片, 不同 resize 尺寸。

预期:
- 策略 A: true_hit (仅 text 匹配 — 危险! resize 不同但 key 相同)
- 策略 B: true_hit (text + image_bytes 相同 → key 相同 — 同样危险!)
- 策略 C: safe_miss (resize 不同 → key 不同 — 正确)

说明: resize 参数影响 visual token 的 patch_grid 和 num_visual_tokens,
进而影响 KV cache 的 shape 和语义对应。策略 A/B 无法感知此差异,
只有策略 C 通过 resized_sizes/patch_grids/num_visual_tokens 能正确区分。

注意: 本 case 的 "must miss" 要求是对安全性的期待——实际只有策略 C 能做到。
策略 A/B 的 hit 在语义上是错误的 (resize 不同 → visual tokens 布局不同),
但按照 false_hit 定义 (image_hash 不同), 此处归类为 true_hit + 语义警告。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_kv_cache_management.mm_cache_simulator import (
    CacheSimulator, MultimodalRequest, _read_sample_bytes,
)

CASE_NAME = "same_image_different_resize"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_case_data():
    """相同图片, 不同 resize。"""
    img = _read_sample_bytes()
    text = [101, 102, 103, 104, 105]

    r1 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img],
        original_sizes=[(1024, 768)],
        resized_sizes=[(336, 336)],
        patch_grids=[(12, 12)],
        num_visual_tokens_list=[256],
    )
    r2 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img],
        original_sizes=[(1024, 768)],
        resized_sizes=[(672, 672)],    # ← 不同 resize
        patch_grids=[(24, 24)],         # ← 不同 grid
        num_visual_tokens_list=[576],   # ← 不同 visual token 数
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

    print(f"\n{'='*60}")
    print(f"  Case: {CASE_NAME}")
    print(f"{'='*60}")
    for strat in ("A", "B", "C"):
        v = results[strat]
        s = all_stats[strat]
        note = ""
        if strat in ("A", "B") and v["verdict"] == "true_hit":
            note = " ⚠️ 语义不安全 (resize 不同, visual layout 不匹配)"
        print(f"  策略 {strat}: verdict={v['verdict']:12s}  "
              f"true_hits={s['true_hits']}  false_hits={s['false_hits']}  "
              f"safe_misses={s['safe_misses']}{note}")

    out = {
        "case": CASE_NAME,
        "description": (
            "same image bytes, different resize params. "
            "A/B hit (unsafe - resize mismatch), C correctly misses."
        ),
        "note": (
            "策略 A/B 的 hit 是因为 image_bytes 相同 → key 匹配。"
            "但语义上 resize 不同导致 visual token 布局不同 (num_visual_tokens: 256→576), "
            "KV cache 无法安全复用。只有策略 C 通过纳入 resized_sizes 等参数能正确识别此差异。"
        ),
        "per_strategy": {s: {"verdict": results[s]["verdict"], **all_stats[s]}
                         for s in ("A", "B", "C")},
    }
    json_path = os.path.join(RESULTS_DIR, f"{CASE_NAME}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON → {json_path}")

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
        if s == "C":
            vc = "#4caf50"
        elif s in ("A", "B"):
            vc = "#e65100"  # 警告色
        else:
            vc = "#333"
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
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px}}
th{{background:#1a1a2e;color:#fff;padding:12px 16px;text-align:left;font-size:13px}}
td{{padding:10px 16px;border-bottom:1px solid #eee;font-size:14px}}
tr:last-child td{{border-bottom:none}}
.warn-box{{padding:16px;background:#fff3e0;border-left:4px solid #ff9800;border-radius:4px;margin-bottom:24px}}
.key-finding{{padding:16px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px}}
.key-finding h3, .warn-box h3{{margin-bottom:8px}}
</style></head>
<body>
<h1>📐 {case}</h1>
<p class="sub">描述: {data['description']}</p>
<div class="warn-box">
  <h3>⚠️ 语义警告</h3>
  <p>{data['note']}</p>
</div>
<table><thead><tr>
  <th>策略</th><th>判定</th><th>true_hits</th><th>false_hits</th>
  <th>safe_misses</th><th>prefill_tokens_saved</th><th>kv_blocks_reused</th><th>memory_saved</th>
</tr></thead><tbody>{rows}</tbody></table>
<div class="key-finding">
  <h3>🔍 关键发现</h3>
  <p><strong>同一张图片在不同分辨率下, visual token 数量和 layout 可能完全不同。</strong></p>
  <p>即使对于策略 B (含 image SHA-256), 相同图片内容也会产生相同的 key, 导致不同分辨率的 KV cache 被错误复用 (visual tokens: 256 → 576, 拼接后的序列长度和布局不匹配)。</p>
  <p>只有策略 C (full_multimodal) 通过纳入 resized_sizes、patch_grids、num_visual_tokens_list 等参数, 能正确区分不同分辨率下的 cache 条目。</p>
</div>
</body></html>"""


if __name__ == "__main__":
    run()
