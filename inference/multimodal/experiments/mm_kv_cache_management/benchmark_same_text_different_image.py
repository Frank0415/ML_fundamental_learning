#!/usr/bin/env python3
"""
Case 2: same_text_different_image — **关键验收场景**。

预期 (硬性验收):
- 策略 A: false_hit_count > 0  (text 相同 → key 匹配, 但 image 不同 → 语义错配)
- 策略 B: false_hit_count = 0  (image hash 不同 → key 不同 → safe miss)
- 策略 C: false_hit_count = 0  (全量元数据不同 → safe miss)

此 case 是 Task 11 的核心验收指标。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_kv_cache_management.mm_cache_simulator import (
    CacheSimulator, MultimodalRequest, _read_sample_bytes, _make_synthetic_bytes,
)

CASE_NAME = "same_text_different_image"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_case_data():
    """构造两个请求: 相同文本, 不同图片。"""
    img_a = _read_sample_bytes()                    # 真实 demo.jpg
    img_b = _make_synthetic_bytes(seed=42)           # 合成 "不同图片"
    text = [101, 102, 103, 104, 105]

    r1 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img_a],
        original_sizes=[(512, 512)],
        resized_sizes=[(336, 336)],
        patch_grids=[(12, 12)],
        num_visual_tokens_list=[256],
    )
    r2 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img_b],
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

    # ---- 验收检查 ----
    a_false = all_stats["A"]["false_hits"]
    b_false = all_stats["B"]["false_hits"]
    c_false = all_stats["C"]["false_hits"]
    acceptance = (a_false > 0 and b_false == 0 and c_false == 0)

    print(f"\n{'='*60}")
    print(f"  Case: {CASE_NAME}  {'✅ ACCEPTANCE PASS' if acceptance else '❌ ACCEPTANCE FAIL'}")
    print(f"{'='*60}")
    for strat in ("A", "B", "C"):
        v = results[strat]
        s = all_stats[strat]
        print(f"  策略 {strat}: verdict={v['verdict']:12s}  "
              f"true_hits={s['true_hits']}  false_hits={s['false_hits']}  "
              f"safe_misses={s['safe_misses']}")
    print(f"\n  验收: A.false_hits={a_false} (需>0)  B.false_hits={b_false} (需=0)  C.false_hits={c_false} (需=0)")

    # 写 JSON
    out = {
        "case": CASE_NAME,
        "description": "same text, different image — KEY: A must false_hit>0, B/C must false_hit=0",
        "acceptance_pass": acceptance,
        "acceptance_detail": {
            "A_false_hits": a_false, "A_required": ">0",
            "B_false_hits": b_false, "B_required": "==0",
            "C_false_hits": c_false, "C_required": "==0",
        },
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
        if v == "false_hit":
            vc = "#f44336"
        elif v == "safe_miss":
            vc = "#4caf50"
        else:
            vc = "#ff9800"
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

    acc = data["acceptance_pass"]
    acc_color = "#4caf50" if acc else "#f44336"
    acc_text = "✅ 通过" if acc else "❌ 未通过"

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
.acceptance{{padding:16px;border-radius:4px;margin-bottom:24px;font-weight:bold;font-size:16px;color:{acc_color};background:#e8f5e9}}
.key-finding{{padding:16px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px}}
.key-finding h3{{margin-bottom:8px}}
</style></head>
<body>
<h1>🔬 {case}</h1>
<p class="sub">描述: {data['description']}</p>
<div class="acceptance">验收状态: {acc_text}  (A false_hits{'>' if data['acceptance_detail']['A_required'] == '>0' else ''}{data['acceptance_detail']['A_required']}, B{data['acceptance_detail']['B_required']}, C{data['acceptance_detail']['C_required']})</div>
<table><thead><tr>
  <th>策略</th><th>判定</th><th>true_hits</th><th>false_hits</th>
  <th>safe_misses</th><th>prefill_tokens_saved</th><th>kv_blocks_reused</th><th>memory_saved</th>
</tr></thead><tbody>{rows}</tbody></table>
<div class="key-finding">
  <h3>🔍 关键发现</h3>
  <p><strong>text-only prefix cache 在多模态下不安全。</strong></p>
  <p>当两个请求使用相同文本但不同图像时, 策略 A (仅 hash 文本) 错误地判定为命中 (false_hit), 导致 KV cache 被复用给不匹配的图像上下文。</p>
  <p>策略 B 通过纳入图像 SHA-256 避免了此问题。策略 C 在此基础上进一步将图像处理参数 (resize/grid/layout/order) 纳入 key, 提供最严格的 cache 安全性。</p>
  <p><strong>建议</strong>: 多模态推理中, 前缀 cache 的 key 必须至少包含图像内容的 hash (策略 B), 理想情况下应包含完整的图像处理元数据 (策略 C)。</p>
</div>
</body></html>"""


if __name__ == "__main__":
    run()
