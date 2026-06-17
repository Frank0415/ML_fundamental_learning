#!/usr/bin/env python3
"""
Case 5+6+7: 多图顺序 & 视频帧采样。

Case 5 - multi_image_same_order: 多图相同顺序 → 应命中 (所有策略)
Case 6 - multi_image_different_order: 多图不同顺序 → A false_hit, B/C miss
  - 策略 A: text 相同 → false_hit (image 顺序不同但 key 相同)
  - 策略 B: image hash 在 key 中按序拼接 → 顺序不同则 key 不同 → safe_miss
  - 策略 C: multi_image_order 纳入 key → safe_miss
Case 7 - same_video_different_frame_sampling: 设计说明占位 (未实现视频)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_kv_cache_management.mm_cache_simulator import (
    CacheSimulator, MultimodalRequest, _read_sample_bytes, _make_synthetic_bytes,
)

CASE_NAME = "multi_image_order"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def case5_same_order():
    """多图相同顺序 → 应命中。"""
    img_a = _read_sample_bytes()
    img_b = _make_synthetic_bytes(seed=99)
    text = [101, 102, 103, 104, 105]

    r1 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img_a, img_b],
        multi_image_order="image_0,image_1",
        original_sizes=[(512, 512), (512, 512)],
        resized_sizes=[(336, 336), (336, 336)],
        patch_grids=[(12, 12), (12, 12)],
        num_visual_tokens_list=[256, 256],
    )
    r2 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img_a, img_b],   # 相同顺序
        multi_image_order="image_0,image_1",
        original_sizes=[(512, 512), (512, 512)],
        resized_sizes=[(336, 336), (336, 336)],
        patch_grids=[(12, 12), (12, 12)],
        num_visual_tokens_list=[256, 256],
    )
    return r1, r2


def case6_different_order():
    """多图不同顺序 → A false_hit, B/C miss。"""
    img_a = _read_sample_bytes()
    img_b = _make_synthetic_bytes(seed=99)
    text = [101, 102, 103, 104, 105]

    r1 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img_a, img_b],
        multi_image_order="image_0,image_1",
        original_sizes=[(512, 512), (512, 512)],
        resized_sizes=[(336, 336), (336, 336)],
        patch_grids=[(12, 12), (12, 12)],
        num_visual_tokens_list=[256, 256],
    )
    r2 = MultimodalRequest(
        text_token_ids=list(text),
        image_bytes_list=[img_b, img_a],   # ← 顺序反转
        multi_image_order="image_1,image_0",
        original_sizes=[(512, 512), (512, 512)],
        resized_sizes=[(336, 336), (336, 336)],
        patch_grids=[(12, 12), (12, 12)],
        num_visual_tokens_list=[256, 256],
    )
    return r1, r2


def run_case(label, r1, r2, expected_desc):
    print(f"\n  --- {label} ---")
    results = {}
    all_stats = {}
    for strat in ("A", "B", "C"):
        sim = CacheSimulator(strategy=strat)
        sim.insert(r1, prefill_tokens=len(r1.text_token_ids) + 512, kv_blocks_used=44)
        verdict = sim.query(r2)
        stats = sim.stats()
        results[strat] = verdict
        all_stats[strat] = stats

    for strat in ("A", "B", "C"):
        v = results[strat]
        s = all_stats[strat]
        print(f"    策略 {strat}: verdict={v['verdict']:12s}  "
              f"true_hits={s['true_hits']}  false_hits={s['false_hits']}  "
              f"safe_misses={s['safe_misses']}")

    return {
        "label": label,
        "expected": expected_desc,
        "per_strategy": {s: {"verdict": results[s]["verdict"], **all_stats[s]}
                         for s in ("A", "B", "C")},
    }


def run():
    r1_c5, r2_c5 = case5_same_order()
    r1_c6, r2_c6 = case6_different_order()

    c5 = run_case("Case 5: multi_image_same_order", r1_c5, r2_c5,
                  "所有策略应命中 (完全相同的多图请求)")
    c6 = run_case("Case 6: multi_image_different_order", r1_c6, r2_c6,
                  "A: false_hit > 0; B/C: safe_miss")

    # ---- Case 7: 视频帧采样 (说明占位) ----
    print(f"\n  --- Case 7: same_video_different_frame_sampling ---")
    print(f"    ⚠️ 未实现视频 (说明占位)")
    print(f"    视频帧采样差异与多图类似:")
    print(f"      策略 A: text 相同 → false_hit (帧不同但 key 相同)")
    print(f"      策略 B: 帧 hash 不同 → key 不同 → miss")
    print(f"      策略 C: frame_sampling_meta 不同 → key 不同 → miss")

    c7 = {
        "label": "Case 7: same_video_different_frame_sampling",
        "expected": "未实现视频 — 说明: 不同帧采样参数会导致 A false_hit, B/C safe_miss",
        "status": "placeholder",
        "per_strategy": {
            "A": {"verdict": "would_be_false_hit", "note": "text 相同, 帧不同 → key 相同但语义错配"},
            "B": {"verdict": "would_be_safe_miss", "note": "帧 hash 纳入 key → 帧不同则 key 不同"},
            "C": {"verdict": "would_be_safe_miss", "note": "frame_sampling_meta 纳入 key → 严格区分"},
        },
    }

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  Case: {CASE_NAME} (Cases 5+6+7)")
    print(f"{'='*60}")

    out = {
        "case": CASE_NAME,
        "description": "Multi-image order sensitivity + video frame sampling (placeholder)",
        "sub_cases": [c5, c6, c7],
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
    sections = ""
    for sc in data["sub_cases"]:
        ps = sc.get("per_strategy", {})
        rows = ""
        for s in ("A", "B", "C"):
            st = ps.get(s, {})
            v = st.get("verdict", "N/A")
            if "false_hit" in v:
                vc = "#f44336"
            elif "safe_miss" in v:
                vc = "#4caf50"
            elif "true_hit" in v:
                vc = "#4caf50"
            else:
                vc = "#ff9800"
            note = st.get("note", "")
            rows += f"""
            <tr>
              <td><strong>策略 {s}</strong></td>
              <td style="color:{vc};font-weight:bold">{v}</td>
              <td>{st.get('true_hits','-')}</td>
              <td>{st.get('false_hits','-')}</td>
              <td>{st.get('safe_misses','-')}</td>
              <td>{st.get('prefill_tokens_saved','-')}</td>
              <td>{st.get('kv_blocks_reused','-')}</td>
              <td>{note}</td>
            </tr>"""
        sections += f"""
        <h2>{sc['label']}</h2>
        <p class="sub">预期: {sc['expected']}</p>
        <table><thead><tr>
          <th>策略</th><th>判定</th><th>true_hits</th><th>false_hits</th>
          <th>safe_misses</th><th>prefill_tokens_saved</th><th>kv_blocks_reused</th><th>备注</th>
        </tr></thead><tbody>{rows}</tbody></table>
        """

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>{case} — KV Cache Report</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#333;padding:32px}}
h1{{font-size:24px;margin-bottom:8px}}
h2{{font-size:18px;margin:24px 0 8px}}
.sub{{color:#666;margin-bottom:16px}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:24px}}
th{{background:#1a1a2e;color:#fff;padding:12px 16px;text-align:left;font-size:13px}}
td{{padding:10px 16px;border-bottom:1px solid #eee;font-size:14px}}
tr:last-child td{{border-bottom:none}}
.key-finding{{padding:16px;background:#fff3cd;border-left:4px solid #ffc107;border-radius:4px}}
.key-finding h3{{margin-bottom:8px}}
</style></head>
<body>
<h1>🔄 {case}</h1>
<p class="sub">描述: {data['description']}</p>
{sections}
<div class="key-finding">
  <h3>🔍 关键发现</h3>
  <p><strong>多图顺序敏感性是 text-only cache 的又一个安全隐患。</strong></p>
  <p>当多张图片以不同顺序排列时 (如 [imgA, imgB] vs [imgB, imgA]), 策略 A 仅看文本会错误命中, 但 visual token 在序列中的排列顺序不同, KV cache 的语义对应关系完全错位。</p>
  <p>策略 B 通过将图片 hash 按序拼接, 能天然捕捉到顺序差异。策略 C 更进一步通过独立的 multi_image_order 字段显式保证顺序敏感性。</p>
  <p><strong>视频场景</strong>: 不同帧采样参数 (如 fps=1 vs fps=0.5) 产生不同的帧序列, 与多图顺序问题本质相同。需要策略 B 或 C 级别的 key 才能安全区分。</p>
</div>
</body></html>"""


if __name__ == "__main__":
    run()
