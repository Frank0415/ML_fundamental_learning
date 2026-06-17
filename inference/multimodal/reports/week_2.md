# Week 2 进度报告

> **日期**: 2026-06-07
> **Wave**: Wave 2 — Engine Patch + KV Cache Integration

---

## Task 4: Engine Patch（HF 对齐） — ✅ 完成

- 修复 `Qwen3Attn` → `Attn` 参数不兼容（`S=` → `head_dim=`, `is_decode=` → `is_causal=True`）
- 修复 `Qwen3FFN.act_fn = None` → `SiluAndMul()`
- 修复 RoPE `rope_theta` 透传（`config.rope_parameters.get("rope_theta", 10000)`）
- **HF 对齐结果**: max |diff|=8.2e-5, cosine sim=0.99999994, verdict=IDENTICAL

## Task 5: KV Cache 接入 Forward 路径 — ✅ 完成（达到最小接受标准）

### 改动文件
- `minivLLM/minivllm/model/qwen3.py`：KV cache 接入到 Qwen3Attn/Qwen3DecoderLayer/Qwen3Model/Qwen3 的 forward 路径
- **新增** `experiments/text_engine_audit/audit_kv_cache_compare.py`：对比测试脚本
- 未修改 `attention.py`、`kv_cache.py`、`context.py`

### 回退检查
- `validate_model.py --full` → ALL CHECKS PASSED（无回归）
- `rope.py` / `attention.py` / `kv_cache.py` 数学未动
- HF 对齐未破坏

### 对齐测试结果

| seq_len | max\|diff\| | cosine sim | 判定 |
|---------|-------------|------------|------|
| 1 | 0.00e+00 | 1.000005 | PASS |
| 8 | 4.65e-06 | 1.000004 | PASS |
| 64 | 5.01e-06 | 1.000002 | PASS |
| 512 | 4.29e-06 | 1.000002 | PASS |

**全部通过**（阈值 atol=1e-5, rtol=1e-4）。

### 简化策略
- decode 时临时禁用 causal mask（`is_causal=False`），单 token 查询应看到所有已缓存 key
- paged attention 阶段（Task 6）可补正确的 offset mask

### 错误处理验证
- OOB write/read、负位置、空读、reset、batch write 全部通过（9/9）

### 当前状态: contiguous KV cache 已正式接线，prefill vs decode 对齐 ✅

## Task 6: 最小 Paged KV 路径 — ✅ 已实现（correctness-first）

### 是否达到最小 paged attention 接受标准

- 已新增外部模块：`BlockManager`、`BlockTable`、`RequestState`、`PagedKVCache` 与 lifecycle helpers。
- `prefill_allocate()`、`decode_append()`、`free_request()` 已覆盖 prompt 分配、decode 追加、请求释放与 block 复用。
- `gather_kv_for_attention()` 先按 block table 拼接 K/V 为 contiguous tensor，再复用 Task 5 的 attention 路径；未改 `minivLLM/`。
- 测试入口：`experiments/paged_attention_fix_or_impl/tests/run_paged_kv_checks.py --block-sizes 16 32`。
- 对齐入口：`experiments/paged_attention_fix_or_impl/benchmarks/compare_contiguous_vs_paged.py`，复用 Task 5 的 `audit_kv_cache_compare.py` 模型配置与 `cached_prefill_then_decode()` 参考逻辑。
- 结果写入：`experiments/paged_attention_fix_or_impl/results/` 与 `.omo/evidence/task-6-paged-{tests,compare}.txt`。

当前结论：达到最小接受标准；这是 correctness-first paged KV，不包含 scheduler、prefix sharing 或 CUDA paged attention kernel。
