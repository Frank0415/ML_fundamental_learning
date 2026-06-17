# Week 1 周报：Atlas Wave 1 启动审计

日期：2026-06-01 至 2026-06-07

---

## 1. 审计结论摘要

Week 1 完成了 minivLLM 推理框架的全面**审计**，聚焦三个维度：文本引擎现状、多模态流水线完整性、paged attention 就绪度。

### 1.1 文本引擎审计

- minivLLM 的纯文本推理路径（Tokenizer → Embed → Transformer → LM Head → Sampler）结构完整，`config.py` 已提供 `kvcache_block_size=16` 的配置锚点。
- **阻塞发现**：`paged_attention` 路径在整个框架中**未实现**。当前所有 KV 缓存操作 fallback 到 contiguous 实现，无 block 管理、无动态分配/释放。这导致 12GB GPU 上 `max_model_len` 受限于预分配连续内存的上限。
- 文本推理的 attention 路径在 contiguous 模式下已具备参考正确性，可作为 paged attention 对齐的基础。

### 1.2 多模态流水线审计

- `mm_token_pipeline` 目录下已规划 VLM 的 visual token 注入路径，但**未实现**与 paged KV 的对接。
- `mm_kv_cache_management` 目录仅占位，没有具体的 visual token KV 块分配逻辑。
- **阻塞**：没有 paged KV 底座，visual token 的 KV 写入就没有安全的显存边界。这阻断了 VLM 端到端推理。

### 1.3 Paged Attention 就绪度

- `minivLLM/minivllm/config.py` 的 `kvcache_block_size=16` 和 `num_kvcache_blocks=-1` 已就绪，但没有任何 consumer 代码使用这些参数。
- `gpu_memory_utilization=0.9` 定义了显存安全边界，可按此计算 `total_blocks`。
- **结论**：基础设施（config）就绪，核心实现（BlockManager、PagedKVCache）**未实现**。这是 Week 2 的最优先事项。

### 1.4 审计关键词映射

| 关键词 | 审计状态 | 说明 |
|--------|----------|------|
| **审计** | ✅ 完成 | minivLLM 文本引擎、多模态流水线、paged attention 三条线已审计 |
| **未实现** | 🔴 阻塞 | paged_attention 核心代码未实现；BlockManager / PagedKVCache 不存在 |
| **阻塞** | 🔴 存在 | 无 block 管理 = VLM 推理路径被阻断 |
| **paged attention** | 🔴 待实现 | 设计已固化（见 `design.md`），代码待 Week 2 完成 |
| **contiguous** | 🟢 可用 | 作为 paged attention 的对齐参考 |

---

## 2. Week 2 唯一关键路径

Week 2 只有一条关键路径，不允许 fork 到其他方向：

```
基础正确性 → contiguous KV → paged KV
```

具体步骤（严格顺序）：

### 步骤 1：基础正确性

- 用纯文本模型（Qwen2.5-0.5B 或 SmolVLM2-2.2B text-only）跑通 minivLLM 的 prefill + decode 循环。
- 输出 logits 与 HF transformers 的 `model.generate()` 对齐（`allclose(atol=1e-4)`）。
- **通过条件**：单 token prefill + 单步 decode logits 对齐。

### 步骤 2：contiguous KV

- 在步骤 1 的基础上，将 KV 缓存储存在 contiguous 张量中（不做 block 切分）。
- 验证 prefill 写入、decode 追加、多步 decode 后 logits 一致。
- **通过条件**：N 步 decode 后 logits 与无 KV cache 的逐个 prefill 对齐。

### 步骤 3：paged KV

- 用 `design.md` 中的接口实现 `BlockManager`、`BlockTable`、`RequestState`、`PagedKVCache`。
- 用 `prefill_allocate()`、`decode_append()`、`free_request()` 管理 KV 块。
- 用 `gather_kv_for_attention()` 拼接 K/V，与步骤 2 的 contiguous KV 做 logits 对齐。
- **通过条件**：`allclose(atol=1e-5, rtol=1e-5)` + MAC-1~MAC-5 全部通过。

### 关键路径之外不做的

- 不做 scheduler / continuous batching
- 不做 prefix sharing / cascade attention
- 不做定制 CUDA kernel
- 不做多卡 / tensor parallel
- 不做 VLM 视觉注入（等 paged KV 稳定后再对接）

---

## 3. Week 1 交付物清单

| 文件 | 内容 | 状态 |
|------|------|------|
| `experiments/paged_attention_fix_or_impl/README.md` | 项目目的、MAC、运行命令 | ✅ 已交付 |
| `experiments/paged_attention_fix_or_impl/design.md` | 接口设计、指标、对齐阈值、接口签名表 | ✅ 已交付 |
| `learning/notes/08_12gb显存预算.md` | 4 模型显存预算矩阵 | ✅ 已交付 |
| `reports/week_1.md` | 本周报 | ✅ 本文件 |

---

## 4. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| paged KV 实现引入 offset bug | logits 不对齐，Week 2 卡住 | 步骤 2 contiguous KV 作为 solid baseline，diff 排查 |
| 12GB 显存不够跑 VLM | 只能 text-only 验证 | 已有 4 模型 fallback 链 + text-only 降级路径 |
| Qwen3-VL-4B 的 ViT 过大 | prefill 时 activation OOM | 降 `max_pixels`；换 Qwen2.5-VL-3B |
| 与并行 agent 的交付物冲突 | 重复工作 | 仅写设计文档，不实现代码，不触碰 `minivLLM/` 源码 |

---

*Atlas Wave 1 / Task 3 执行完成。Week 2 从步骤 1 基础正确性开始。*
