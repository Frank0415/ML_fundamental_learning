# 最终报告：minivLLM 多模态推理实验工作区

> 日期: 2026-06-07
> 关联: Wave 5 / Task 12 — 最终成果说明
> 本文档是 6 周最小 VLM 执行计划的最终交付物。

---

## 执行总结

本项目在约 此前按受限显存假设做规划时，从纯文本推理引擎 `minivLLM` 出发，经过 6 周（Wave 1-5）逐步构建了一个可运行的最小多模态（VLM）推理实验流水线。核心成果包括：文本引擎 HF-IDENTICAL 对齐、contiguous 与 paged KV cache 接入、inputs_embeds 路径打通、最小 VLM demo 工程跑通、4 个 VLM reference 模型的对照矩阵（含降级路径验证）、以及多模态 KV cache key 管理的完整实验（确认 text-only prefix cache 在多模态下的 false_hit 风险）。

**诚实声明**：4 个 VLM reference 模型（Qwen3-VL-4B / Qwen2.5-VL-3B / InternVL3.5-4B / SmolVLM2-2.2B）在本机 macOS (MPS, 无 CUDA, 缺 `accelerate`) 环境下全部加载失败。未假装成功。降级到 processor-only smoke，tokenizer 验证通过。Local VLM demo 使用随机 projector，不保证语义质量。

---

## 6 周路线实际完成度

| Wave | 主题 | 计划任务 | 实际状态 | 备注 |
|------|------|----------|----------|------|
| Wave 1 | 目录骨架 + 引擎盘点 | 创建全目录、engine_inventory.md、显存预算与模型选型矩阵 | ✅ 完成 | 18 模块静态审计，标识 2 个阻塞 Bug + 2 项未接线 |
| Wave 2 | 文本引擎审计 + Engine Patch | 修复 B1/B2/B3、HF 对齐、KV cache 接入 | ✅ 完成 | 5 Bug 修复，HF IDENTICAL（max\|diff\|=8.2e-5），contiguous KV cache 接入 |
| Wave 3 | Paged Attention + Token Pipeline | PagedKVCache 实现、教学型 token pipeline | ✅ 完成 | BlockManager+BlockTable+correctness-first gather；5 模块 pipeline（2 种 visual token 模式） |
| Wave 4 | inputs_embeds + VLM Demo + VLM Reference | inputs_embeds 接入、最小 VLM demo、4 模型 reference | ✅ 完成（降级） | inputs_embeds text_parity 0.00 diff；local VLM demo 工程跑通；4 VLM reference 全部 fail → 降级 smoke |
| Wave 5 | mm KV Cache 管理 + 收尾 | 3 策略 × 7 场景实验、最终成果说明 | ✅ 完成 | 策略 A false_hit 验收通过；策略 B/C 正确；最终文档产出 |

---

## 关键验证数字

### 纯文本引擎：HF Parity

| 指标 | 值 | 判定 |
|------|-----|------|
| max \|diff\| | 8.2e-5 | ✅ |
| cosine similarity | 0.99999994 | ✅ IDENTICAL |
| 测试模型 | Qwen3-0.6B (随机权重) | |
| 测试脚本 | `validate_model.py --compare-hf --full` | |

### KV Cache 对齐

| 测试 | 结果 |
|------|------|
| contiguous KV cache (seq_len=1/8/64/512) | ✅ allclose (atol=1e-5, rtol=1e-4) |
| paged KV cache vs contiguous | ✅ allclose (correctness-first) |
| KV cache 越界/空读/复位 | ✅ 全部通过 |

### inputs_embeds 路径

| 测试 | 结果 |
|------|------|
| text_parity (input_ids == inputs_embeds) | ✅ max\|diff\|=0.00e+00 |
| invalid_dual_input | ✅ ValueError caught |
| HF parity 无回归 | ✅ IDENTICAL 保持 |

### 多模态 KV Cache 管理

| 场景 | 策略 A false_hit | 策略 B false_hit | 策略 C false_hit |
|------|-----------------|------------------|------------------|
| same_text_different_image (关键验收) | **1** | 0 | 0 |
| multi_image_different_order | **1** | 0 | 0 |
| same_image_different_resize | 0* (语义警告) | 0* (语义警告) | 0 |

> *Case 4 中策略 A/B 的 false_hit=0 但语义上不安全（resize 不同 → visual layout 不匹配）。仅靠 image_bytes hash 判定不足，resize 差异需单独建模。

---

## 已完成项

1. **目录骨架与文档入口**：`multimodal/` 全目录创建，`docs/index.html` 导航页面发布，`style.css` 统一样式。
2. **引擎静态审计**：`engine_inventory.md` 覆盖 18 个模块，标识 2 个阻塞 Bug + 2 项未接线。
3. **Engine Patch**：修复 Attn 参数不兼容、act_fn=None、head_dim 接线、RoPE 实现、rope_theta 传参共 5 个 Bug。
4. **HF 对齐**：`validate_model.py --compare-hf --full` 输出 `verdict: IDENTICAL`。
5. **Contiguous KV cache 接入**：`KVCache` 类写入 prefill/decode 路径，4 组 seq_len 对齐。
6. **Paged KV cache 实现**：`BlockManager` + `BlockTable` + `PagedKVCache` + `gather_kv_for_attention`，与 contiguous 对齐。
7. **教学型 Token Pipeline**：5 个管线模块（图像预处理 + Patch Embed + Visual Token + 序列构造 × 2 种布局）。
8. **inputs_embeds 路径**：双输入冲突拒绝，text_parity 通过，HF parity 无回归。
9. **最小 VLM Demo**：随机 tiny-ViT + 随机 projector，prefill_only / prefill_decode 模式工程跑通。
10. **VLM Reference 矩阵**：4 模型 cascade 脚本，降级路径（tokenizer-only smoke）生效，4 份 .fail.json + processor_only_smoke.json。
11. **多模态 KV Cache 管理**：3 策略 × 7 场景纯 Python 模拟器，策略 A false_hit 验收通过，5 份 JSON + 5 份 HTML 结果报告。
12. **静态文档**：`docs/01_*.html` 至 `docs/10_*.html` 共 10 篇完整中文技术文档，覆盖 engine audit 至 final summary。
13. **学习资料**：12 篇核心论文笔记 + 12 篇学习笔记（覆盖 Transformer 至 SGLang RadixAttention）。
14. **周报**：`reports/week_1.md` 至 `reports/week_6.md` 共 6 篇完整进度报告。

---

## 未完成项

| 项目 | 原因 | 后续建议 |
|------|------|----------|
| 4 个 VLM reference 模型成功运行 | macOS 缺 `accelerate` 包 + 无 CUDA | `pip install accelerate` 后重试；InternVL 需排查 config 注册 |
| Local VLM demo 产生有语义输出 | 使用 random projector，非 pretrained 权重 | 替换为 HF pretrained visual.merger |
| Paged attention CUDA kernel | 无 CUDA 环境，仅实现 correctness-first gather | 有 GPU 时实现 GPU-native paged_attention |
| Scheduler / Batch 推理 | minivLLM 无 scheduler | 参照 vLLM 架构实现 |
| mm cache 接入真实推理 | 模拟器未接入引擎读/写循环 | 迁移策略到 minivLLM KV cache 读/写 |

---

## 限制项

1. **无 CUDA 环境**：所有实验在 macOS + MPS 上运行，性能数字不代表 GPU 表现。
2. **缺 `accelerate` 包**：`device_map="auto"` 强制依赖，3 个 Qwen/SmolVLM 模型因此加载失败。
3. **Random projector**：`run_minimal_vlm.py` 的 visual-to-text projector 是随机初始化，不携带语义。
4. **纯 Python mm cache 模拟器**：策略设计完整，但未接入真实推理引擎。
5. **单序列推理**：无 batching，所有实验均为 batch_size=1。
6. **显存预算为理论估算**：未通过真实 GPU memory_stats 校准。
7. **InternVL3.5-4B 额外兼容性问题**：`InternVLChatConfig` 不被 `AutoModelForImageTextToText` 识别。

---

## 失败案例：4 个 VLM Reference 全失败

**执行环境**：macOS (Apple Silicon, MPS), transformers 5.8.0, torch 2.11.0, CUDA=False

| 模型 | 状态 | 失败原因 |
|------|------|----------|
| Qwen3-VL-4B-Instruct | ❌ | `ValueError`: `device_map="auto"` 需要 `accelerate` 包 |
| Qwen2.5-VL-3B-Instruct | ❌ | `ValueError`: `device_map="auto"` 需要 `accelerate` 包 |
| InternVL3.5-4B | ❌ | `ValueError`: `InternVLChatConfig` 不被 `AutoModelForImageTextToText` 识别 |
| SmolVLM2-2.2B-Instruct | ❌ | `ValueError`: `device_map="auto"` 需要 `accelerate` 包 |

**4 个 VLM reference 在本机环境下全部因缺 `accelerate` / 无 CUDA 失败；降级到 processor-only smoke**。降级 smoke 使用 `Qwen/Qwen3-0.6B` tokenizer-only 模式，tokenizer 加载正确，11 token round-trip 解码正确。

修复命令（非本任务范围）：
```bash
minivLLM/.venv/bin/pip install accelerate
```

## 失败案例：mm cache 策略 A false_hit

在 `same_text_different_image` 场景下，仅基于文本 token ID 的 SHA-256 cache key（策略 A）无法区分"相同文本前缀 + 不同图像输入"的情况，导致错误复用 KV cache。策略 B（text + image_hash）和策略 C（full multimodal metadata）均正确 miss。

策略 A false_hit 是多模态 prefix cache 的核心风险——任何仅基于文本的 cache key 在生产环境中会错误地将不同视觉上下文的 KV cache 当作命中，导致模型基于错误的 visual context 解码。

---

## 后续建议

1. **安装 `accelerate` 并重试 VLM reference**：`minivLLM/.venv/bin/pip install accelerate`，重新运行 `run_qwen_vl_reference.py` 验证 3 个模型能否在 MPS 上加载。
2. **排查 InternVL3.5-4B**：`InternVLChatConfig` 的 AutoModel 注册问题可能需要特定加载器或更新的 transformers 版本。
3. **替换 random projector 为 HF pretrained**：将 `run_minimal_vlm.py` 中的随机 `nn.Linear` 替换为真实 visual.merger 权重，使 local VLM demo 具备语义。
4. **将 mm cache 策略接入真实推理循环**：迁移 `mm_cache_simulator.py` 的 3 策略设计到 minivLLM 引擎 KV cache 读/写循环。
5. **在 NVIDIA GPU 上验证 paged attention 性能**：运行 `compare_contiguous_vs_paged.py`，测量 real fragmentation ratio 和 throughput。
6. **实现 scheduler 与 batch 推理**：参照 vLLM scheduler 架构，实现 ReqState / Lifecycle / 批量调度。
7. **真实 GPU memory_stats 校准**：在有 CUDA 的 GPU 上用 `torch.cuda.memory_stats()` 校准 显存预算与模型选型。

---

> **本文件由 Wave 5 / Task 12 子任务执行者写入。所有数据均为实际运行结果，不夸大、不修饰、不假装成功。**
