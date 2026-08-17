# 多模态推理实验工作区 (Atlas Wave 5 - 最终交付)

本目录是 Atlas 计划的多模态推理实验工作区。经过 6 周（Wave 1-5），已完整体成引擎审计、HF 对齐、paged attention 实现、多模态 token pipeline、inputs_embeds 路径接入、最小 VLM demo、4 模型 VLM reference 对照矩阵、以及多模态 KV cache 管理实验。

**所有命令从 `multimodal/` 目录执行。** macOS 环境无系统 `python` 命令，使用 `minivLLM/.venv/bin/python` 替代。

---

## 1. 当前目录结构

本节目录结构与 Wave 1 保持一致，各子目录已填充完整内容。使用说明与运行入口如下。

```
multimodal/
├── docs/                        # 文档与静态页面（入口: docs/index.html）
├── docs_md/                     # 文档的 Markdown 镜像（入口: docs_md/README.md）
│   ├── README.md                # docs_md/ 目录说明
│   ├── 00_index.md              # 文档导航（与 docs/index.html 等价）
│   ├── 01_*.md ~ 10_*.md        # 10 篇技术文档的 MD 版
│   ├── papers/                  # 12 篇论文的中文笔记（与 learning/papers/ 同步）
│   └── notes/                   # 9 篇学习笔记（与 learning/notes/ 同步）
├── paper/                       # 12 篇核心论文 PDF 原件
├── learning/                    # 学习资料（笔记原始位置）
│   ├── papers/                  # 12 篇论文中文笔记
│   └── notes/                   # 9 篇学习笔记（engine audit → SGLang）
├── experiments/                 # 实验区（23 个脚本）
│   ├── text_engine_audit/       # 文本引擎审计（4 个审计脚本 + results/）
│   ├── paged_attention_fix_or_impl/  # Paged Attention 实现
│   │   ├── tests/               # PagedKVCache correctness 测试
│   │   ├── benchmarks/          # contiguous vs paged 对比基准
│   │   └── results/             # 实验结果（JSON + HTML）
│   ├── mm_token_pipeline/       # 多模态 Token Pipeline（5 模块）
│   │   └── results/             # Shape 契约
│   ├── vlm_minimal_demo/        # VLM 最小可运行 Demo + VLM Reference
│   │   ├── sample_images/       # 示例图片
│   │   └── results/             # Demo 结果 + 4 份 fail.json + smoke
│   └── mm_kv_cache_management/  # 多模态 KV Cache 管理
│       └── results/             # 5 份 JSON + 5 份 HTML（3 策略 × 7 场景）
├── reports/                     # 报告区（engine_inventory + week_1~6 + final_report）
├── minivLLM/                    # 当前文本引擎（Python + PyTorch + Transformers）
└── .omo/                        # Atlas 内部元数据（plans/evidence/notepads）
```

所有实验和工作产物均在 `multimodal/` 下，未移动 `minivLLM/`，未创建新 git repo。

---

## 2. 如何打开 docs/index.html

静态文档入口页面，无需服务器，直接在浏览器打开。

```bash
# 从仓库根目录（/Users/franksair/Documents/learning_ML）：
open multimodal/docs/index.html

# 或从 multimodal/ 目录：
open docs/index.html
```

页面包含 10 篇中文技术文档的完整导航（引擎审计 → 最终成果说明），以及学习笔记和 6 周周报的链接。纯静态 HTML + CSS，不依赖任何外部框架或 CDN。

---

## 3. 如何运行文本引擎审计

文本引擎审计包含静态审计（不 import 引擎）和运行时验证（需 PyTorch）。

### 静态审计（纯 Python，无依赖）

```bash
# Attention 实现审计 → results/attention.json
python3 experiments/text_engine_audit/audit_attention.py

# KV cache 静态审计 → results/kv_cache.json
python3 experiments/text_engine_audit/audit_kv_cache.py

# Paged attention 静态审计 → results/paged_attention.json
python3 experiments/text_engine_audit/audit_paged_attention.py
```

### 运行时验证（需 PyTorch + minivLLM）

```bash
# HF 对齐验证（核心验收）- 需要网络下载 HF config
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/text_engine_audit/audit_kv_cache_compare.py

# 或直接在 minivLLM 内运行：
minivLLM/.venv/bin/python minivLLM/validate_model.py --compare-hf --full
```

预期输出：`verdict: IDENTICAL, max |diff| ≈ 8.2e-5, cosine sim ≈ 1.0`。

审计发现 2 个构造时阻塞 Bug（Attn 参数不兼容 / act_fn=None）和 2 项未接线（KV cache 未接入 forward / Context 脚手架未调用），均已在 Wave 2 修复。详细审计报告见 `reports/engine_inventory.md`。

---

## 4. 如何运行 paged attention 测试

PagedAttention 实现为 correctness-first 版本（BlockManager + BlockTable + PagedKVCache + gather_kv_for_attention），使用 `torch.gather` 完成逻辑-物理地址映射，验证与 contiguous KV cache 在 logits 上对齐。

```bash
# PagedKVCache correctness 测试（含越界/空读/复位）
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/paged_attention_fix_or_impl/tests/run_paged_kv_checks.py

# contiguous vs paged 对比基准
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/paged_attention_fix_or_impl/benchmarks/compare_contiguous_vs_paged.py
```

PagedKVCache 已通过全部 correctness 测试，logits 与 contiguous cache 通过 `torch.allclose`。**注意**：无 CUDA kernel，decode 阶段使用 gather fallback；性能非优化，仅正确性验证通过。

---

## 5. 如何运行最小多模态 pipeline

### Token Pipeline 教学管线（纯 Python，无需 PyTorch）

5 个管线模块验证图像 → visual token 的完整 shape 契约：

```bash
# 图像预处理
python3 experiments/mm_token_pipeline/image_preprocess.py

# Patch Embed（Conv2d 模拟 ViT）
python3 experiments/mm_token_pipeline/patch_embed_demo.py

# Visual Token（tiny-vit-random / clip-reference）
python3 experiments/mm_token_pipeline/visual_token_demo.py

# 序列构造（bos_image_text / placeholder_expanded）
python3 experiments/mm_token_pipeline/mm_sequence_builder.py
```

### 最小 VLM Demo（需 PyTorch + minivLLM）

```bash
# prefill_only：视觉+文本 embedding 拼接，一次前向
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_minimal_vlm.py --mode prefill_only

# prefill_decode：prefill 写入 KV cache，decode 追加 max_new_tokens
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_minimal_vlm.py --mode prefill_decode --max-new-tokens 16

# text_parity：验证 input_ids == inputs_embeds 路径
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_minimal_vlm.py --mode text_parity
```

**重要说明**：此 demo 使用随机 tiny-ViT + 随机 linear projector，不保证语义质量。工程路径已跑通，真实语义需要 HF pretrained vision encoder 权重。

---

## 6. 如何运行 Qwen-VL reference

对 4 个 VLM 候选模型进行 reference 对照推理，按主路径 → 稳定 fallback → 先进对照 → 轻量对照顺序尝试。

```bash
# 默认参数（4 模型 cascade，失败时自动降级到 tokenizer-only smoke）
PYTHONPATH=minivLLM minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_qwen_vl_reference.py \
  --image experiments/vlm_minimal_demo/sample_images/demo.jpg \
  --prompt "请描述这张图片。" \
  --max-new-tokens 64
```

4 个模型候选：

| 模型 | 角色 |
|------|------|
| Qwen3-VL-4B-Instruct | 主路径 |
| Qwen2.5-VL-3B-Instruct | 稳定 fallback |
| InternVL3.5-4B | 先进对照 |
| SmolVLM2-2.2B-Instruct | 轻量对照 |

**当前状态**：本机 macOS (MPS, 无 CUDA) 环境下，因缺 `accelerate` 包，4 个模型全部加载失败。降级 smoke（`Qwen/Qwen3-0.6B` tokenizer-only）通过。需先执行：

```bash
minivLLM/.venv/bin/pip install accelerate
```

InternVL3.5-4B 额外需排查 `InternVLChatConfig` 的 `AutoModelForImageTextToText` 注册问题。

---

## 7. 如何运行多模态 KV cache benchmark

多模态 KV cache key 管理实验：3 种策略 × 7 类场景。纯 Python 模拟器，无需 PyTorch/GPU/transformers。

```bash
# 关键验收：策略 A false_hit（同文不同图）
python3 experiments/mm_kv_cache_management/benchmark_same_text_different_image.py

# 基础场景
python3 experiments/mm_kv_cache_management/benchmark_same_text_same_image.py
python3 experiments/mm_kv_cache_management/benchmark_same_image_different_question.py

# 进阶场景
python3 experiments/mm_kv_cache_management/benchmark_same_image_different_resize.py
python3 experiments/mm_kv_cache_management/benchmark_multi_image_order.py

# 一键运行全部 5 个 benchmark
for f in experiments/mm_kv_cache_management/benchmark_*.py; do python3 "$f"; done
```

核心发现：
- **策略 A（text-only cache key）在多模态下不安全**：`same_text_different_image` 场景 false_hits=1
- **策略 B（text + image_hash）是最小防御线**
- **策略 C（full multimodal metadata）提供完整保护**

结果产物：`experiments/mm_kv_cache_management/results/*.json` + `*.html`（共 10 个文件）。

---

## 8. 当前完成度

### Wave 1 - 目录骨架与文档入口：✅ 完成
- [x] 全目录创建
- [x] engine_inventory.md（18 模块静态审计，标识 2 阻塞 Bug + 2 未接线）
- [x] 显存预算与模型选型矩阵（4 模型）
- [x] README.md / TODO.md / reports/ 入口文件

### Wave 2 - 文本引擎审计 + Engine Patch：✅ 完成
- [x] 文本引擎静态审计（4 个审计脚本）
- [x] 5 个 Bug 修复（Attn 参数 / act_fn / head_dim / RoPE / rope_theta）
- [x] HF 对齐 → `verdict: IDENTICAL, max |diff|=8.2e-5, cos_sim=0.99999994`
- [x] Contiguous KV cache 接入 prefill/decode 路径（seq_len=1/8/64/512 全部通过）
- [x] docs/ 静态页面（01_已有引擎审计.html 至 02_paged_attention基础.html）

### Wave 3 - Paged Attention + 多模态 Token Pipeline：✅ 完成
- [x] PagedKVCache correctness-first 实现（BlockManager + BlockTable + gather）
- [x] contiguous vs paged 对齐通过（torch.allclose）
- [x] 5 模块教学型 token pipeline（2 种 visual token 模式 + 2 种序列布局）
- [x] docs/ 静态页面（03_vit和图像patch.html 至 05_qwen_vl多模态输入.html）

### Wave 4 - inputs_embeds + VLM Demo + VLM Reference：✅ 完成（含降级）
- [x] inputs_embeds 路径接入（text_parity: max|diff|=0，双输入冲突拒绝）
- [x] 最小 VLM demo（random projector 工程跑通）
- [x] 4 模型 VLM reference 对照脚本（cascade + 降级 smoke）
- [ ] ~~4 个 VLM reference 模型成功运行~~ - **降级**（缺 `accelerate` + 无 CUDA，4 个全失败）
- [x] 降级路径生效（tokenizer-only smoke 通过）
- [x] docs/ 静态页面（06_多模态prefill_decode.html 至 09_sglang多模态推理参考.html）

### Wave 5 - 多模态 KV Cache 管理 + 收尾：✅ 完成
- [x] 3 策略 × 7 场景 mm cache 模拟器（策略 A false_hit 关键验收）
- [x] 5 份 JSON + 5 份 HTML 结果报告
- [x] 最终成果说明（docs/10_最终成果说明.html + reports/final_report.md）
- [x] README.md 全部 9 章填实
- [x] TODO.md 全部 Wave 状态更新

---

## 9. 已知限制

1. **4 个 VLM reference 全部失败**：本机 macOS (MPS, 无 CUDA) 因缺 `accelerate` 包导致 `device_map="auto"` 失败。3 个 Qwen/SmolVLM 需 `pip install accelerate`，InternVL3.5-4B 额外需排查 config 注册。
2. **Local VLM demo 使用 random projector**：`run_minimal_vlm.py` 的 visual-to-text projector 是随机初始化的 `nn.Linear`，不携带任何预训练语义。不保证输出有意义。
3. **无 CUDA / 仅 CPU + MPS**：所有实验在 Apple Silicon MPS 上运行，性能数字不代表 NVIDIA GPU 表现。CUDA paged_attention kernel 未实现。
4. **Paged KV 使用 gather fallback**：decode 阶段通过 `torch.gather` 拼接 KV block，非 GPU-native kernel。正确性已验证，性能非优化。
5. **单序列推理**：minivLLM 无 scheduler、无 request queue、无批量调度，所有实验 batch_size=1。
6. **显存预算与模型选型为理论估算**：权重 + KV + 激活按 Paper-based 公式估算，未通过 `torch.cuda.memory_stats()` 在真实 GPU 上校准。
7. **mm cache 策略未接入真实推理循环**：3 策略设计在纯 Python 模拟器中验证，未接入 minivLLM 引擎的 KV cache 读/写。
8. **minivLLM 引擎不修改原则**：所有实验代码独立于 `minivLLM/`，引擎仅作为只读基础存在。对引擎的修改仅限 Wave 2 的 5 个 Bug 修复，任何超出此范围的引擎改动均不在本工作区范围。
9. **不引入外部框架**：全部文档为纯静态 HTML + CSS，无 npm / React / Vue / MkDocs / Sphinx 依赖。
10. **不在 multimodal/ 下创建新 git repo**：本目录是 monorepo 的一部分，使用仓库根目录的 git。

---

## 10. 工作量估算（重做一遍需要多久？）

> 实际执行：Atlas + 子 agent 并行加速，**5h 12m 23s** 完成 19/19 任务。
> 但这是 subagent 并行 + AI 自动调试的极限时间。**人工等价时间** 是另一个量级。

### 10.1 人工每周投入（按 6 周路线）

| Week | 任务 | 预计人工时间 |
|------|------|-------------|
| **Week 1** | 引擎审计 + 静态文档（任务 1-3） | **6-8h** |
| | - 读懂 minivLLM 代码 + 静态扫描 attention/KV/paged 状态 | |
| | - 写 inventory.md + 4 篇学习笔记 + 2 个 HTML | |
| | - 设计 paged KV 接口 + 显存预算与模型选型 | |
| **Week 2** | 文本引擎基础修复 + paged 路径（任务 4-6, 13） | **10-14h** |
| | - 修 4 个 BUG（Attn 构造、act_fn、head_dim、RoPE）直到 HF 对齐 | |
| | - 接通 contiguous KV cache + prefill/decode | |
| | - 最小 paged KV manager 实现 + 6 个测试 | |
| | - 学习知识库基础（13 篇论文笔记 + index.html） | |
| **Week 3** | mm token 管线 + inputs_embeds 路径（任务 7-8） | **6-8h** |
| | - 图像预处理 / patch embed / visual token / sequence builder | |
| | - 给 Qwen3 加 `inputs_embeds` 入口 + text_parity 验证 | |
| **Week 4** | 最小 VLM demo + reference 矩阵（任务 9-10, 14） | **8-12h** |
| | - 自有引擎多模态 prefill/decode 跑通 | |
| | - 4 个 reference VLM 对照（含降级路径） | |
| | - 中后期 docs + week_3/4/5 报告 | |
| **Week 5** | mm cache 实验 + 最终汇总（任务 11-12） | **6-8h** |
| | - 3 策略 × 7 case benchmark | |
| | - 最终报告 + README + 10_最终成果说明 | |
| **Week 6** | 回归 + F1-F4 验收（任务 15 + Final Wave） | **4-6h** |
| | - 6 条主回归 + 作用域守门 | |
| | - 4 个 final reviewer 并行跑 | |

### 10.2 总览

| 维度 | 数值 |
|------|------|
| **人工总计** | **40-56h**（6 周） |
| **平均每周** | **7-9h** |
| **最重的一周** | **Week 2（10-14h）**，4 个 BUG 修到 HF 对齐 + paged KV 实现 |
| **最轻的一周** | **Week 6（4-6h）**，收尾 + 验收 |

### 10.3 不同投入强度的时间表

| 投入强度 | 工期 |
|----------|------|
| 每天 1h | 6-8 周 |
| 每天 2h | 3-4 周 |
| 全职 8h/天 | 5-7 天 |
| AI 加速（subagent 并行） | **5h 12m**（本次实际） |

### 10.4 关键瓶颈

1. **Week 2 占比最大**（≈30% 总时间）：因为要修数值正确性 + 写新 paged 路径。建议不要在 Week 2 中断。
2. **Week 4 变数最大**：4 个 reference VLM 加载，硬件/依赖不确定。本计划里**全部失败**；如果用户机器有 CUDA，可能要预留 1-2 天调试。
3. **Wave 2/3/4 内部都是串行**（KV → paged → inputs_embeds → VLM demo），不能并行压缩。

### 10.5 是否需要 GPU？

按你本机是否 NVIDIA GPU 决定：

| 环境 | 预计人工时间 | 备注 |
|------|--------------|------|
| **有 NVIDIA GPU**（推荐） | **45-50h**，6 周每周 8h | reference VLM 可跑通，4 个模型可对比 |
| **无 GPU / Mac（本次实际路径）** | **35-40h** | reference VLM 必失败，可省下"调试 VLM 失败"的几小时 |

> **诚实建议**：本机 macOS (MPS, 无 CUDA) 实际节省了时间，因为 VLM reference 必失败，Wave 4 走降级路径反而快于预期。但生产环境请优先使用 NVIDIA GPU。

### 10.6 加速建议

- **Week 1 + Week 2 并行**：如果你提前确认了 minivLLM 引擎代码和 HF 权重，audit 和 base correctness 可以合并 1 周完成
- **论文笔记找人代写**：Wave 1/2 的 13 篇论文笔记（约 6-8h）可委托阅读速度更快的人
- **Week 5 的 cache 实验纯跑模拟**（无 GPU 依赖），可以离线/在最后一周密集做
- **AI 加速**：本次 Atlas + 多个 subagent 并行执行将总时间从 40-56h 压到 5h 12m。但 AI 修复引擎 Bug 的成功率不能 100% 保证，本项目里 5 个 BUG 有 4 个一次修对，1 个（head_dim）修了 2 次。

---

## 11. 补充资源

| 资源 | 路径 | 说明 |
|------|------|------|
| Markdown 文档镜像 | [`docs_md/`](docs_md/) | 与 `docs/` HTML 一致的 Markdown 版 |
| 论文 PDF | [`paper/`](paper/) | 12 篇核心论文 PDF 原件 |
| 论文中文笔记 | [`docs_md/papers/`](docs_md/papers/) | 12 篇论文的中文详解 |
| 学习笔记 | [`docs_md/notes/`](docs_md/notes/) | 9 篇源码与设计笔记 |
| 6 周周报 | [`reports/week_1.md`](reports/week_1.md) ~ `week_6.md` | Wave 1-5 的周报 |
| 最终报告 | [`reports/final_report.md`](reports/final_report.md) | 路线完成度、关键数字、已知限制 |
| 最终成果说明 | [`docs/10_最终成果说明.html`](docs/10_最终成果说明.html) 或 [`.md`](docs_md/10_最终成果说明.md) | 6 周路线的最终交付说明 |

---

> 所有实验命令执行前请先确认已安装 minivLLM 的 Python 依赖：`cd minivLLM && uv sync`。
