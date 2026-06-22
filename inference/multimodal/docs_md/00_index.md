# minivLLM 多模态推理实验工作区 — 文档导航

> 这是 `docs/` 静态 HTML 文档的 Markdown 镜像版本。两个版本内容完全一致：
> - **HTML 版**（`docs/*.html`）：纯静态网页，可在浏览器直接打开
> - **Markdown 版**（`docs_md/*.md`）：适合在编辑器、GitHub 预览、grep 搜索
>
> 推荐阅读路径：**先看 [01 已有引擎审计](01_已有引擎审计.md)**，再按章节顺序向下。

## 项目目标

从纯文本推理引擎出发，逐步构建一个可运行的最小多模态（VLM）推理流水线。最终在 minivLLM 上运行 Qwen3-VL-4B 或同类开源 VLM。本文档区记录引擎审计、模块实现、论文笔记与实验日志。

## 文档目录

以下章节按实现顺序编排。所有页面均已发布。

| # | 标题 | 说明 |
|---|------|------|
| 01 | [已有引擎审计](01_已有引擎审计.md) | minivLLM 文本引擎完整静态审计：模块状态、阻塞 Bug、已实现能力与缺口。 |
| 02 | [Paged Attention 基础](02_paged_attention基础.md) | PagedAttention 原理：块表管理、逻辑-物理映射、碎片率分析，以及 minivLLM 当前 contiguous KV cache 的差距。 |
| 03 | [ViT 与图像 Patch 编码](03_vit和图像patch.md) | Vision Transformer 如何将图像切割为 patch token、patch embedding 与位置编码。 |
| 04 | [CLIP 与图文对齐](04_clip和图文对齐.md) | CLIP 对比学习机制：双塔架构、InfoNCE 损失、CLIP 视觉编码器在 VLM 中的角色。 |
| 05 | [Qwen-VL 多模态输入](05_qwen_vl多模态输入.md) | Qwen-VL 系列输入格式：特殊 token、visual token 拼接规则、image_grid_thw 与 token 预算。 |
| 06 | [多模态 Prefill 与 Decode](06_多模态prefill_decode.md) | 多模态 prefill 中 visual token 一次性写入 KV cache，decode 阶段追加文本 token 的差异与显存分析。 |
| 07 | [多模态 KV Cache 管理](07_多模态kv_cache管理.md) | 为什么 multimodal-aware cache key 必要：text-only prefix 在不同图像下 KV 不同，需分阶段 cache 策略。 |
| 08 | [vLLM 多模态推理参考](08_vllm多模态推理参考.md) | vLLM 多模态支持：PagedAttention 在 VLM 上的应用、自动前缀缓存、参考实验中的环境阻塞分析。 |
| 09 | [SGLang 多模态推理参考](09_sglang多模态推理参考.md) | SGLang RadixAttention 的多模态扩展、Radix Tree 与文本-视觉分离、受限显存场景下的对比分析。 |
| 10 | [最终成果说明](10_最终成果说明.md) | 项目最终交付物：6 周路线总结、关键验证数字（HF IDENTICAL / KV 对齐 / mm cache false_hit）、已知限制与后续建议。 |

## 学习资料

- 论文笔记（中文）存放于 [`../learning/papers/`](../learning/papers/) 目录，已完成 12 篇核心论文的精华笔记，覆盖 Transformer 至 SGLang RadixAttention。完整论文清单见 [00 论文清单](../learning/papers/00_论文清单.md)。
- 论文 PDF 原件存放于 [`../paper/`](../paper/) 目录，与笔记同名（`01_attention_is_all_you_need.pdf` 等）。**PDF 不入 git**，见 `multimodal/.gitignore`。
- 学习笔记（engine audit → SGLang）存放于 [`../learning/notes/`](../learning/notes/) 目录，共 9 篇。

### 论文笔记

| # | 标题 | 笔记 | PDF |
|---|------|------|-----|
| 01 | Attention Is All You Need | [notes](../learning/papers/01_attention_is_all_you_need.md) | [pdf](../paper/01_attention_is_all_you_need.pdf) |
| 02 | ViT | [notes](../learning/papers/02_vit.md) | [pdf](../paper/02_vit.pdf) |
| 03 | CLIP | [notes](../learning/papers/03_clip.md) | [pdf](../paper/03_clip.pdf) |
| 04 | Flamingo | [notes](../learning/papers/04_flamingo.md) | [pdf](../paper/04_flamingo.pdf) |
| 05 | BLIP-2 | [notes](../learning/papers/05_blip2.md) | [pdf](../paper/05_blip2.pdf) |
| 06 | LLaVA | [notes](../learning/papers/06_llava.md) | [pdf](../paper/06_llava.pdf) |
| 07 | Qwen-VL | [notes](../learning/papers/07_qwen_vl.md) | [pdf](../paper/07_qwen_vl.pdf) |
| 08 | Qwen2-VL | [notes](../learning/papers/08_qwen2_vl.md) | [pdf](../paper/08_qwen2_vl.pdf) |
| 09 | Qwen2.5-VL | [notes](../learning/papers/09_qwen2_5_vl.md) | [pdf](../paper/09_qwen2_5_vl.pdf) |
| 10 | Qwen3-VL | [notes](../learning/papers/10_qwen3_vl.md) | [pdf](../paper/10_qwen3_vl.pdf) |
| 11 | PagedAttention (vLLM) | [notes](../learning/papers/11_pagedattention_vllm.md) | [pdf](../paper/11_pagedattention_vllm.pdf) |
| 12 | SGLang RadixAttention | [notes](../learning/papers/12_sglang_radixattention.md) | [pdf](../paper/12_sglang_radixattention.pdf) |

### 学习笔记（9 篇）

| # | 标题 | 笔记 |
|---|------|------|
| 01 | 已有引擎结构 | [notes](../learning/notes/01_已有引擎结构.md) |
| 02 | Attention 实现审计 | [notes](../learning/notes/02_attention实现审计.md) |
| 03 | KV Cache 实现审计 | [notes](../learning/notes/03_kv_cache实现审计.md) |
| 04 | Paged Attention 实现进度 | [notes](../learning/notes/04_paged_attention实现进度.md) |
| 05 | 多模态输入设计 | [notes](../learning/notes/05_多模态输入设计.md) |
| 06 | Vision Encoder 接入方案 | [notes](../learning/notes/06_vision_encoder接入方案.md) |
| 08 | 显存预算与模型选型 | [notes](../learning/notes/08_显存预算与模型选型.md) |
| 09 | vLLM 源码参考 | [notes](../learning/notes/09_vllm源码参考.md) |
| 10 | SGLang 源码参考 | [notes](../learning/notes/10_sglang源码参考.md) |

> 说明：原计划 10 个学习笔记中"07"编号被 mm cache 实验的"video frame_sampling"用例占用语义，原目录下没有 07 文件，docs_md 保留同样的编号空白。

## 周报

| # | 标题 | 链接 |
|---|------|------|
| Week 1 | 启动审计 | [week_1](../../reports/week_1.md) |
| Week 2 | Engine Patch + KV Cache | [week_2](../../reports/week_2.md) |
| Week 3 | 多模态 Token Pipeline | [week_3](../../reports/week_3.md) |
| Week 4 | inputs_embeds 路径 + VLM Demo | [week_4](../../reports/week_4.md) |
| Week 5 | 多模态 KV Cache | [week_5](../../reports/week_5.md) |
| Week 6 | VLM Reference 矩阵 | [week_6](../../reports/week_6.md) |

---

> minivLLM 多模态推理实验工作区 · 文档导航 · 更新时间 2026-06-07
> Markdown 版本与 `docs/index.html` 内容完全一致，仅做格式转换。
