# docs_md · 多模态推理文档镜像（Markdown 版）

本目录是 `docs/` 静态 HTML 文档的 **Markdown 镜像**。HTML 版与 Markdown 版内容完全一致，仅格式不同：

| 版本 | 路径 | 用途 |
|------|------|------|
| HTML | [`../docs/`](../docs/) | 浏览器直开、纯静态网页、零依赖 |
| Markdown | `docs_md/`（本目录） | 编辑器预览、GitHub 渲染、grep/全文搜索、diff review |

## 目录结构

```
docs_md/
├── README.md                          # 本文件
├── 00_index.md                        # 文档导航（与 docs/index.html 等价）
├── 01_已有引擎审计.md                 # 10 篇技术文档
├── 02_paged_attention基础.md
├── 03_vit和图像patch.md
├── 04_clip和图文对齐.md
├── 05_qwen_vl多模态输入.md
├── 06_多模态prefill_decode.md
├── 07_多模态kv_cache管理.md
├── 08_vllm多模态推理参考.md
├── 09_sglang多模态推理参考.md
└── 10_最终成果说明.md
```

> **说明**：`docs_md/` 只镜像 10 篇技术文档。论文笔记在 [`../learning/papers/`](../learning/papers/)，学习笔记在 [`../learning/notes/`](../learning/notes/)，避免单点真相重复。

## 相关目录

- [`../docs/`](../docs/) — 同一内容的 HTML 镜像版
- [`../paper/`](../paper/) — 12 篇论文 PDF 原件（按 `01_xxx.pdf` ~ `12_xxx.pdf` 命名，与 `learning/papers/` 笔记一一对应）。**PDF 不入 git**，见 `multimodal/.gitignore`。
- [`../learning/papers/`](../learning/papers/) — 12 篇论文的中文笔记
- [`../learning/notes/`](../learning/notes/) — 9 篇学习笔记（engine audit → SGLang）
- [`../reports/`](../reports/) — 6 周周报与最终报告

## 如何阅读

**推荐顺序**：

1. 先看 [00_index.md](00_index.md) 了解全貌
2. 从 [01_已有引擎审计.md](01_已有引擎审计.md) 开始，按章节顺序向下
3. 章节中涉及的论文，查 [`../learning/papers/`](../learning/papers/) 下的中文笔记 + [`../paper/`](../paper/) 下的 PDF
4. 章节中涉及的源码细节，查 [`../learning/notes/`](../learning/notes/) 下的学习笔记

## 维护说明

- 任何对 `docs/*.html` 的修改都应同步到 `docs_md/*.md`，反之亦然
- 论文 PDF 不会自动更新。如需更新，跑：
  ```bash
  cd paper && curl -sL -o <file>.pdf "https://arxiv.org/pdf/<arxiv_id>" --max-time 60
  ```
- 论文笔记与学习笔记的**唯一权威位置**是 `learning/papers/` 与 `learning/notes/`；`docs_md/` 不再保存副本

---

minivLLM 多模态推理实验工作区 · Markdown 文档镜像 · 更新时间 2026-06-07
