# docs_md/ — Markdown 知识库

> `docs/` 目录下 HTML 知识库的 Markdown 版本。所有内容、链接、表格、shape 约定完全对应 HTML 版本。便于：
> - 在终端中直接 `cat` 或 `grep` 阅读
> - 用任意 Markdown 编辑器（VSCode、Obsidian、Typora）打开
> - 搜索特定章节（grep + heading）
> - 版本管理（git diff 友好）

## 13 个页面 + 1 篇 README 索引

| 编号 | 文件 | 主题 |
|------|------|------|
| — | [index.md](index.md) | 知识库首页导航 |
| 01 | [01_任务总览.md](01_任务总览.md) | 项目目标、路线选择 |
| 02 | [02_老引擎审计.md](02_老引擎审计.md) | minivLLM 复用决策 C |
| 03 | [03_现代diffusion推理最小背景.md](03_现代diffusion推理最小背景.md) | 推理数据流 9 要点 |
| 04 | [04_rectified_flow和flow_matching.md](04_rectified_flow和flow_matching.md) | Rectified Flow 理论 |
| 05 | [05_diffusion_transformer架构.md](05_diffusion_transformer架构.md) | DiT 架构 |
| 06 | [06_stable_diffusion_3_mmdit.md](06_stable_diffusion_3_mmdit.md) | SD3 MMDiT 详解 |
| 07 | [07_flux和现代文生图推理.md](07_flux和现代文生图推理.md) | FLUX 推理系统 |
| 08 | [08_sana高效高分辨率生成.md](08_sana高效高分辨率生成.md) | Sana 高效架构 |
| 09 | [09_sora_style视频生成架构.md](09_sora_style视频生成架构.md) | Sora 视频架构 |
| 10 | [10_wan_hunyuan_cogvideox_ltx视频模型.md](10_wan_hunyuan_cogvideox_ltx视频模型.md) | 视频模型对比 |
| 11 | [11_diffusion推理系统优化.md](11_diffusion推理系统优化.md) | 系统优化 + LLM KV 差异 |
| 12 | [12_diffusion_gemma.md](12_diffusion_gemma.md) | DiffusionGemma 架构与推理 |
| 13 | [13_最终成果说明.md](13_最终成果说明.md) | 成果汇总、运行命令 |

## 论文阅读卡片（位于 `../learning/papers/`）

10 篇必读论文/模型卡的中文阅读卡片位于 [`../learning/papers/`](../learning/papers/)，是已纳入版本控制的**唯一**正式论文学习资料。`docs_md/` 不再重复存放论文解读。

| 论文 | 卡片 |
|------|------|
| 论文清单与模板 | [../learning/papers/00_论文清单.md](../learning/papers/00_论文清单.md) |
| SD3 / MMDiT | [../learning/papers/01_scaling_rectified_flow_transformers_sd3.md](../learning/papers/01_scaling_rectified_flow_transformers_sd3.md) |
| FLUX | [../learning/papers/02_flux_architecture_notes.md](../learning/papers/02_flux_architecture_notes.md) |
| Sana | [../learning/papers/03_sana.md](../learning/papers/03_sana.md) |
| Sora | [../learning/papers/04_sora_style_video_generation.md](../learning/papers/04_sora_style_video_generation.md) |
| Wan | [../learning/papers/05_wan_video.md](../learning/papers/05_wan_video.md) |
| HunyuanVideo | [../learning/papers/06_hunyuanvideo.md](../learning/papers/06_hunyuanvideo.md) |
| CogVideoX | [../learning/papers/07_cogvideox.md](../learning/papers/07_cogvideox.md) |
| LTX-Video | [../learning/papers/08_ltx_video.md](../learning/papers/08_ltx_video.md) |
| Pyramid Flow | [../learning/papers/09_pyramid_flow_matching.md](../learning/papers/09_pyramid_flow_matching.md) |
| Consistency / LCM | [../learning/papers/10_consistency_distillation_and_fast_sampling.md](../learning/papers/10_consistency_distillation_and_fast_sampling.md) |

## HTML ↔ MD 对应关系

每个 MD 文件的内容与 HTML 版本完全对应：
- 表格 → Markdown 表格
- `<pre><code>` 块 → `\`\`\`python 围栏代码块
- `<h1>` / `<h2>` / `<h3>` → `#` / `##` / `###`
- `<a href="...">` → `[...](...)`
- HTML 标签 → 纯文本

## 阅读路径推荐

### 第一次来（30 分钟快速了解项目）

1. `01_任务总览.md` — 项目目标
2. `03_现代diffusion推理最小背景.md` — 数据流
3. `13_最终成果说明.md` — 成果汇总
4. `12_diffusion_gemma.md` — DiffusionGemma

### 深入技术（2-3 小时）

按顺序：
- 04 Rectified Flow
- 05 DiT 架构
- 06 SD3 / MMDiT
- 11 系统优化

### 论文精读（按需）

`../learning/papers/01-10_*.md` 是每篇论文的"中文摘要 + 中等显存配置 评估 + 对 diffusion_engine 的启发"，已纳入版本控制。建议先读这些卡片，再决定是否精读原 PDF（PDF 在 `../paper/` 下，**不**纳入版本控制，需要从 arXiv 自行下载）。

## 与 docs/HTML 的差异

| 维度 | docs/HTML | docs_md/MD |
|------|----------|-----------|
| 浏览器阅读 | ✅ file:// 直接打开 | ❌ 需 Markdown 编辑器 |
| 终端阅读 | ❌ 标签噪音 | ✅ cat / grep 友好 |
| 版本管理 diff | ❌ HTML 标签噪音 | ✅ 纯文本 diff |
| 搜索定位 | ⚠️ 需要解析 HTML | ✅ grep + heading 即可 |
| 美观度 | ✅ 主题样式 | ⚠️ 取决于渲染器 |
| 移动端阅读 | ✅ 浏览器友好 | ⚠️ 需 App |

**建议**：日常阅读用 HTML（`docs/`），代码开发/搜索用 MD（`docs_md/`）。
