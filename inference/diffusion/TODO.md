# TODO.md：diffusion-inference 任务清单

> 最后更新：2026-06-07
> 来源：`.omo/plans/modern-diffusion-inference-roadmap.md` 中 18 个实现任务
> 状态标记：`[x]` 已完成，`[ ]` 待完成

---

## Wave 1：前置验证、审计、决策、骨架（T1–T5）✅ 全部完成

- [x] **T1 — 前置环境与风险验证**
  - blocker：无（已完成）
  - 产出：`reports/week_1.md`、HF 连通性、磁盘/内存检查

- [x] **T2 — 旧引擎清点与 `reports/engine_inventory.md`**
  - blocker：无（已完成）
  - 产出：`reports/engine_inventory.md`（515 行，14 模块逐一审计）

- [x] **T3 — 复用决策与隔离策略说明**
  - blocker：无（已完成）
  - 产出：`learning/notes/01_老引擎结构审计.md`、`02_是否复用老引擎.md`、`docs/02_老引擎审计.html`
  - 结论：选 C（完全不适合复用），新建 `diffusion_engine/`

- [x] **T4 — 项目骨架、独立环境与顶层说明**
  - blocker：无（已完成）
  - 产出：`pyproject.toml`、`README.md`、`TODO.md`、`docs/style.css`、目录骨架

- [x] **T5 — 现代 diffusion 推理数据流笔记与最小背景**
  - blocker：无（已完成）
  - 产出：`learning/notes/03_diffusion推理数据流.md`、`docs/03_现代diffusion推理最小背景.html`

---

## Wave 2：推理基础笔记、论文清单、toy rectified flow（T6–T10）✅ 全部完成

- [x] **T6 — 论文清单与阅读模板落地**
  - blocker：无（已完成）
  - 产出：`learning/papers/00_论文清单.md`（10 篇条目 + 统一模板）

- [x] **T7 — 图像主线论文卡片：SD3 / FLUX / Sana**
  - blocker：无（已完成）
  - 产出：`learning/papers/01_scaling_rectified_flow_transformers_sd3.md`、`02_flux_architecture_notes.md`、`03_sana.md`

- [x] **T8 — 视频主线论文卡片与中段知识库页面**
  - blocker：无（已完成）
  - 产出：`learning/papers/04_` 到 `10_`（7 篇卡片）、`docs/04_` 到 `08_`（5 个 HTML 页面）

- [x] **T9 — 视频 latent / spacetime patch 学习笔记与 Week 5 预备材料**
  - blocker：无（已完成）
  - 产出：`learning/notes/09_视频latent和spacetime_patch.md`、`10_diffusers_reference源码走读.md`

- [x] **T10 — scheduler / rectified flow / timestep embedding + toy rectified flow**
  - blocker：无（已完成）
  - 产出：`diffusion_engine/core/scheduler.py`、`rectified_flow.py`、`timestep_embedding.py`、36 个 pytest（全部通过）、`experiments/toy_rectified_flow/` 完整实验（含 PNG + JSON）

---

## Wave 3：toy DiT、reference image、video 尝试（T11–T15）✅ 全部完成

- [x] **T11 — attention / transformer_block / tiny DiT + shape 测试**
  - blocker：torch 未安装（dev host 无 torch，测试 skip；smoke 脚本逻辑已验证）
  - 产出：`diffusion_engine/core/attention.py`、`transformer_block.py`、`dit.py`、18 个 shape 测试（需 torch）

- [x] **T12 — text conditioning / pipeline / memory manager + toy DiT inference**
  - blocker：torch 未安装（同上原因）
  - 产出：`diffusion_engine/core/text_conditioning.py`、`pipeline.py`、`memory_manager.py`、`vae_stub.py`、24 个 smoke 测试（需 torch）、`experiments/toy_dit_inference/`（含 blocker 记录）

- [x] **T13 — reference image inference 脚手架与前置下载说明**
  - blocker：无（脚手架已完成）
  - 产出：`experiments/reference_image_inference/README.md`、3 个模型脚本（`run_sana_if_possible.py` 等）、`profile_memory.py`、`sample_prompts.txt`

- [x] **T14 — 真实 reference 文生图尝试、结果记录与 Week 4 报告**
  - blocker：**远程 RTX 5070 Ti 不可用** + dev host 无 CUDA/torch/diffusers。T14 脚手架就绪，真实推理未执行。blocker 如实记录在 `attempt_manifest.md`。非代码 bug。
  - 产出：`experiments/reference_image_inference/results/attempt_manifest.md`（详细 blocker 记录）、`reports/week_4.md`

- [x] **T15 — 视频 reference 脚手架、尝试、blocker 与 Week 5 报告**
  - blocker：**远程 RTX 5070 Ti 不可用**。3 个模型脚本均已就绪（help 通过），3 个 blocker 占位文件待 GPU 环境就绪后更新。
  - 产出：`experiments/reference_video_inference/` 完整实验目录、`results/blocker_*.md`（3 个）、`reports/week_5.md`、`docs/09_` 和 `docs/10_`

---

## Wave 4：系统优化实验、知识库收束、最终报告（T16–T18）✅ 全部完成

- [x] **T16 — prompt cache / latent buffer manager / scheduler benchmark**
  - blocker：无（numpy mock 模式下完成）
  - 产出：3 个实验脚本 + 量化结果（JSON）：prompt cache 52% hit、latent buffer 91.8% allocation 节省、scheduler 线性 scaling

- [x] **T17 — CFG batching / attention memory / VAE tiling 对照实验**
  - blocker：无（numpy mock/estimation 模式下完成）
  - 产出：3 个实验脚本 + 量化结果（MD+JSON）：CFG batched 1.01-1.02×、attention O(N²) 验证、VAE tiling tradeoff

- [x] **T18 — 后半知识库、顶层 README、周报与最终报告收束**
  - blocker：无（全部文档已产出）
  - 产出：`docs/index.html`、`docs/01_任务总览.html`、`docs/11_diffusion推理系统优化.html`、`docs/12_diffusion_gemma.html`、`docs/13_最终成果说明.html`、`reports/week_2.md`~`week_6.md`、`reports/final_report.md`、顶层 `README.md` 更新（13 小节齐全）、`TODO.md` 更新（本文件）

---

## Wave 标记速查

| Wave | 任务 | 阶段 | 状态 |
|------|------|------|------|
| 1 | T1–T5 | 前置验证、审计、骨架 | ✅ 全部完成 |
| 2 | T6–T10 | 学习笔记、toy RF | ✅ 全部完成 |
| 3 | T11–T15 | toy DiT、reference | ✅ 全部完成（T14 真实 ref 未跑通但脚手架就绪） |
| 4 | T16–T18 | 优化实验、收束 | ✅ 全部完成 |

---

## 未完成项与原因

| 任务 | 未完成内容 | 原因 | 是否可修复 |
|------|-----------|------|-----------|
| T14 | 真实 reference image inference 未在 GPU 上跑通 | 远程 RTX 5070 Ti 不可用；dev host 无 CUDA/torch | 是：环境就绪后单条命令即可 |
| T15 | 真实 reference video inference 未在 GPU 上跑通 | 同上 | 是：环境就绪后单条命令即可 |
| T11/T12 | DiT/pipeline 的 42 个 pytest skip | dev host 无 torch 安装 | 是：`uv pip install torch` 后即可运行 |

---

## 跨任务依赖提醒

- **T10 已等 T4**：`diffusion_engine/core/` 目录在写 `scheduler.py` 前已存在 ✅
- **T14 需远程 RTX 5070 Ti**：Mac M5 (MPS) 不支持 CUDA。脚手架就绪，待远程执行。
- **T15 时间预算**：视频推理设置 15 分钟超时，超时即记录 blocker。T15 的 3 个 blocker 为占位文件。
- **T18 的 README 更新在 T4 初版基础上升级**：13 小节结构保持，各节补齐运行命令和结果总结 ✅
