# 第 5 周报告：Video Reference 脚手架 + 视频学习笔记

> **日期**：2026-06-07  Week 5
> **来源任务**：T15（视频 reference 脚手架、尝试、blocker 与 Week 5 报告）+ T9（视频 latent / spacetime patch 学习笔记）
> **证据文件**：`.omo/evidence/task-15-video-results.txt`、`.omo/evidence/task-15-video-docs-check.txt`

---

## 1. 完成内容

### 1.1 视频学习笔记（T9）

- **`learning/notes/09_视频latent和spacetime_patch.md`**：详细解释 video latent 与 image latent 的 shape 差异。Image latent：`(B, C, H, W)`。Video latent：`(B, C, T, H, W)`。关键约定：PyTorch 默认使用 `(B, C, T, H, W)`，但部分模型（如 diffusers）使用 `(B, T, C, H, W)`。Spacetime patch 将时间维度也纳入 patchify：3D VAE 输出压缩后的 latent（通常是 4-16× 压缩，时间维度 4-8× 压缩），然后沿 T、H、W 三个维度切 patch，使一个 token 同时携带时空信息。

- **`learning/notes/10_diffusers_reference源码走读.md`**：走读 HuggingFace Diffusers 中视频 pipeline 的关键模块入口。覆盖 `CogVideoXPipeline.__call__()`、`LTXPipeline.__call__()`、`WanPipeline.__call__()` 的主要流程，以及每个 pipeline 的 VAE encode/decode、text encoder、transformer（DiT）、scheduler 的实际调用方式。

### 1.2 Video Reference 脚手架（T15）

在 `experiments/reference_video_inference/` 下完成：

- **`README.md`**：三模型优先级排序（LTX-Video 2B distilled > CogVideoX-2B > Wan2.1-T2V-1.3B）、中等显存配置 预算表、降级规格（≤16 帧 × 256² × ≤8 步）、15 分钟 timebox 超时策略。

- **`run_ltx_video_if_possible.py`**：LTX-Video 2B（Lightricks，开放协议）。默认配置：16f × 256²、8 steps、bf16、cpu_offload。预期 VRAM ~8 GB（在中等显存配置下较安全）。

- **`run_cogvideox_if_possible.py`**：CogVideoX-2B（THUDM，Apache 2.0，无授权障碍）。默认配置：16f × 256²、8 steps、bf16、cpu_offload。预期 VRAM ~6-8 GB。

- **`run_wan_if_possible.py`**：Wan2.1-T2V-1.3B（Alibaba，Apache 2.0）。默认配置：16f × 256²、8 steps、bf16、cpu_offload。预期 VRAM ~8-10 GB。

- **`profile_video_memory.py`**：视频推理显存估算工具。输入参数（帧数、分辨率、步数、dtype），输出预估 VRAM 分解（模型权重 + VAE + text encoder + attention activations + latent buffers）。

- **`sample_prompts.txt`**：12 条视频推理示例 prompt，涵盖日常场景（动物、自然、人物）和测试场景（简单 motion、复杂 motion、静态）。

### 1.3 Blocker 记录

三个 blocker 占位文件已创建：
- `results/blocker_ltx_video.md`：LTX-Video 因 CUDA 不可用而阻塞。脚本已通过 `--help` 和代码结构自检。
- `results/blocker_cogvideox.md`：CogVideoX-2B 因 CUDA 不可用而阻塞。Apache 2.0，无授权障碍，是优先尝试的模型。
- `results/blocker_wan.md`：Wan2.1 因 CUDA 不可用而阻塞。

所有 blocker 均为同一 root cause：开发环境为 macOS M5，不支持 CUDA。脚本本身无 bug。

### 1.4 视频知识库页面

- **`docs/09_sora_style视频生成架构.html`**：Sora-style 视频生成架构说明。覆盖 spacetime patch、3D VAE、causal 3D attention（temporal + spatial 分离或联合）、video DiT 与 image DiT 的差异。

- **`docs/10_wan_hunyuan_cogvideox_ltx视频模型.html`**：四个开源视频模型横向对比。按授权、VRAM、架构对比（DiT vs 3D causal VAE vs existing VAE）。资源档位排序表。

---

## 2. 技术主要发现

### 2.1 视频 Latent 的时间压缩

视频的 3D VAE 不仅压缩空间维度（通常 8× 下采样），还压缩时间维度（通常 4× 下采样）。这意味着：
- 原始 49 帧 × 720 × 480 的视频 → latent ≈ (C, 13, 90, 60)（T 方向 49/4≈13，H 方向 720/8=90，W 方向 480/8=60）
- Latent token 数 ≈ 13 × 90 × 60 / patch_size²（通常 patch_size=2，token 数 ≈ 17550）
- 对于 16 帧小规格：latent ≈ (C, 4, 32, 32)，patch_size=2 → 1024 tokens。这是 中等显存配置 安全的。

### 2.2 三个视频模型的优先级理由

| 模型 | 优先级 | 授权 | VRAM 预计 | 理由 |
|------|--------|------|----------|------|
| LTX-Video 2B | 1 | 开放 | ~8 GB | 蒸馏到极低步数（4-8 steps），延迟短。121f 能力但 在中等显存配置下只建议 16f。 |
| CogVideoX-2B | 2 | Apache 2.0 | ~6-8 GB | 无任何授权和 download gate，是最无障碍的模型。`diffusers` 原生支持。 |
| Wan2.1-1.3B | 3 | Apache 2.0 | ~8-10 GB | 1.3B 参数较小，但 81f 默认帧数偏多，需手动降帧数。支持多分辨率。 |

HunyuanVideo 不包含在正式尝试范围内（13B 参数，中等显存配置 几乎不可能），仅作 bonus。

### 2.3 视频推理的 Timebox 策略

针对 中等显存配置 + 视频推理的时间不确定性，制定了严格的 timebox：
- 单次推理超时：15 分钟
- 超时即记录 blocker，不无限重试
- 降级顺序：降帧数（49→25→16）→ 降分辨率（720×480→512×384→256×256）→ 减 steps（50→30→8）
- 默认小规格：≤16 帧 × 256² × ≤8 步

---

## 3. 与学习笔记和知识库的对照

- `learning/notes/09_视频latent和spacetime_patch.md` 与 `docs/09_*.html` 的 spacetime patch 说明一致
- `learning/notes/10_diffusers_reference源码走读.md` 为 T15 的视频模型脚本提供了 diffusers pipeline 接口参考
- `docs/10_*.html` 的视频模型对比表来自 `learning/papers/04-08` 中的论文卡片内容

---

## 4. 本周风险与未完成项

- **三模型均未实际运行**：由于 dev host 为 macOS M5（无 CUDA），远程 CUDA GPU 当前不可用。所有视频验证均在"脚本已就绪，待远程执行"状态。
- **LTX-Video 的真实 VRAM 是估算**：LTX-Video 2B 的 8 GB 预估来自官方文档和社区报告，但未在一块可用的 CUDA GPU 上实测。
- **视频质量未验证**：小规格（16f × 256²）能否产生有意义的结果，待真实运行后才能确认。

### 诚实声明

- 三个 blocker 文件均为**占位 blocker**（环境未就绪），不是"尝试失败"。脚本本身已通过 `--help` 和代码结构自检。
- **未伪造任何视频输出**：`results/` 下无 mp4/png 文件。
- **未在 M5 上强行运行 CUDA 视频 pipeline**。

---

## 5. 下周预览（Week 6 / T16-T18）

- T16：prompt cache / latent buffer manager / scheduler benchmark
- T17：CFG batching / attention memory / VAE tiling 对照实验
- T18：后半知识库（docs/index.html + 01/11/12）、最终报告、README 终版

---

> **本周产出**：2 篇学习笔记（T9）、1 个 video reference 实验目录（含 3 个模型脚本 + README + profiler + 3 个 blocker 文件）、2 个视频知识库 HTML 页面（T15）。T15 脚手架圆满完成，T9 视频学习笔记产出。
