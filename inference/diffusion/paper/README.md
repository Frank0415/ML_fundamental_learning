# paper/ — 论文与模型卡 PDF 存档

> 本目录存储本项目阅读的 10+ 篇核心论文 PDF 和两份无 PDF 的模型/技术报告占位文件。

## 10 篇主论文 + 2 篇 bonus + 2 个占位

| 编号 | 文件 | 类型 | 显存友好 |
|------|------|------|----------|
| 00 | [00_flow_matching_original.pdf](00_flow_matching_original.pdf) | 理论（Flow Matching 原始） | — |
| 00 | [00_rectified_flow_original.pdf](00_rectified_flow_original.pdf) | 理论（Rectified Flow 原始） | — |
| 01 | [01_scaling_rectified_flow_transformers_sd3.pdf](01_scaling_rectified_flow_transformers_sd3.pdf) | 文生图（SD3/MMDiT） | Medium no-T5 ✅ |
| 02 | [02_flux_architecture.md](02_flux_architecture.md) | 文生图（FLUX，无 PDF） | schnell ✅ |
| 03 | [03_sana_efficient_high_resolution.pdf](03_sana_efficient_high_resolution.pdf) | 文生图（Sana） | 0.6B + int4 ✅ |
| 04 | [04_sora_technical_report.md](04_sora_technical_report.md) | 文生视频（Sora，无 PDF） | ❌ |
| 05 | [05_wan_video.pdf](05_wan_video.pdf) | 文生视频（Wan） | 1.3B 极限 ⚠️ |
| 06 | [06_hunyuanvideo.pdf](06_hunyuanvideo.pdf) | 文生视频（HunyuanVideo） | ❌ |
| 07 | [07_cogvideox.pdf](07_cogvideox.pdf) | 文生视频（CogVideoX） | 2B ✅ |
| 08 | [08_ltx_video.pdf](08_ltx_video.pdf) | 文生视频（LTX-Video） | 2B distilled ✅ |
| 09 | [09_pyramid_flow_matching.pdf](09_pyramid_flow_matching.pdf) | 文生视频（Pyramid Flow） | 理论 |
| 10 | [10_consistency_model.pdf](10_consistency_model.pdf) | 蒸馏（Consistency Model） | — |
| 10 | [10_lcm_latent_consistency.pdf](10_lcm_latent_consistency.pdf) | 蒸馏（LCM） | — |

总计：12 PDF + 2 MD 占位（FLUX + Sora），约 250 MB。

## 无 PDF 的资料说明

### FLUX (`02_flux_architecture.md`)

Black Forest Labs 未发表 arXiv 论文。本文件从 [https://github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux) 拉取的 README 简化版，作为 FLUX 架构理解的参考。完整信息需阅读：

- 官方 blog：https://blackforestlabs.ai/announcing-black-forest-labs/
- diffusers 源码：`diffusers/pipelines/flux/pipeline_flux.py`
- HF discussions：https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions

### Sora (`04_sora_technical_report.md`)

OpenAI 未发表 arXiv 论文，仅有技术说明网页。本文件是占位说明，指向：

- 官方技术说明：https://openai.com/index/video-generation-models-as-world-simulators/
- 第三方复现：https://github.com/hpcaitech/Open-Sora

如需本地存档，使用 `curl` 或 `webfetch` 拉取 HTML。

## 中文解读位置

每篇论文的中文解读位于 `../docs_md/paper_*.md`：

- `paper_00_flow_matching_original_中文解读.md` — Flow Matching 原始
- `paper_00_rectified_flow_original_中文解读.md` — Rectified Flow 原始
- `paper_01_sd3_中文解读.md` — SD3
- `paper_02_flux_中文解读.md` — FLUX
- `paper_03_sana_中文解读.md` — Sana
- `paper_04_sora_中文解读.md` — Sora
- `paper_05_wan_中文解读.md` — Wan
- `paper_06_hunyuanvideo_中文解读.md` — HunyuanVideo
- `paper_07_cogvideox_中文解读.md` — CogVideoX
- `paper_08_ltx_video_中文解读.md` — LTX-Video
- `paper_09_pyramid_flow_中文解读.md` — Pyramid Flow Matching
- `paper_10_consistency_distillation_中文解读.md` — Consistency / LCM

## 资源档位总览

| 类别 | 适合中等显存配置 | 极限 | 不适合 |
|------|----------|------|--------|
| **文生图** | SD3 Medium no-T5 / FLUX schnell / Sana 0.6B int4 / Sana 1.6B int4 | SD3 Medium with T5 / FLUX dev | SD3 Large |
| **文生视频** | LTX-Video 2B distilled / CogVideoX-2B | Wan2.1-1.3B | HunyuanVideo / SD3 Large video |
| **蒸馏** | FLUX schnell (4步) / Sana-Sprint (2步) / SD-Turbo / LCM | — | — |

**核心洞察**：受限显存场景下，**few-step 蒸馏模型**比"小参数 + 多步"模型更有效。FLUX.1-schnell (4 步) 和 LTX-Video 2B distilled (4-8 步) 是 中等显存配置 用户最舒适的选择。
