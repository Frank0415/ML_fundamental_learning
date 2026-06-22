# Reference Video Inference 实验

> **优先级**：P1（视频是小规格尝试，失败也接受，但必须留 attempt trace 或 blocker）
> **负责任务**：T15 — 视频 reference 脚手架、尝试、blocker 与 Week 5 报告
> **执行环境**：远程 CUDA GPU（中等显存配置），Mac M5 不支持 CUDA
> **最后更新**：2026-06-07

---

## 1. 实验目的与定位

### 1.1 核心定位

本实验属于 **bonus 性质**。项目核心交付是文生图 reference inference（T14），视频方向不强求成功。但本项目要求**系统的现代 diffusion 推理理解**，因此视频模型的以下部分必须覆盖：

1. **架构理解**：深入理解视频 DiT 与图像 DiT 的结构差异（temporal attention、spacetime patch、视频 VAE）。
2. **显存策略**：验证 受限显存配置下视频推理的现实可行性，测试 cpu_offload / vae_tiling / frame_chunk 等策略。
3. **Blocker 文化**：如实记录每一个失败点，不美化、不伪造、不跳过。

**一句话总结**：只要留下了完整的 attempt trace 或 blocker 记录，T15 就是成功交付。

### 1.2 对后续任务的依赖关系

| 下游任务 | 依赖方式 |
|---------|---------|
| T18（最终报告） | 无论成功或失败，T15 的实验记录是最终报告中"视频推理 资源档位"章节的核心输入 |
| T16（系统优化） | 视频 latent buffer 的 memory 估算对 T16 的 latent_buffer_manager 有交叉参考价值 |
| 不阻塞 | 任何 diffusion_engine 核心开发 |

---

## 2. 模型优先级与决策树

### 2.1 优先级排序（按 资源档位从高到低）

| 优先级 | 模型 | HF ID | 参数量 | 默认小规格 | 预计 VRAM | 授权要求 |
|--------|------|-------|--------|----------|----------|---------|
| **1（首选）** | LTX-Video 2B distilled | `Lightricks/LTX-Video` | 2B | 16f×256×256, 8 steps | ~6-8 GB | ✅ 需 HF token + accept license |
| **2（次选）** | CogVideoX-2B | `THUDM/CogVideoX-2b` | 2B | 16f×256×256, 8 steps | ~6-9 GB | 无需额外协议（Apache 2.0） |
| **3（第三选）** | Wan2.1-T2V-1.3B | `Wan-AI/Wan2.1-T2V-1.3B` | 1.3B | 16f×256×256, 8 steps | ~8-10 GB | ✅ 需 HF token + accept license |

### 2.2 决策树（执行时严格按此顺序）

```
开始
 │
 ├─ 1️⃣ 尝试 LTX-Video 2B distilled（--model Lightricks/LTX-Video）
 │    ├─ 成功 → 记录结果，尝试 CogVideoX（bonus）
 │    └─ 失败 → 记录 blocker_ltx_video.md → 继续
 │
 ├─ 2️⃣ 尝试 CogVideoX-2B（--model THUDM/CogVideoX-2b）
 │    ├─ 成功 → 记录结果，尝试 Wan（bonus）
 │    └─ 失败 → 记录 blocker_cogvideox.md → 继续
 │
 ├─ 3️⃣ 尝试 Wan2.1-T2V-1.3B（--model Wan-AI/Wan2.1-T2V-1.3B）
 │    ├─ 成功 → 记录结果
 │    └─ 失败 → 记录 blocker_wan.md
 │
 └─ 结束：产出至少一个成功视频，或 3 个 blocker.md
```

### 2.3 不纳入主线的模型

| 模型 | 排除原因 |
|------|---------|
| HunyuanVideo (13B+) | 权重 >26GB，中等显存配置 连加载都做不到。仅在 README 中做架构对比参考（docs/10）。 |
| Wan2.1-14B | 权重 ~28GB，远超 中等显存配置。 |
| CogVideoX-5B | 权重 ~10GB + 中间激活可能超 中等显存配置，不强制尝试。 |
| Sora | 未开源，无可用权重。 |

---

## 3. 手动前置步骤清单

在执行任何视频脚本之前，必须完成以下步骤。未完成即运行的后果是脚本会报错 blocker。

### 3.1 环境准备

```bash
# 1. 安装 Python 3.13（通过 uv）
uv python install 3.13

# 2. 创建虚拟环境并安装依赖
cd /path/to/diffusion/
uv sync

# 3. 验证 CUDA 可用（远程 CUDA GPU）
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 4. 验证 diffusers 可用
python -c "import diffusers; print('Diffusers version:', diffusers.__version__)"
```

### 3.2 HuggingFace 授权

```bash
# 1. 登录 HF
huggingface-cli login
# 输入你的 HF token（从 https://huggingface.co/settings/tokens 获取）

# 2. 接受模型协议（在网页上操作）：
#    LTX-Video:    https://huggingface.co/Lightricks/LTX-Video  → 点击 "Agree and access repository"
#    Wan2.1-T2V:   https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B  → 同上
#    CogVideoX-2B: https://huggingface.co/THUDM/CogVideoX-2b  → Apache 2.0，通常无需额外授权

# 3. 验证模型可访问
python -c "from huggingface_hub import list_repo_files; print(list_repo_files('THUDM/CogVideoX-2b')[:5])"
```

### 3.3 显存预检

```bash
# 远程 CUDA GPU 上运行
nvidia-smi --query-gpu=memory.free --format=csv,noheader
# 预期：> 10 GB 空闲
```

---

## 4. 下载命令

每个模型首次运行时需要下载权重（约 5~10 GB），耗时 5~15 分钟（取决于网络）。后续运行使用 `~/.cache/huggingface/` 中的缓存。

| 模型 | 预计下载大小 | 下载方式 |
|------|------------|---------|
| LTX-Video 2B | ~9 GB | 脚本首次 `from_pretrained()` 自动下载 |
| CogVideoX-2B | ~10 GB | 同上 |
| Wan2.1-T2V-1.3B | ~5 GB | 同上 |

**预下载建议**（避免首次推理时的等待）：

```bash
# 在远程 CUDA GPU 上提前下载（只需一次）
python -c "
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-2b', torch_dtype=torch.float16)
# 下载完成即可退出，不需跑推理
"
```

---

## 5. 关键脚本列表

| 脚本 | 功能 | 默认规格 |
|------|------|---------|
| `run_ltx_video_if_possible.py` | LTX-Video 2B distilled 视频生成 | 16f×256×256, 8 steps, bf16, cpu_offload |
| `run_cogvideox_if_possible.py` | CogVideoX-2B 视频生成 | 16f×256×256, 8 steps, bf16, cpu_offload |
| `run_wan_if_possible.py` | Wan2.1-T2V-1.3B 视频生成 | 16f×256×256, 8 steps, bf16, cpu_offload |
| `profile_video_memory.py` | 视频 VRAM 预估与 profile（dry-run） | 默认不加载模型 |
| `sample_prompts.txt` | 6 条中英视频 prompt 参考 | — |

### 预期运行命令

```bash
# LTX-Video 2B（首选，蒸馏少步模型）
python experiments/reference_video_inference/run_ltx_video_if_possible.py \
  --prompt "一只白猫在草地上缓步走向镜头" \
  --num_frames 16 --height 256 --width 256 --num_steps 8 \
  --dtype bf16 --enable_cpu_offload --enable_vae_tiling \
  --output_dir experiments/reference_video_inference/results/

# CogVideoX-2B（次选，Apache 2.0 无需授权）
python experiments/reference_video_inference/run_cogvideox_if_possible.py \
  --prompt "一只白猫在草地上缓步走向镜头" \
  --num_frames 16 --height 256 --width 256 --num_steps 8 \
  --dtype bf16 --enable_cpu_offload --enable_vae_tiling \
  --output_dir experiments/reference_video_inference/results/

# Wan2.1-T2V-1.3B（第三选，1.3B 轻量）
python experiments/reference_video_inference/run_wan_if_possible.py \
  --prompt "一只白猫在草地上缓步走向镜头" \
  --num_frames 16 --height 256 --width 256 --num_steps 8 \
  --dtype bf16 --enable_cpu_offload --enable_vae_tiling \
  --output_dir experiments/reference_video_inference/results/

# VRAM profile（dry-run，不加载实体模型）
python experiments/reference_video_inference/profile_video_memory.py \
  --script ltx --dry_run --output_dir experiments/reference_video_inference/results/
```

---

## 6. 失败处理与降级路径

### 6.1 OOM 降级路径

当 GPU 报 CUDA Out of Memory 时，严格按以下级别降级，不做随机调参：

```
Level 0（默认尝试）:
  num_frames=16, res=256×256, steps=8, dtype=bf16, model_cpu_offload=enabled

    ↓ OOM

Level 1（轻度降级）:
  num_frames=12, res=240×240, steps=6, dtype=fp16, sequential_cpu_offload=enabled

    ↓ OOM

Level 2（重度降级）:
  num_frames=8, res=192×192, steps=4, dtype=fp16, + enable_vae_slicing

    ↓ OOM

Level 3（记录 blocker）:
  全部降级方案均 OOM → 记录 blocker，停止该模型尝试。
  连续 3 次 OOM 或 30 分钟无果 → 记 blocker，跳到下一个模型。
```

### 6.2 其他失败类型

| 失败类型 | 症状 | 处理 |
|---------|------|------|
| gated repo 拒绝 | `403 Client Error: Forbidden` | 检查 HF token（`huggingface-cli login`）和协议接受状态 |
| torch/diffusers 缺失 | `ModuleNotFoundError: No module named 'diffusers'` | 运行 `uv sync`；若环境无法创建，记录 blocker |
| 模型下载超时 | 下载卡住 >15min | 检查网络，使用 `HF_HUB_ENABLE_HF_TRANSFER=1` 加速 |
| 推理超时 | 单次推理 >15min | 记录 timeout blocker，尝试减帧数/减步数 |
| 输出全黑/全噪 | 视频内容不正常 | 检查 dtype、CFG scale（可能太小/太大）、seed |

### 6.3 失败停止条件

以下任一条件触发即停止该模型尝试，记录 blocker 并转向下一个模型：

- **连续 3 次 OOM**（所有降级方案均已尝试）
- **30 分钟无果**（从启动脚本算起，不是从下载算起）
- **无法解决的依赖/授权问题**（如 gated repo 无法访问）

---

## 7. Timebox 与时间预算

### 7.1 每模型时间盒

| 阶段 | 时间预算 | 说明 |
|------|---------|------|
| 快速配置尝试 | ≤10 分钟 | 使用默认小规格（16f×256×256, 8 steps）直接运行 |
| 慢速配置尝试 | ≤15 分钟 | 若 OOM，尝试降级路径（Level 1~2） |
| 总计（单模型） | ≤25 分钟 | 若全部失败则记录 blocker 并停止 |

### 7.2 总时间盒

| 内容 | 时间预算 |
|------|---------|
| LTX-Video 尝试 | ≤25 分钟 |
| CogVideoX 尝试 | ≤25 分钟 |
| Wan2.1 尝试 | ≤25 分钟 |
| 结果整理 + 报告 | ≤15 分钟 |
| **总计** | **≤ 1.5 小时**（理想情况 ≤ 1 小时） |

---

## 8. 结果目录结构与输出约定

```
experiments/reference_video_inference/
├── README.md                              # 本文件
├── run_ltx_video_if_possible.py           # LTX-Video 脚本
├── run_cogvideox_if_possible.py           # CogVideoX 脚本
├── run_wan_if_possible.py                 # Wan2.1 脚本
├── profile_video_memory.py                # VRAM profiling 工具
├── sample_prompts.txt                     # 示例 prompt
└── results/                               # 输出目录
    ├── ltx_video_16f_256_001.mp4          # LTX-Video 首次成功（如成功）
    ├── ltx_video_profiling.json           # LTX-Video VRAM profile
    ├── cogvideox_16f_256_001.mp4          # CogVideoX 首次成功（如成功）
    ├── cogvideox_profiling.json           # CogVideoX VRAM profile
    ├── wan_16f_256_001.mp4                # Wan 首次成功（如成功）
    ├── wan_profiling.json                 # Wan VRAM profile
    ├── blocker_ltx_video.md               # LTX-Video 失败记录（如失败）
    ├── blocker_cogvideox.md               # CogVideoX 失败记录（如失败）
    ├── blocker_wan.md                     # Wan 失败记录（如失败）
    └── video_summary.md                   # T15 全面总结
```

**命名约定**：`{model}_blah.{ext}`，文件名不含中文。

---

## 9. Blocker 记录模板

当模型失败时，必须在 `results/blocker_<model>.md` 中填写以下内容：

```markdown
# Reference Video Inference — Blocker (<模型名>)

**日期**：YYYY-MM-DD
**模型**：<HF model ID>
**设备**：可用的 CUDA GPU（中等显存配置）
**执行者**：T15 系统尝试

## 尝试配置
| 配置项 | Level 0 | Level 1 | Level 2 |
|--------|---------|---------|---------|
| dtype | bf16 | fp16 | fp16 |
| offload | model_cpu_offload | sequential_cpu_offload | sequential_cpu_offload |
| vae_tiling | enabled | enabled | enabled |
| resolution | 256×256 | 240×240 | 192×192 |
| num_frames | 16 | 12 | 8 |
| num_steps | 8 | 6 | 4 |
| cfg_scale | 1.0 | 1.0 | 1.0 |

## 错误类型
[OOM at denoising step X / CUDA error / 模型加载失败 / 视频 VAE 解码失败 / access denied / module not found / ...]

## 峰值 VRAM
（`nvidia-smi` 或 PyTorch `torch.cuda.max_memory_allocated()` 读数，标注发生在哪个阶段：文本编码/去噪循环/VAE 解码）

## 结论
[中等显存配置 不可行 / 需 >16GB VRAM / 依赖缺失 / 授权未通过 / ...]

## 对后续的建议
[如果本项目继续，建议在什么硬件上尝试视频；在中等显存配置下是否有替代路径]
```

---

## 10. 视频 Latent Shape 约定速查

不同视频模型的维度约定不同，脚本在处理中间 latent 时需注意：

| 模型 | Latent Shape 约定 | 视频 VAE 空间压缩 | 视频 VAE 时间压缩 |
|------|-------------------|------------------|------------------|
| LTX-Video 2B | `(B, C, T, H, W)` | 8× | 1×（无压缩）→ T=16 保持 |
| CogVideoX-2B | `(B, C, T, H, W)` | 8× | 4× → T=4@16 帧输入 |
| Wan2.1-1.3B | `(B, C, T, H, W)` | 8× | 4× |

**关键注意事项**：
- diffusers 的 pipeline 内部已处理 shape 转换，用户通常无需手动 transpose。
- 但若需要读取中间 latent 做 VRAM 或 shape 分析，必须确认当前是 `(B,C,T,H,W)` 还是 `(B,T,C,H,W)`。
- 视频 VAE 的时间压缩因子不一定等于空间压缩因子（如 CogVideoX 的 8× 空间 + 4× 时间 ≠ 8× 统一压缩）。

---

## 11. 参考

- **模型页面**：
  - LTX-Video: <https://huggingface.co/Lightricks/LTX-Video>
  - CogVideoX: <https://huggingface.co/THUDM/CogVideoX-2b>
  - Wan2.1: <https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B>
- **Diffusers 文档**：
  - LTX-Video Pipeline: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video>
  - CogVideoX Pipeline: <https://huggingface.co/docs/diffusers/main/en/api/pipelines/cogvideox>
- **项目内参考**：
  - 视频 latent 与 spacetime patch 学习笔记：`learning/notes/09_视频latent和spacetime_patch.md`
  - 视频架构知识库页：`docs/09_sora_style视频生成架构.html`
  - 四个视频模型对比：`docs/10_wan_hunyuan_cogvideox_ltx视频模型.html`
  - 参考实现结构（同模式）：`experiments/reference_image_inference/README.md`
- **计划详情**：`.omo/plans/modern-diffusion-inference-roadmap.md` T15 章节
