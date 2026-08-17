# 10 · Wan / Hunyuan / CogVideoX / LTX 开源视频模型对比

> 这里把四个开源文本到视频模型放在同一张表里，对照架构、推理路径、latent shape 和资源需求。HunyuanVideo 对硬件要求较高，本页主要用它比较架构。

## 1. 一表纵览

| 维度 | Wan2.1-T2V-1.3B | HunyuanVideo | CogVideoX-2B | LTX-Video 2B |
|------|-----------------|--------------|--------------|--------------|
| **开发者** | Wan-AI（阿里） | Tencent | THUDM（清华） | Lightricks |
| **参数量** | 1.3B | 13B+ | 2B | 2B |
| **协议** | 需接受许可 | 需申请访问 | Apache 2.0 | 需接受许可 |
| **HF ID** | `Wan-AI/Wan2.1-T2V-1.3B` | `tencent/HunyuanVideo` | `THUDM/CogVideoX-2b` | `Lightricks/LTX-Video` |
| **权重大小** | ~5 GB | ~26 GB | ~10 GB | ~9 GB |
| **资源档位** | 极限（10GB，需全部 offload） | 不可行（权重即超中等显存配置） | 可行（官方 min 4GB） | 可行（2B distilled，4-8 步） |

## 2. 架构对比

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/diffusion/docs/assets/architecture/wan_architecture.png" alt="Wan 视频模型架构图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：Wan 论文 Figure 9。图中明确给出 Wan-Encoder、DiT blocks、cross-attention 与 Wan-Decoder 的主干关系。
  </figcaption>
</figure>

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/diffusion/docs/assets/architecture/hunyuanvideo_architecture.png" alt="HunyuanVideo 总体架构图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：HunyuanVideo 论文 Figure 5。它强调了 causal 3D VAE、文本条件输入和视频 backbone 的整体闭环，是双流视频扩散系统的代表图。
  </figcaption>
</figure>

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/diffusion/docs/assets/architecture/cogvideox_architecture.png" alt="CogVideoX Expert Transformer 架构图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：CogVideoX 论文 Figure 3。图里同时展示了 3D causal VAE、text encoder 和 expert transformer blocks 的拼接方式，适合和 Wan / HunyuanVideo 做结构对照。
  </figcaption>
</figure>

### 2.1 Latent Shape 与 VAE

| 模型 | VAE 类型 | 空间压缩 | 时间压缩 | Latent 通道 | Latent Shape (16f×256² 例) |
|------|---------|---------|---------|------------|--------------------------|
| Wan2.1 | 3D VAE（时空联合压缩） | 8× | 4× | 16 | `(B,16,4,32,32)` |
| HunyuanVideo | 3D VAE（独立设计） | 8× | 4× | 16 | `(B,4,16,32,32)`（T 在前） |
| CogVideoX | 3D VAE（自训练） | 8× | 4× | 16 | `(B,16,4,32,32)` |
| LTX-Video | 2D VAE（无时间压缩） | 8× | **1×**（无） | **128** | `(B,128,16,32,32)` |

> **注意 HunyuanVideo 的维度约定**：HunyuanVideo 使用 `(B, T, C, H, W)` 而非 diffusers 默认的 `(B, C, T, H, W)`。这意味着在与其他模型交互或做 pipeline 集成时，需要显式 `transpose(1, 2)` 转换。这是视频模型中常见的容易出错的桩点。

### 2.2 DiT 结构

| 模型 | DiT 类型 | Attention 设计 | 位置编码 | 特殊设计 |
|------|---------|---------------|---------|---------|
| Wan2.1 | 基础 DiT（image DiT 扩展） | 分离式 spatial + temporal attention | 2D sin-cos + 1D temporal embed | 用 T5-XXL 做 text encoder（额外 ~1.5B 参数） |
| HunyuanVideo | MMDiT（双流，视频扩展） | 双流 joint attention（文本 / 视频 token 分别走各自分支后在 attention 中合并） | 3D RoPE | 使用 3D RoPE 统一编码空间和时间位置 |
| CogVideoX | 改良 DiT | 分离式 spatial + temporal causal attention | 2D sin-cos + 1D temporal embed | **Temporal causal mask**（每帧只能看当前和过去的帧，不能看未来帧） |
| LTX-Video | DiT（distilled） | Joint spacetime attention（所有 T×S token 统一进入 full attention） | 可学习 embed（简化，蒸馏后精度下降可接受） | **蒸馏模型**：4-8 步即可完成推理，无需 50 步 |

### 2.3 Text Encoder

| 模型 | Text Encoder | 额外显存 | 对受限显存配置的影响 |
|------|-------------|---------|--------------|
| Wan2.1 | T5-XXL（~1.5B） | ~3 GB (fp16) | 压力较大：text encoder 是主显存占用之一 |
| HunyuanVideo | CLIP-L + T5-XXL（双编码器） | ~5 GB | 压力很大：双编码器额外占用约 5 GB |
| CogVideoX | T5-XXL | ~3 GB (fp16) | 可缓解：cpu_offload 可转移 text encoder 到 CPU |
| LTX-Video | T5-small（~300M） | ~0.6 GB | 压力较小：text encoder 约占 0.6 GB |

## 3. 推理路径对比

### 3.1 通用推理流程

四个模型共享以下高层推理数据流（与图像推理的最大区别是 latent 的 T 维和 temporal attention）：

```python
# 1. 文本编码
prompt_emb = text_encoder(prompt)          # (B, L, d_text)
uncond_emb  = text_encoder(negative_prompt) # CFG 用

# 2. 噪声初始化
latents = randn(B, C, T_latent, H_latent, W_latent)  # 5D

# 3. 迭代去噪 (t = 1 → 0)
for t in timesteps:
    # 3a. 条件 / 无条件推理（分开或 batched CFG）
    v_cond   = dit(latents, t, prompt_emb)
    v_uncond = dit(latents, t, uncond_emb)

    # 3b. CFG 融合（在 vector field 层面，不在 latent 层面）
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

    # 3c. Euler step
    latents = latents + (t_next - t) * v_cfg

# 4. VAE 解码（3D → 像素）
video_frames = vae.decode(latents)  # (B, 3, T_frames, H_pixel, W_pixel)

# 5. 保存
export_to_video(video_frames, "output.mp4", fps=8)
```

### 3.2 各模型的推理差异

| 差异点 | Wan2.1 | HunyuanVideo | CogVideoX | LTX-Video |
|--------|--------|--------------|-----------|-----------|
| 默认步数 | 50 | 50 | 50 | **4-8**（蒸馏） |
| 默认帧数 | 81 | 129 | 49 | 121 |
| 默认分辨率 | 832×480 | 720×720 | 720×480 | 768×512 |
| CFG Scale | 5.0 | 6.0 | 6.0 | 1.0（蒸馏模型通常不需要 CFG） |
| 受限显存示例耗时 | ~10-15 min | 无法运行 | ~5-10 min | **~1-3 min** |

## 4. 受限显存配置下的现实路径

### 4.1 各模型的 资源档位分析

| 模型 | 权重 VRAM | Text Enc VRAM | Latent VRAM | Attn 激活 | 总计（估计） | 资源档位结论 |
|------|----------|---------------|------------|----------|-------------|-----------|
| LTX-Video 2B | ~3.7 GB | ~0.6 GB | ~0.5 GB | ~1.5 GB | ~6-8 GB | 首选验证候选 |
| CogVideoX-2B | ~3.7 GB | ~3.0 GB | ~0.1 GB | ~1.0 GB | ~6-9 GB | 可行：官方 min 4GB |
| Wan2.1-1.3B | ~2.4 GB | ~3.0 GB | ~0.1 GB | ~1.5 GB | ~8-10 GB | 极限：需所有 offload + 小规格 |
| HunyuanVideo | ~26 GB | ~5.0 GB | - | - | >30 GB | 当前配置不可行 |

### 4.2 推荐执行顺序

1. **LTX-Video 2B distilled**：2B 参数 + T5-small（~300M）+ 蒸馏 4-8 步 + 无 VAE 时间压缩。RTX 4060 8GB 上有 720×480×121 帧、耗时少于 1 分钟的实测记录，可以先用它检查环境和 pipeline。
2. **CogVideoX-2B**：Apache 2.0 协议无授权障碍。官方标注 min 4GB VRAM。49 帧 720×480 在 fp16 + offload 下约 9GB。小规格 (16f×256²) 更安全。
3. **Wan2.1-T2V-1.3B**：DiT 只有 1.3B 参数，但 T5-XXL text encoder 还要占用约 3GB。中等显存配置需要全量 offload，并降低帧数和分辨率。
4. **HunyuanVideo**：显存需求超过低资源路线的范围，本页只比较它的架构；实际评测需要更大显存的机器。

### 4.3 中等显存的参考配置

| 模型 | 帧数 | 分辨率 | 步数 | dtype | offload | 预计 VRAM |
|------|------|--------|------|-------|---------|----------|
| LTX-Video | 16 | 256×256 | 8 | bf16 | model_cpu_offload | ~6 GB |
| CogVideoX | 16 | 256×256 | 8 | bf16 | model_cpu_offload | ~6 GB |
| Wan2.1 | 16 | 256×256 | 8 | bf16 | model_cpu_offload | ~8 GB |

## 5. 四种架构取舍

### 5.1 CogVideoX：分离式时空 attention

- **结构**：spatial attention 与 temporal attention 分开计算，使用 causal temporal mask 和标准 16ch VAE。
- **取舍**：causal mask 限制未来帧信息，但每个 block 需要两次 attention。
- **资源**：官方最低要求为 4GB；中等显存配置仍建议降低帧数和分辨率。

### 5.2 LTX-Video：蒸馏与无时间压缩 VAE

- **结构**：高通道 VAE（128ch，不压缩时间）配合 joint attention。
- **取舍**：蒸馏减少去噪步数，未压缩的时间维会增加 DiT token 数。
- **资源**：RTX 4060 8GB 有公开实测记录。

### 5.3 Wan2.1：小 DiT 与大 text encoder

- **结构**：1.3B DiT 搭配 T5-XXL，参数分别落在较小的生成主干和较大的 text encoder。
- **取舍**：text encoder 增加 prompt 处理开销，DiT 本身较小。
- **资源**：中等显存需要 offload，并降低帧数与分辨率。

### 5.4 HunyuanVideo：MMDiT 双流

- **结构**：MMDiT 双流、3D RoPE 和较大的模型参数量。
- **取舍**：模型容量更高，显存和计算需求也明显增加。
- **资源**：需要 A100/H100 级硬件，不适用于本页讨论的受限显存环境。

> **结论**：低资源环境可以先测 LTX-Video 2B distilled 或 CogVideoX-2B。Wan2.1-1.3B 需要更积极的 offload 和降规格；HunyuanVideo 的资源需求更高，适合用来学习架构，或放到高配机器上评估。

## 6. 和我的 diffusion_engine 的关系

> **和我的 diffusion_engine 的关系**：diffusion_engine 目前只支持图像。四个 video DiT 对现有模块的影响如下：

| 视频概念 | diffusion_engine 模块 | 启发 |
|---------|----------------------|------|
| CogVideoX 分离式 attention | `core/attention.py` | 若未来扩展视频支持，应参考分离式 spatial + temporal attention（分开两个 Attention 实例，先 spatial 后 temporal）而非一个 monolithic full attention。这比 joint attention 的显存需求低。 |
| LTX-Video 蒸馏 + CFG=1.0 | `core/pipeline.py` | 蒸馏模型在 cfg_scale=1.0 时可以关闭 CFG 双 forward，少做一次 forward，并省下约 50% memory。 |
| Wan2.1 T5-XXL text encoder | `core/text_conditioning.py` | text encoder 可能是显存瓶颈（T5-XXL ~3GB）。在 ToyTextConditioner 中很小，但在 HFCachedTextConditioner 中需考虑 encoder 的 offload 策略。 |
| HunyuanVideo MMDiT 双流 | `core/dit.py` | TinyDiT 的拼接式 joint attention 是 toy 简化。真实的 MMDiT 视频扩展（HunyuanVideo）使用双流架构，文本和视频 token 分别走独立的 adaLN 调制路径后在 attention 中合并。这是 T11 TinyDiT 简化清单中应记录的差异。 |
| Latent shape 约定差异 | `core/pipeline.py` | pipeline 入口应显式处理 `(B,C,T,H,W)` 和 `(B,T,C,H,W)` 两种格式，避免未来接入真实模型时的 shape 错误。 |

**当前状态**：diffusion_engine 只支持图像。若继续扩展视频，CogVideoX 的分离式 spatial+temporal attention 改动最少：在现有 DiTBlock 中加入 TemporalAttention，原有 spatial attention 和 FFN 仍可复用。

## 7. 延伸阅读

- **Wan2.1 论文**：阿里 Wan 团队的 T2V 模型论文，1.3B/14B 两档，3D VAE 设计。
- **HunyuanVideo 论文**：Tencent HunyuanVideo，MMDiT 视频扩展 + 3D RoPE。
- **CogVideoX 论文**：THUDM 的因果视频生成模型，分离式 temporal attention + causal mask。
- **LTX-Video 论文**：Lightricks 的 real-time DiT 视频模型，蒸馏 + 128ch VAE。
- **学习笔记**：`learning/notes/09_视频latent和spacetime_patch.md` - 视频 latent shape 与 受限显存策略。
- **Sora 架构**：`docs/09_sora_style视频生成架构.html` - spacetime patch 与 video VAE 概念。
- **实验脚本**：`experiments/reference_video_inference/` - T15 三个视频推理脚本与 VRAM profile 工具。
