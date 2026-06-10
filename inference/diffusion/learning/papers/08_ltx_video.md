# 08 — LTX-Video 2B：Real-Time DiT 视频生成

> **模型名称**：LTX-Video（Lightricks）
> **官方仓库**：[github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
> **HF Model Card**：[huggingface.co/Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)
> **分类**：文生视频 + image-to-video — 最快开源视频模型（real-time DiT）
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

LTX-Video 是目前**最快的开源视频生成模型**，其 2B distilled 变体可以在消费级 GPU（如 RTX 4060 8GB）上以 < 1 分钟生成 121 帧 × 720×480 的视频。它代表了"few-step distillation + 激进 VAE 压缩 + 紧凑 DiT"的组合策略，是 12GB 场景下视频推理的**最佳选择**。对于 `diffusion_engine`，LTX-Video 验证了：① 视频可以在 few-step（4-8 步）下生成；② VAE 激进的压缩比可以大幅降低 DiT 的 token 数；③ real-time 推理不需要巨大的模型。

---

## 2. 模型类型

**文生视频（text-to-video）+ image-to-video**。LTX-Video 的核心规格：

| 属性 | 值 |
|------|-----|
| 参数量 | **2B**（DiT backbone） |
| VAE | 3D VAE（高压缩比，C=4） |
| 推理步数 | **4~8 步**（distilled） |
| 视频长度 | 最大 **121 帧** |
| 典型分辨率 | 720×480 ~ 768×512 |
| 最低 VRAM | **4 GB**（社区优化） |
| 许可 | 开源（需查看官方最新许可） |

**与同类模型的"速度"对比**：

| 模型 | 帧数 | 分辨率 | 步数 | 12GB 推理时间（估） | 速度等级 |
|------|------|--------|------|-------------------|---------|
| **LTX-Video 2B** | 121 | 720×480 | 4-8 | < 1 min | ⚡ 极快 |
| CogVideoX-2B | 49 | 720×480 | 50 | ~5-10 min | 🚶 中等 |
| Wan2.1-1.3B | 81 | 480×832 | 50 | ~8-15 min | 🚶 中等 |

LTX-Video 之所以快，不是因为 DiT 更高效（同样 2B 参数），而是因为：① 步数少（4 vs 50，12× 减少）；② VAE 压缩更激进（token 更少，attention 更轻）。

---

## 3. 核心架构

### 3.1 Denoiser：紧凑型 DiT

LTX-Video 的 denoiser 是一个**紧凑的 2B DiT**，并非追求极致质量，而是追求实时推理的可行性：

- **Transformer 层数**：较少（大约 20-24 层，远少于 SD3/FLUX 的 30-50 层）
- **Hidden dim**：较小（约 1536-2048，vs FLUX/SD3 的 3072+）
- **Attention**：使用 FlashAttention 或 xformers 优化
- **DiT block**：标准的 AdaLN + Self-Attention + FFN（无特殊的双流或 cross-attention 设计）

**设计哲学**："足够好"比"完美"更重要。LTX-Video 不追求在 academic benchmark 上超越 Wan/HunyuanVideo，而是追求在消费级 GPU 上的实时可用性。

### 3.2 Latent 表示：激进 VAE 压缩

LTX-Video 最独特的设计是其 VA E 的**激进压缩比**：

| VAE 特性 | LTX-Video | 其他视频 VAE |
|----------|-----------|------------|
| 空间压缩 | **32×**（比标准的 8× 激进 4 倍） | 通常 8× |
| 时间压缩 | **8×**（比标准的 4× 激进 2 倍） | 通常 4× |
| 通道数 | 4 | 4~16 |
| 压缩产物 | 极小的 latent（token 极少） | 标准 latent |

**空间 32× 压缩的影响**：
- 720×480 像素 → latent 空间尺寸 = 720/32 ≈ **22**，480/32 = **15**
- 对比 8× 压缩：720×480 → 90×60（空间面积 ~5400）
- 32× 压缩 → 22×15（空间面积 ~330）
- **空间面积减少 16×**！

**时间 8× 压缩的影响**：
- 121 帧 → T_latent = 121/8 ≈ **15**
- 对比 4× 压缩：121 帧 → T_latent ≈ 30
- 时间维度减半

**综合影响**：
```
标准 VAE（8×空间, 4×时间, 121f×720×480）：
  latent (4, 30, 90, 60) → tokens = 30 × 45 × 30 = 40,500

LTX-Video VAE（32×空间, 8×时间, 121f×720×480）：
  latent (4, 15, 22, 15) → tokens = 15 × 11 × 8 = 1,320
```

**Token 数从 40,500 → 1,320，减少了 30×！** 这才是 LTX-Video 速度快的根本原因——不是模型小，而是 latent token 数极少，attention 成本极低。

**代价**：激进压缩意味着 latent 的信息损失更大（VAE 需要在更少的 latent 像素中编码更多的视频信息），因此 LTX-Video 在纹理细节和超高清场景下不如 Wan 或 HunyuanVideo。但对于"可接受的视频质量 + 实时推理"这一目标，这个 tradeoff 是合理的。

### 3.3 Few-Step Distillation

LTX-Video 的蒸馏策略是其"实时"能力的关键：
- **Teacher**：标准 50 步 DiT 视频模型
- **Student**：4~8 步推理的蒸馏版本
- **蒸馏方法**：推测使用 progressive distillation 或 consistency distillation
- **CFG 行为**：蒸馏后 guidance 部分内化（cfg 可以接近 0 或使用较低 scale）

**4 步推理的 loop**：
```
for t in [1.0, 0.7, 0.3, 0.0]:   # 仅 4 个 timestep
    v = DiT(z_t, t, text)
    z_next = z_t + (t_next - t) * v
```

4 次 DiT forward，合计 ~4 × 0.1s = 0.4s（不含 VAE decode）。这比 50 步的 ~5s 快了 12×。

### 3.4 视频 Chunking（低延迟设计）

LTX-Video 支持视频 chunking：对于不满足显存的超长视频，可以将视频分成多个 chunk 分别生成后在时间维度上拼接。每个 chunk 独立走 denoising loop，chunk 之间通过少量重叠帧保持连贯性。

**对 12GB 的影响**：chunking 允许以更小的 per-chunk 显存需求生成长视频——例如 241 帧的视频可以分两个 121 帧的 chunk 分别生成。

### 3.5 Attention 结构

| 特性 | LTX-Video 2B |
|------|-------------|
| 自注意力类型 | Full attention, optimized with FlashAttention |
| Token 数（典型） | ~1,320 tokens（因激进 VAE 压缩） |
| Text-image 交互 | Cross-attention（text tokens 作为 K/V） |
| 位置编码 | 3D sinusoidal / learnable |
| Causal mask | 无 |

---

## 4. 推理数据流

```
prompt ("一只柴犬在草地上奔跑")
   │
   └─→ T5 tokenizer → T5 (4096d) → text embeddings
   │
   ▼
noise latent z_T ~ N(0, I)  shape: (1, 4, T_latent, H_latent, W_latent)
   121f×720×480 → VAE 32×空间 + 8×时间 → (1, 4, 15, 22, 15)
   │
   ▼
denoising loop（4~8 步）
   ├─ patchify: (1,4,15,22,15) → p=(1,2,2) → N_st = 15 × 11 × 8 = 1,320
   │   注：空间 patch 后尺寸 = (22/2=11, 15/2≈8)
   ├─ DiT forward: self-attn(1,320 tokens) + cross-attn(text)
   ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)     s ≈ 3.0~5.0
   └─ Euler step: z_next = z + dt * v
   │
   ▼
VAE decoder: (1, 4, 15, 22, 15) → (1, 3, 121, 720, 480)
```

**为什么 LTX-Video 的推理这么轻量**：
1. Token 数极少（~1,320 vs 40,500 for standard VAE）
2. DiT 步数极少（4~8 vs 50 步）
3. DiT 参数量小（2B vs 8B+）
4. 三者共同作用：总计算量约为同类模型的 `(1320²/40500²) × (4/50) × (2B/8B) ≈ 0.1%` ——是的，不到 1%！

---

## 5. 关键 Tensor Shape

### 5.1 激进 VAE 的 Token 数优势

| VAE 类型 | 帧数 | 分辨率 | Latent Shape | Spacetime Tokens |
|----------|------|--------|-------------|-----------------|
| **LTX (32×/8×)** | 121 | 720×480 | `(4, 15, 22, 15)` | `15 × 11 × 8 = 1,320` |
| **LTX (32×/8×)** | 61 | 512×384 | `(4, 8, 16, 12)` | `8 × 8 × 6 = 384` |
| 标准 VAE (8×/4×) | 121 | 720×480 | `(4, 30, 90, 60)` | `30 × 45 × 30 = 40,500` |
| 标准 VAE (8×/4×) | 49 | 720×480 | `(4, 13, 90, 60)` | `13 × 45 × 30 = 17,550` |

**对比结论**：LTX-Video 在生成长视频（121 帧）时的 token 数（1,320）甚至少于其他模型生成短视频（49 帧）的 token 数（17,550）。这就是"激进 VAE 压缩 + few-step"路线的威力。

### 5.2 不同规格下的 VRAM 估算

| 规格 | Token 数 | Attention 矩阵（fp16） | 总 VRAM 估算（含 T5） |
|------|---------|---------------------|---------------------|
| 61f×512p（小） | 384 | ~295 KB | ~4-5 GB |
| 121f×720p（标准） | 1,320 | ~3.5 MB | ~5-6 GB |
| 121f×1024p（大） | ~2,700 | ~14.6 MB | ~7-8 GB |
| 241f×720p（chunked） | 1,320 ×2 | ~3.5 MB ×2 | ~8-10 GB |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

LTX-Video 的显存瓶颈很低——即使标准规格也远低于 12GB：

| 排序 | 组件 | VRAM（121f×720p, fp16） | 说明 |
|------|------|------------------------|------|
| 🟡 1 | T5 text encoder | ~5 GB | 最大单组件 |
| 🟢 2 | DiT 权重（2B） | ~4 GB | |
| 🟢 3 | Attention activations | ~0.03 GB | 仅 1,320 tokens！ |
| 🟢 4 | VAE decoder | ~1-2 GB | |

### 6.2 12GB RTX 5070 Ti 可行性判断

| 配置 | 判断 | VRAM 估算 |
|------|------|----------|
| **LTX-Video 2B, 121f×720p, 4 steps** | 🟢 非常舒适 | ~6-7 GB |
| **LTX-Video 2B, 121f×720p, 8 steps** | 🟢 舒适 | ~7-8 GB |
| **LTX-Video 2B, 241f×720p (chunked)** | 🟡 极限可跑 | ~9-10 GB |

**推荐 12GB fallback 命令**：

```bash
# LTX-Video：最佳视频选择
python -c "
from diffusers import LTXPipeline
pipe = LTXPipeline.from_pretrained(
    'Lightricks/LTX-Video',
    torch_dtype=torch.float16
)
pipe = pipe.to('cuda')
# 不需要 offload！2B + T5 + 1,320 tokens 完全 fit 12GB
video = pipe(
    '一只柴犬在草地上奔跑',
    num_frames=121,
    width=720,
    height=480,
    num_inference_steps=8,
    guidance_scale=3.5
).frames[0]
# VRAM ≈ 6.5 GB, wall time < 1 min
"
```

**HF Model Card**：`https://huggingface.co/Lightricks/LTX-Video`

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `scheduler.py`
- Few-step（4 步）scheduler 是 LTX-Video 的核心。当前 `RectifiedFlowScheduler` 支持任意步数，但 timestep 序列是均匀分布的——distilled scheduler 需要非均匀分布（如 teacher 学到的分布）。
- T16 的 scheduler benchmark 应以 LTX-Video 的 4 步推理作为 few-step benchmark 基线。

### 7.2 `vae_stub.py`
- 激进 VAE 压缩（32×空间）对 vae_stub 的接口设计有重要启发：VAE 不仅是 `encode(image_pixels) → latent`，还应该暴露 `spatial_compression_ratio` 和 `temporal_compression_ratio` 属性，因为 pipeline 需要根据这些参数计算 latent 尺寸。

### 7.3 `attention.py`
- 1,320 tokens 的 attention 成本极低（矩阵 ~1.7M 元素），这意味着对于 LTX-Video，不需要 linear attention 或 sparse attention。但这也说明"减少 token 数"比"改变 attention 算法"更直接有效。

### 7.4 `pipeline.py`
- Few-step loop（仅 4 步）是 pipeline 的最简实现：text_encode() → noise init → for t in [1.0, 0.7, 0.3, 0.0]: denoise → decode
- Chunking 支持：如果视频太长，pipeline 应支持分段生成（`num_chunks` 参数，自动在时间维度拼接）

### 7.5 `memory_manager.py`
- 激进 VAE 压缩使 token 数极少（~1,320），memory manager 的"max_token_budget"在这种场景下自动放宽（因为远低于预算）
- Chunked generation 的显存管理：每个 chunk 是独立的，不需要在 chunk 间共享 latent buffer

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 官方 GitHub：`https://github.com/Lightricks/LTX-Video`
- HF Model Card：`https://huggingface.co/Lightricks/LTX-Video`
- arXiv：搜索 "LTX-Video real-time video generation"

**读**：
- 官方 README 中的推理命令和 VRAM 建议
- VAE 压缩比的详细说明（为什么选择 32×而不是 8×）
- Distillation 方法的描述（如何从 50 步降到 4 步）
- Diffusers pipeline 源码中的 scheduler 配置

**输出**：
- 本文档：`learning/papers/08_ltx_video.md`（8 字段完整 + VAE 激进压缩分析 + 12GB 判断）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
