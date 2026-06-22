# 04 — Sora-Style Video Generation（OpenAI Sora 技术说明）

> **来源**：OpenAI 技术报告 *Video generation models as world simulators*（2024-02-15）
> **官方链接**：[openai.com/index/video-generation-models-as-world-simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
> **开源复现参考**：Open-Sora（`github.com/hpcaitech/Open-Sora`）、Open-Sora-Plan（`github.com/PKU-YuanGroup/Open-Sora-Plan`）
> **分类**：文生视频 + image-to-video — 视频 DiT 架构范式奠基
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

Sora 虽未开源，但它的技术说明提出了视频生成的**关键架构范式**，后续所有开源视频模型（Wan、HunyuanVideo、CogVideoX、LTX-Video）都在不同程度上沿用了这些概念。Sora 定义了三件事：① spacetime patch（时空统一的 tokenization）；② 视频 VAE（时间+空间联合压缩）；③ 可变时长/分辨率/宽高比的统一处理。理解 Sora 的架构轮廓，是理解其他视频 DiT 的先决条件。

---

## 2. 模型类型

**文生视频（text-to-video）+ image-to-video**。Sora 可以接受文本提示、图像、视频或 combined 条件生成新视频。它支持可变时长（最长 ~1 分钟）、可变分辨率（最高 1080p）和可变宽高比（不限于固定方形），这在当时是前所未有的。模型规模未公开，但第三方估计参数量在数十 B 级别，训练计算量极大。

---

## 3. 核心架构

### 3.1 Spacetime Patch：视频的"Token 化"

这是 Sora 最基础也最重要的架构贡献。图像 DiT 将 latent `(B, C, H, W)` 切割为 2D patches——每个 patch 是 `(p, p)` 的方形区域。Sora 将这一思想扩展到三维：**spacetime patch** 同时在时间和空间维度上切割视频 latent。

具体而言，一个 spacetime patch 覆盖 `(t_p, h_p, w_p)` 的时空块，例如 `(1 帧, 2×2 空间)` 或 `(2 帧, 2×2 空间)`。这些 patches 被 flatten 为 token 序列送入 DiT。

```
图像 DiT：patchify(latent_2d) → (B, N_img, D)      N_img = (H/p) × (W/p)
视频 DiT：patchify(latent_3d) → (B, N_spacetime, D) N_st = (T/t_p) × (H/h_p) × (W/w_p)
```

**为什么这很重要对推理**：
- Attention 是 full attention over all spacetime tokens。token 总数 = `(T_latent/t_p) × (H_latent/h_p) × (W_latent/w_p)`，三个维度乘在一起，token 数极易破万。
- 这与 LLM 的序列长度瓶颈类似——视频越长、分辨率越高、帧数越多，attention 的 O(n²) 代价就越严重。
- 因此 Sora 必须依赖非常激进的视频 VAE 压缩（见 3.2）来把 token 数压到可控范围。

### 3.2 视频 VAE（Video Compression Network）

Sora 使用一个专用的视频 VAE（或称 video compressor）在时间和空间维度上同时压缩原始视频像素。与图像 VAE 不同，视频 VAE 在 3D 卷积/transformer 中处理 `(T, H, W)` 三维。

**压缩比**（第三方推测）：
- 空间压缩：8×（与图像 VAE 相同，如 256×256 → 32×32）
- 时间压缩：4×（如 16 帧 → 4 帧 latent）
- 通道数：4（推测，与 SD VAE 保持一致）

**示例计算**：
```
原始视频：16 frames × 256×256 × RGB = (3, 16, 256, 256)
                  ↓ VAE 空间 8× + 时间 4×
Video latent：(4, 4, 32, 32)            # (C, T_latent, H_latent, W_latent)
                  ↓ spacetime patch (t_p=1, h_p=2, w_p=2)
Spacetime tokens：4 × 16 × 16 = 1024 tokens
```

1024 个 tokens 对于 full attention 是可控的（attention 矩阵 1024² ≈ 1M 元素）。如果没有时间压缩（16 帧 → 16 帧 latent），token 数 = 16 × 16 × 16 = 4096，仍然可控但接近极限。如果视频是 64 帧 × 512×512，无时间压缩则 token 数 = 64 × 32 × 32 = 65536，attention 矩阵 65536² ≈ 4.3B 元素 → 不可能。

### 3.3 Variable Duration / Resolution / Aspect Ratio

Sora 可以处理任意时长、分辨率和宽高比的视频——训练时使用原生分辨率/帧率/宽高比的视频，而非所有视频都 resize 到固定大小。这依赖两个设计：

1. **Spacetime patch 的灵活性**：patch 大小固定，不同分辨率的视频自然产生不同数量的 tokens。
2. **Position embedding 的泛化**：Sora 对每个 spacetime patch 分配一个基于其 `(t, y, x)` 坐标的位置编码。由于训练时见过各种尺寸，推理时可以处理训练分布内的任意尺寸。

这对推理意味着什么：不需要在推理前将输入 resize 到固定分辨率——直接输入原始尺寸即可。但这也意味着 token 数不固定，attention 的峰值显存因输入而异。

### 3.4 Recaptioning（视频字幕重写）

Sora 的技术报告特别强调了 recaptioning 的重要性：使用一个专门的 video captioner 对训练视频生成高度描述性的文本，然后在训练时使用这些 rewritten captions 而非原始的简短 alt-text。这提升了 text-video 对齐质量。

**对推理的影响**：推理时的 prompt 不需要是"高度描述性的"——recaptioning 仅在训练时使用。但推理 prompt 的质量仍然影响生成结果（如所有扩散模型），只是不需要像训练那样用专门的 recaptioner 预处理。

### 3.5 Denoiser：DiT（未公开细节）

Sora 的 denoiser 是 DiT，但具体架构细节未公开。根据技术报告和第三方分析：
- **Attention**：full attention over all spacetime tokens（所有帧的所有 patches 互相 attend）
- **Timestep conditioning**：推测使用 adaLN 或 cross-attention 注入 timestep（与 SD3/FLUX 类似）
- **Text conditioning**：未公开 encoder 类型（推测使用 T5 或内部 LLM）
- **Diffusion 框架**：未公开是 DDPM 还是 rectified flow（第三方推测可能是 rectified flow）

---

## 4. 推理数据流

```
prompt ("一只猫在草地上奔跑，阳光穿过树叶...")
   │
   └─→ text encoder（未公开）→ text embeddings
   │
   ▼
noise latent z_T ~ N(0, I)  shape: (1, C, T_latent, H_latent, W_latent)
   │
   ▼
denoising loop（T → 0, 推测 50~100 步）
   ├─ patchify: (B,C,T,H,W) → (B, N_st, D)
   ├─ DiT full attention over all spacetime tokens
   ├─ CFG（在 vector field 或 noise prediction 层面）
   ├─ scheduler step（类型未公开）
   └─ unpatchify: (B, N_st, D) → (B,C,T,H,W)
   │
   ▼
video VAE decoder → 像素视频 (B, 3, T_frame, H_pixel, W_pixel)
```

**注**：Sora 使用 "video decompressor" 而非标准 VAE decoder 这一术语，暗示其 decoder 可能包含额外的超分辨率或细节增强步骤。

---

## 5. 关键 Tensor Shape

### 5.1 原始视频 → Video Latent

| 阶段 | Shape | 说明 |
|------|-------|------|
| 原始视频 | `(B, 3, T_frame, H, W)` | 如 16f×256×256 → `(1, 3, 16, 256, 256)` |
| VAE encode（时间 4× + 空间 8×） | `(B, C, T_latent, H_latent, W_latent)` | 如 `(1, 4, 4, 32, 32)` |

### 5.2 Spacetime Patch Token 计算

| Patch 配置 | T_latent | H_latent | W_latent | Token 总数 |
|-----------|----------|----------|----------|-----------|
| `(1, 2, 2)` | 4 | 32 | 32 | `4 × 16 × 16 = 1024` |
| `(2, 2, 2)` | 4 | 32 | 32 | `2 × 16 × 16 = 512` |
| `(1, 2, 2)` | 8 | 64 | 64 | `8 × 32 × 32 = 8192` |
| `(1, 2, 2)` | 16 | 64 | 64 | `16 × 32 × 32 = 16384` |

**关键观察**：128 帧 × 512×512 的视频（t_p=1, h_p=w_p=2）：T_latent=32, H_latent=W_latent=64 → 32×32×32 = **32768 tokens**。Full attention 矩阵 32768² ≈ 1.07B 元素，在 fp16 下约 2.1 GB per attention layer。这已经超出了 中等显存配置的承受范围。

### 5.3 视频 VAE Latent Shape 与其他模型对比

| 模型 | VAE 类型 | 空间压缩 | 时间压缩 | C | 典型 latent shape（16f×256²px） |
|------|---------|--------|--------|---|-------------------------------|
| **Sora（推测）** | 3D VAE | 8× | 4× | 4 | `(1, 4, 4, 32, 32)` |
| **Wan2.1** | 3D VAE | 8× | 4× | 16 | `(1, 16, 4, 32, 32)` |
| **CogVideoX** | 3D Causal VAE | 8× | 4× | 4 | `(1, 4, 4, 32, 32)` |
| **LTX-Video** | 3D VAE（高压缩） | 32× | 8× | 4 | `(1, 4, 2, 8, 8)` |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

Sora 未开源，以下基于公开信息推理：

- **最大瓶颈**：DiT full attention over spacetime tokens。token 数随帧数/分辨率线性增长，attention 内存 O(n²) 增长。
- **次要瓶颈**：视频 VAE decoder 一次性 decode 所有帧（而不是逐帧 decode），中间激活大。
- **模型权重**：由于模型未公开，参数量不可知，但数十 B 级别的 DiT + 大型 text encoder + 视频 VAE 意味着权重加载就需要数十 GB VRAM。

### 6.2 哪些可以 Cache

| 可 Cache | 收益 |
|---------|------|
| **Text embeddings** | 一次 encode，全程复用 |
| **VAE encoder 输出** | image-to-video 场景下输入图像/视频的 latent 可 cache |

### 6.3 哪些不能 Cache

- **Denoiser K/V**：每步 latent 刷新，K/V 不能复用（与所有 diffusion 模型相同）。
- **Spacetime attention activation**：不同 timestep 下 attention pattern 不同。

### 6.4 资源档位与运行边界

> 🔴 **不适合**。Sora 未开源且原模型巨大（数十 B 参数），在中等显存配置下也远远不够。即使降分辨率到 128×128 和 8 帧，数十 B 的 DiT 权重加载就已超过 中等显存配置。

但这**不意味着理解 Sora 没有价值**。Sora 定义的 spacetime patch + 视频 VAE 范式正是理解 Wan、HunyuanVideo、CogVideoX、LTX-Video 等小模型的基础。这些模型在更小参数规模（0.6B~14B）下实现了类似 Sora 的视频生成能力，其中多个可以在中等显存配置上运行。

**结论：Sora 不直接跑，其架构概念是小模型的推理基础。**

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `dit.py`
- 当前 `TinyDiT` 只支持 2D patchify（`(B,C,H,W)` → `(B,N,D)`）。需要扩展为 3D spacetime patchify ——接受 `(B,C,T,H,W)` 输入，在 `(T,H,W)` 三维上 patchify。
- 3D patchify 需要 3D 位置编码：`(t_pos, y_pos, x_pos)` 分别编码后相加。

### 7.2 `attention.py`
- Spacetime full attention 是视频 attention 的基础形式。不同于图像的 2D full attention（所有 H×W patches 互 attend），视频的 3D full attention 是所有 T×H×W patches 互 attend。
- Token 数膨胀意味着 O(n²) attention 在实际中不可行——这为 `LinearAttention`（参考 Sana 的 linear attention）提供了更强的动机。

### 7.3 `vae_stub.py`
- 视频 VAE 需要定义 `encode_video()` 和 `decode_video()` 接口，输入/输出为 5D tensor `(B,C,T,H,W)`。
- 视频 VAE decode 应支持分帧/chunk decode 以避免一次性分配全部像素 buffer。

### 7.4 `memory_manager.py`
- 视频 latent buffer 是 5D tensor `(B,C,T,H,W)`，需要不同于图像 4D 的分配/复用策略。
- Spacetime attention 的显存预估公式：`n² × 2bytes × num_layers × 2(QK+AV)`，其中 n = `(T_latent/t_p) × (H_latent/h_p) × (W_latent/w_p)`。

### 7.5 `pipeline.py`
- 视频 denoising loop 与图像 loop 的逻辑本质相同（text encode → noise → N 步 denoise → decode），只是 shape 从 4D 变为 5D。
- 这意味着 pipeline 可以设计为 shape-agnostic（通过统一接口 + shape dispatch），而不是分别维护 image_pipeline 和 video_pipeline。

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- OpenAI 官方技术说明：`https://openai.com/index/video-generation-models-as-world-simulators/`
- Open-Sora 开源复现（architecture 文档）：`https://github.com/hpcaitech/Open-Sora`
- Open-Sora-Plan 开源复现：`https://github.com/PKU-YuanGroup/Open-Sora-Plan`

**读**：
- 技术说明全文，尤其是 "Turning visual data into patches" 和 "Video compression network" 两节
- 开源复现项目中的 architecture doc（了解工程落地时的 shape 约定和技术取舍）
- 社区关于 spacetime patch token 计算和 video VAE 压缩比的讨论

**输出**：
- 本文档：`learning/papers/04_sora_style_video_generation.md`（8 字段完整 + spacetime patch 计算 + 视频 VAE 说明 + 资源档位判断）
- 不要求独立 HTML（Sora 架构内容融入 T8 docs/04-06 + T15 docs/09-10）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
