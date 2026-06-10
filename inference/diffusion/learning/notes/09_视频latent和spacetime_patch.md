# 视频 Latent 与 Spacetime Patch 学习笔记

> **前置阅读**：`03_diffusion推理数据流.md`（第 9 节已建立 image vs video shape 基础差异）。
> **用途**：为 T15（视频 reference 脚手架）和 T18（最终汇总）提供 video latent shape、spacetime patch、维度约定、12GB 策略与 Week 5 执行模板。
> **注意**：本文档中所有规格均为理论推导和公开资料整理，标注"示例"的数值未经过本机/远程实测，T15 执行时需用真实 runtime 校验并纠正。

---

## 1. Image Latent 回顾

在进入视频之前，先巩固 image latent 的 shape 基线。

一个典型的文生图 latent（SD3/FLUX/Sana 的 VAE latent）是 **4 维张量**：

```
image_latent: (B, C, H, W)
```

常见实例：

| 像素分辨率 | VAE 压缩比 | Latent Shape | 通道数 C |
|-----------|-----------|-------------|---------|
| 512×512 | 8× | (1, 4, 64, 64) | 4 |
| 1024×1024 | 8× | (1, 4, 128, 128) | 4 |
| 1024×1024 | 16× | (1, 16, 64, 64) | 16（FLUX 高通道变体） |

**关键事实**：image latent 没有时间维度。当你看到一个 4 维 `(B, C, H, W)` tensor，它在语义上是"一批图，每图 C 个通道，每通道 H×W 空间"。进入 DiT 时，它被 patchify 成 `(B, N_patches, D)`，其中 `N_patches = (H/patch_size) * (W/patch_size)`。

---

## 2. Video Latent 的核心 Shape 差异

视频 latent 比图像 latent 多一个时间轴。这看起来简单，但**维度顺序**在不同的框架和模型之间不统一，造成了实际工程中最容易出错的桩点。

### 2.1 标准 5 维视频 latent

```
video_latent: (B, C, T, H, W)  或  (B, T, C, H, W)
```

典型的小规格示例（256×256、16 帧、8× VAE 空间压缩、4× VAE 时间压缩）：

```
# channels_first（diffusers 默认）
(B, C, T, H, W) = (1, 4, 16, 32, 32)

# channels_second（部分仓库使用）
(B, T, C, H, W) = (1, 16, 4, 32, 32)
```

**数字来源**：
- 原始像素：16 帧 × 256×256 × RGB = `(16, 3, 256, 256)`
- VAE 空间压缩 8×：256/8 = 32。H=W=32。
- VAE 时间压缩 4×：16/4 = 4。但视频 VAE 的时间压缩因子因模型而异：有些用 4×（CogVideoX），有些用 2×（部分 LTX 配置），有些不做时间压缩（逐帧 encode 再 stack）。因此 T 维度的值取决于所使用的 VAE，不能从原始帧数直接除以 8 推算。
- C=4 是 VAE latent 通道数（与 image VAE 相同）。

### 2.2 两种维度约定的来源与使用方

| 约定 | 典型使用方 | 说明 |
|------|----------|------|
| `(B, C, T, H, W)` | diffusers 默认、SD3/FLUX 视频扩展、Wan（diffusers 版本） | channels_first，与 PyTorch `Conv3d` 兼容（channel 在 dim=1） |
| `(B, T, C, H, W)` | CogVideoX（原始仓库）、LTX-Video（原始仓库）、某些自定义 trainer | 时间维在第二，方便做逐帧拆分和 CPU offload 时按 T 切片 |

**核心原则**：不要假设全项目用一种约定。在 `diffusion_engine/` 的 pipeline 入口层，**必须**做显式 rearrange。建议内部统一为 `(B, C, T, H, W)` 以匹配 PyTorch 3D 卷积和 diffusers 主流约定，在调用特定模型前按需 transpose。

### 2.3 从 model card 查约定（不是猜约定）

当接入一个新视频模型时，打开它的 HuggingFace model card，查以下信息：

1. **`pipeline.__call__` 签名**：看 `num_frames` 参数的默认值（如 CogVideoX 默认 49）。
2. **`vae.config`**：看 `sample_size` 和 `scaling_factor`。如果有 `temporal_compression_ratio` 字段，它就是时间压缩因子。
3. **官方示例代码**：看 `latents = torch.randn(...)` 的 shape 注释。
4. **`transformer.config`**：看 `patch_size`、`patch_size_t` 字段。如果有 `patch_size_t`，说明模型使用了 spacetime patch。

**错误的做法**：拿着 `(1, 4, 16, 32, 32)` 直接传给一个期望 `(1, 16, 4, 32, 32)` 的模型，期待它"内部自动转"——它不会。结果通常是静默的形状错误（被 broadcast 吃掉）或 cryptic error。

---

## 3. Spacetime Patch（时空 Patch）

### 3.1 概念

Spacetime patch 是 video DiT 进入 transformer 之前的 tokenization 步骤。它把 5 维 latent `(B, C, T, H, W)` 按 patch_size_t × patch_size_h × patch_size_w 切成 3D 小块，每个小块展平为 transformer 的一个 token。

这和图像 ViT 的 patchify 思路完全一致，只是 **多了一维时间轴**。

### 3.2 Patchify 过程（以 patch_size = (1, 2, 2) 为例）

```
输入: (B, C, T, H, W) = (1, 4, 16, 32, 32)
                               │
                               ▼
                    按 (patch_t=1, patch_h=2, patch_w=2) 切块
                               │
                               ▼
每个 patch: (C * patch_t * patch_h * patch_w) = 4 * 1 * 2 * 2 = 16 维向量
                               │
                               ▼
patch 数量: (T/patch_t) * (H/patch_h) * (W/patch_w) = 16 * 16 * 16 = 4096
                               │
                               ▼
Patch embedding (线性投影): 16 → D（如 D=768 或 1152 等 hidden dim）
                               │
                               ▼
输出: (B, N_patches, D) = (1, 4096, 768)
                               │
                               ▼
                      + time embedding（spacetime 3D）
                               │
                               ▼
                        进入 DiT blocks
```

### 3.3 不同模型的 patch 大小对比

不同模型在时间轴和空间轴上使用不同的 patch 大小，这直接决定了 token 数量和计算量。

| 模型 | patch_t | patch_h | patch_w | 说明 |
|------|---------|---------|---------|------|
| **Sora**（概念框架） | 1 | 2 | 2 | 时间不压缩（patch_t=1），空间 2×2。token 数 = T × (H/2) × (W/2) |
| **CogVideoX** | 1 | 2 | 2 | 同上，时间不 patch，空间 2×2。49 帧 720×480 latent → `49 * (90/2) * (60/2) ≈ 49*45*30 = 66,150` tokens。这是视频 DiT 自注意力的主要计算瓶颈。 |
| **LTX-Video** | 1 | 1 | 1 | 不做 patch（patch_t=1, patch_h=1, patch_w=1），每个 latent 像素直接是一个 token。121 帧 768×512 latent → `121 * (96) * (64) = 742,656` tokens——为什么 LTX 可以这样？因为它是 2B distilled，few-step（4~8 步），每次 forward 虽 token 多但步数极少。 |
| **Wan** | 1 | 2 | 2 | 与 CogVideoX 相同，时间不切，空间 2×2。 |

### 3.4 为什么空间 patch 时间不 patch？

大多数视频 DiT 选择 `patch_t=1`（时间轴不切块），`patch_h=patch_w=2`（空间 2×2 切块）。原因：

- **时间轴本来就短**：16~121 帧，远小于空间轴的 32~128。再切时间轴会丢失帧间细粒度差异。
- **帧间变化是核心信号**：视频模型的输出质量高度依赖对连续帧的精确建模。把相邻帧合并到一个 patch 会模糊运动信息。
- **计算分配**：空间维度是 token 数的主要来源（H/2 × W/2），已经足够压缩。时间维的 token 数（T 帧）相对于空间已经很小。

但也有例外：如果你的模型设计目标是"更激进的压缩"，可以将 patch_t 设为 2 或更大（如某些 Pyramid Flow Matching 变体）。这会减少 token 数但也会降低时域精度。

### 3.5 Token 数计算速查表

给定 latent `(B, C, T, H, W)` 和 patch `(p_t, p_h, p_w)`：

```
N_tokens = (T / p_t) * (H / p_h) * (W / p_w)
```

| 规格 | Latent Shape | Patch | Token 数 | 估算自注意力 FLOPs（B=1, D=1024） |
|------|-------------|-------|---------|----------------------------------|
| 小视频 16f@256 | (1,4,16,32,32) | (1,2,2) | 16 * 16 * 16 = **4,096** | ~34M |
| 中小视频 16f@256 alt | (1,4,16,32,32) | (2,2,2) | 8 * 16 * 16 = **2,048** | ~8.5M |
| 中视频 49f@480p | (1,4,49,60,90) | (1,2,2) | 49 * 30 * 45 = **66,150** | ~8.7G |
| 大视频 121f@512p | (1,4,121,64,96) | (1,2,2) | 121 * 32 * 48 = **185,856** | ~69G |
| LTX 121f raw | (1,4,121,64,96) | (1,1,1) | 121 * 64 * 96 = **742,656** | ~1.1T（但只用 4 步，总计算仍可控） |

**注意**：FLOPs 是粗略估计（2 × N² × D，标准 full attention）。实际模型可能用 flash-attention 或各种 kernel fusion 降低常数，且 MMDiT joint attention 还会加上 text token 的交互。这些数字仅用于理解规模差异，不做精确 benchmark。

---

## 4. Video DiT 与 Image DiT 的结构差异

### 4.1 Image DiT 的 attention 结构

Image DiT（如 SD3 MMDiT）只有 **空间 attention**。latent tokens 是 `(B, N_img, D)`，attention 权重在 N_img 个图像 patch token 之间计算。没有时间维度的概念。

```
Image DiT block:
  [Spatial Self-Attention] → [Cross-Attn 或 Joint-Attn] → [MLP] → [adaLN]
```

### 4.2 Video DiT 的 attention 结构

Video DiT 需要在空间和时间两个维度上都建模。主流方案有两种：

#### 方案 A：分开的 Spatial + Temporal Attention（常见）

```
Video DiT block:
  [Spatial Self-Attention] → [Temporal Self-Attention] → [Cross-Attn] → [MLP] → [adaLN]
```

- **Spatial Self-Attention**：在同一帧内部，H×W 个空间 token 之间做 attention。可以理解为"这一帧里哪些区域相关"。
- **Temporal Self-Attention**：在**同一空间位置**，T 个时间步之间做 attention。可以理解为"这个像素在 16 帧中如何变化"。

这种分离设计的优点：
- **显存高效**：Spatial attention 的复杂度是 O((HW)²)，temporal attention 是 O(T²)。分开算总开销远小于 merged 3D attention 的 O((T×H×W)²)。
- **可独立优化**：可以对 temporal attention 使用更小的 head dim 或不同的 flash-attn kernel。
- **CogVideoX 的 causal temporal**：CogVideoX 的 temporal attention 使用 causal mask（当前帧只能看自己和过去的帧），类似于 LLM 的因果约束。这是一个**重要例外**——大部分视频 DiT 的视频 attention 是 full（双向）的，但 CogVideoX 选择了 causal 以支持流式生成长视频。

#### 方案 B：Merged 3D Attention（简化但重）

```
Video DiT block:
  [Merged 3D Self-Attention] → [Cross-Attn] → [MLP] → [adaLN]
```

把所有 T×H×W 个 token 拼成一个长序列，做 full self-attention。复杂度 O((T×H×W)²)，仅适用于非常短的视频（T≤8）或 toy 实现。Wan 和某些研究用 full 3D attention，生产级部署通常拆分为 spatial + temporal。

### 4.3 为什么 Image DiT 不能直接跑 Video

有人会想："我的 TinyDiT（image-only）能不能直接塞一个 5D latent 进去？"

**不能**。原因如下：

1. **没有 temporal attention 层**：image DiT 只能建模同一帧内的空间关系，无法理解"第 3 帧和第 7 帧之间发生了什么"。强行 flatten T 维到空间维会让模型把相邻帧的不同时间内容当成空间内容混合，输出无意义的闪烁图像。
2. **没有 spacetime patch**：TinyDiT 的 patchify 假设输入是 4D，不能用 `patch_t` 参数切时间轴。
3. **没有视频 VAE 的 decoder**：视频 VAE decoder 可能需要时间维度的特殊处理（3D conv 或 temporal smoothing），image VAE decoder 只能逐帧 decode，帧间不连贯。

**结论**：video DiT 是 image DiT 的结构性扩展，不是简单加一维参数就能覆盖的。TinyDiT 是 image-only 的 toy 实现，视频模型需要独立实现。

---

## 5. 12GB VRAM 下的视频推理现实规格

### 5.1 核心预算

- 有效 VRAM：12GB × 0.85 ≈ **10.2GB**
- 超预算行为：先降 resolution → 降帧数 → 降 steps → 开 CPU offload → 降 dtype。如果全部调到最低仍 OOM，**记录 blocker**，不再无限制调参。

### 5.2 模型优先级与推荐规格

按"大概率能在 12GB 下跑通"从高到低排序：

| 优先级 | 模型 | 推荐规格 | 预计 VRAM | 预计耗时 | 说明 |
|--------|------|---------|----------|---------|------|
| **1（首选）** | LTX-Video 2B distilled | ≤16 帧, 256×256, 8 步, fp16 | ~6-8 GB | <2 min | 2B params + few-step distillation。RTX 4060 8GB 上实测 720×480×121 帧 <1 分钟。是最可能首次就跑通的视频模型。 |
| **2（备选）** | CogVideoX-2B | ≤16 帧, 256×256, 30 步, fp16 | ~6-9 GB | ~5-10 min | 官方要求 min 4GB VRAM。12GB 充裕。49 帧 720×480 需要 ~9GB。 |
| **3（高难度）** | Wan2.1-1.3B | ≤16 帧, 256×256, 30 步, fp16 | ~8-10 GB | ~10-15 min | 1.3B 参数轻量，但 480p × 5s（81 帧）需 ~8GB。在 12GB 下是极限操作，需开启所有 offload。 |

### 5.3 降级路径（每级降到"能跑"为止）

```
Level 0（默认尝试）:
  num_frames=16, res=256×256, steps=30 (或模型推荐值), dtype=fp16, model_cpu_offload=enabled

Level 1（OOM 后降级）:
  num_frames=12, res=240×240, steps=20, dtype=fp16, sequential_cpu_offload=enabled

Level 2（极限降级）:
  num_frames=8, res=192×192, steps=10, dtype=fp16, + enable_vae_slicing

Level 3（记录 blocker）:
  全部降级方案均 OOM → 记录为 blocker，不再尝试。这仍然是有效产出。
```

### 5.4 不推荐的模型（12GB 下无法跑通）

| 模型 | 原因 |
|------|------|
| HunyuanVideo (13B+) | 权重文件 >26GB，12GB 连加载都做不到 |
| Wan2.1-14B | 权重 ~28GB，远超 12GB |
| Sora（任何版本） | 未开源，无可用权重 |
| CogVideoX-5B | 权重 ~10GB，加上 latent 和中间激活会超过 12GB，需要用 24GB+ 卡 |

---

## 6. Week 5 视频尝试执行模板（T15 使用）

### 6.1 Timebox

**总时间盒**：Week 5（约 7 天），每天 1-2 小时用于视频尝试。

**每模型时间盒**：
- LTX-Video 2B：最多 1.5h（下载 + 加载 + 推理 + 记录）
- CogVideoX-2B：最多 1.5h
- Wan2.1-1.3B：最多 1.5h（如果前两个已成功，此为 bonus）

**失败标准**：
- 单个模型连续 3 次 OOM（即：Level 0 OOM → Level 1 OOM → Level 2 OOM 后仍未成功）→ **记录 blocker，停止该模型**
- 单次推理超 15 分钟 → 记录 timeout blocker
- 下载失败 / 缺少 HF token / 协议未接受 → 记录 access blocker

### 6.2 执行清单（每次尝试必做）

```markdown
- [ ] 确认 HF token 已配置（huggingface-cli login）
- [ ] 确认目标模型协议已接受（在 HF model card 页面点击 "Agree and access repository"）
- [ ] 确认 Python 环境正确（uv sync；检查 torch.cuda.is_available()）
- [ ] 运行 `nvidia-smi` 记录初始 VRAM 空闲量
- [ ] 启动推理，监控 VRAM
- [ ] 记录结果到对应 result 文件
```

### 6.3 记录字段模板

每次尝试必须填写以下字段，无论成功或失败：

```markdown
| 字段 | 值 |
|------|---|
| 模型名 | （如 CogVideoX-2B） |
| HF repo id | （如 THUDM/CogVideoX-2b） |
| 分辨率 (width×height) | （如 256×256） |
| 帧数 (num_frames) | （如 16） |
| 推理步数 (num_inference_steps) | （如 30） |
| dtype | （如 torch.float16） |
| 是否 CPU offload | （是/否，具体模式） |
| 是否 VAE tiling/slicing | （是/否） |
| 峰值 VRAM (GB) | （nvidia-smi 记录或脚本内 torch.cuda.max_memory_allocated） |
| 推理耗时 (s) | （wall clock，不含模型下载时间） |
| 输出文件路径 | （如 results/cogvideox_16f_256_001.mp4） |
| Blocker 描述 | （无/OOM at step X/timeout/access denied/...） |
```

### 6.4 输出文件命名约定

```
experiments/reference_video_inference/results/
├── ltx_video_16f_256_001.mp4        # LTX-Video 首次成功
├── ltx_video_blocker.md             # LTX-Video 失败记录（如果失败）
├── cogvideox_16f_256_001.mp4        # CogVideoX 首次成功
├── cogvideox_blocker.md             # CogVideoX 失败记录（如果失败）
├── wan_16f_256_001.mp4              # Wan 首次成功
├── wan_blocker.md                   # Wan 失败记录（如果失败）
└── video_summary.md                 # Week 5 全面总结
```

### 6.5 成功/失败两者的交付标准

**T15 的交付物无论模型是否跑通，都是有效的**：
- ✅ 至少一个模型跑通 → 交付 1+ 个 mp4 + 完整记录字段
- ✅ 全部模型失败 → 交付 3 个 blocker.md + video_summary.md（分析为什么在 12GB 下无法完成，以及如果换 24GB 卡预期会如何）
- ❌ 无效交付：只写了"模型太大跑不了"但没有尝试过任何降级路径

---

## 7. 本页结论

视频 latent 的核心复杂度来自**多出来的时间维度**、**不统一的维度约定 `(B,C,T,H,W)` vs `(B,T,C,H,W)`**、以及**spacetime patch 引入的 3D tokenization**。video DiT 不是 image DiT + 一个参数开关，而是需要独立的 temporal attention 层和修改后的 patchify 流程。在 12GB VRAM 下，视频推理必须走"小规格 + 优先少步蒸馏模型"的路线，并设置明确的 timebox 和 blocker 边界以防无底洞式的调试。

**对下游任务的影响**：
- T15（视频 reference 脚手架）：本文档第 6 节是直接执行手册。
- T18（最终汇总）：本页结论的"实际验证 vs 理论预期"对比是最终报告的核心段落。
- diffusion_engine：`pipeline.py` 必须在入口处显式处理 `(B,C,T,H,W)` vs `(B,T,C,H,W)` 转换，且在 shape 注释中注明。
