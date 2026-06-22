# 03 — Sana：Efficient High-Resolution Image Synthesis

> **论文全称**：Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers
> **arXiv**：[2410.10629](https://arxiv.org/abs/2410.10629)
> **作者**：NVLabs（NVIDIA Research）
> **项目主页**：[nvlabs.github.io/Sana](https://nvlabs.github.io/Sana/)
> **GitHub**：[github.com/NVlabs/Sana](https://github.com/NVlabs/Sana)
> **分类**：文生图（image）— 高效高分辨率 DiT
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

Sana 代表 **"把 DiT 推向高效率极限"** 的路线。它用三个激进设计换取了消费级 GPU 友好的推理：① 用 **Gemma-2B 小 LLM** 替代 CLIP/T5 做 text encoder，② 用 **linear attention** 替代 O(n²) full attention，③ 用 **4-bit SVDQuant** 量化将模型压缩到 < 4GB 权重。结论：Sana 是 可用的 CUDA GPU 上**最现实的高质量文生图模型**，甚至能在 4096×4096 超高分辨率下运行。对于本项目 `diffusion_engine` 来说，Sana 验证了"text encoder 不一定是 CLIP"和"attention 不一定是 O(n²) softmax"两个重要偏离，这影响 T11 attention 接口设计和 T12 text conditioning 的架构选择。

---

## 2. 模型类型

**文生图（text-to-image）**，支持最高 4096×4096 分辨率。Sana 系列有三个参数规模：

| 变体 | 参数量 | 定位 | 资源档位 |
|------|--------|------|-----------|
| **Sana-0.6B** | 0.6B | 最小最快 | 🟢 非常舒适 (~7-9 GB) |
| **Sana-1.6B** | 1.6B | 质量优先（主流） | 🟢 舒适 (~10-12 GB fp16) |
| **Sana-Sprint-0.6B** | 0.6B（蒸馏） | 2 步推理，极速 | 🟢 最舒适 (~7 GB) |

**SVDQuant 4-bit 量化版本**（`SVDQuant-int4-Sana-1.6B`）：将 1.6B 模型压缩到 < 4GB 权重，全精度推理只需 < 8GB VRAM。这是 受限显存场景的**首选路径**。

---

## 3. 核心架构

### 3.1 Denoiser：Sana-DiT（Linear Attention DiT）

Sana 的 denoiser 是 DiT，但 attention 层从 **O(n²) softmax attention 替换为 linear attention**。这是 Sana 的**最核心创新**。

**Linear Attention 的原理**（推理视角）：

标准 softmax attention：
```
A = softmax(QK^T / √d) · V    → O(n²) 内存和时间
```
Linear attention（使用 kernel trick，如 ReLU 核）：
```
A = φ(Q) · (φ(K)^T · V)        → O(n) 内存和时间
其中 φ 是非线性核函数（ReLU、ELU+1、cos 等）
```

关键洞察：先算 `K^T·V`（`d×d` 矩阵，与 token 数 n 无关），再与 `φ(Q)` 相乘。这样 attention 的复杂度从 O(n²d) 降到 O(nd²)。当 n ≫ d 时（高分辨率下 token 数上千甚至上万），收益极大。

**实际影响**：对 1024×1024 图像的 4096 tokens，softmax attention 的中间矩阵 4096×4096 ≈ 16M 元素，而 linear attention 的中间矩阵仅为 d×d ≈ 1536×1536（或更小）。显存和计算量都大幅下降。

### 3.2 Latent 表示

- **自选 AE（AutoEncoder）**：Sana 使用自训练的 AE（非 SD VAE），8× 空间下采样
- **通道数可配置**：默认 32 通道、也可能降为 4 或 16（根据模型版本）
- 对 1024×1024 输入 → latent shape `(B, C, 128, 128)`，C 取决于 AE 配置
- **对比 SD3**：SD3 固定 16ch VAE，Sana 的 AE 更灵活，且论文中展示了"更小 AE + 更大 DiT"的 scaling tradeoff

### 3.3 Text Conditioning：Gemma-2B（LLM as Text Encoder）

这是 Sana 最具破坏性的设计：**用小 LLM 替代 CLIP+T5**。

| 对比 | SD3 / FLUX | Sana |
|------|-----------|------|
| Text encoder | CLIP-L + CLIP-G + T5-XXL（总计 ~13B） | **Gemma-2B**（仅 2B） |
| Hidden dim | 768 + 1280 + 4096 | **2304**（Gemma 隐藏维度） |
| VRAM（text encoder 单独） | ~16 GB（三个 encoder 全载） | **~4 GB**（仅 Gemma） |
| 长文本理解 | 依赖 T5-XXL 的 512 tokens | Gemma 自身有语言理解能力 |

**为什么 LLM 比 CLIP 好**：
1. LLM 的隐空间经过了大量文本训练，语言理解能力远强于 CLIP（后者主要做图文匹配）
2. 2B 小 LLM 比 11B T5-XXL 小 5×+，显存减少 3×+
3. LLM 输出的 hidden states 天然适合作为 diffusion 的条件（与 CLIP 的 last hidden 类似）

但代价是：LLM 的 hidden dim (2304) 与 diffusion 的 hidden dim（通常 1536-4096）不匹配，需要额外的投影。

**Text token 与 image token 的交互方式**：Sana 使用 **cross-attention**（而非 MMDiT 的 joint attention）。text tokens 作为 cross-attention 的 K/V，image tokens 作为 Q。这样 text tokens 不参与自注意力——进一步降低注意力复杂度（自注意力只对 image tokens，O(n_img²) → 或用 linear attention 降到 O(n_img)）。

### 3.4 Timestep / Sigma Conditioning

- **Rectified flow** 风格：`t ∈ [0, 1]`
- Fourier features + MLP → adaLN 调制参数
- 与 SD3 的 adaLN 路径类似，但 Sana 的 DiT block 内部结构更简化（deep-narrow 风格，层数多但 hidden dim 较小）

### 3.5 Attention 结构

| 特性 | Sana | SD3 | FLUX |
|------|------|-----|------|
| 自注意力类型 | **Linear attention**（O(n)） | Full attention（O(n²)） | Full attention（O(n²)） |
| Text-image 交互 | **Cross-attention** | Joint attention | Single-stream concat |
| 位置编码 | 2D sin-cos + 可学习 | 2D sin-cos / 可学习 | **2D RoPE** |
| QK normalization | 无（或简化） | 无 | **有** |
| FlashAttention 兼容 | 不适用（不是 softmax ATTN） | 是 | 是 |

### 3.6 VAE / AE

- **自训练 AE**（非 SD VAE 复用的 stock VAE）
- **支持 tiling**：大分辨率（4096px+）下分块 decode 避免 OOM
- **scaling_factor**：自训练 AE 的归一化常数，加载模型时自动获取

---

## 4. 推理数据流

### 完整路径

```
prompt
  │
  └─→ tokenizer → Gemma-2B (2304d, L_text tokens)
      注：Gemma-2B 的 tokenizer 即标准 SentencePiece，max_length ≈ 256
  │
  ▼
text embeddings: (B, L_text, 2304) → projected to D_text
           + pooled: Gemma 的 last token hidden state (B, 2304) → adaLN
  │
  ▼
noise latent z₁ ~ N(0, I)   shape: (1, C, H/8, W/8)
  │
  ▼
denoising loop: t = 1 → 0（通常 20~40 步，Sprint 仅 2 步）
  ├─ forward: DiT(z_t, text, t) → v_cond
  │  内部：patchify(p=2) → linear self-attn + cross-attn(text) → FFN → unpatchify
  ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)
  │        s ∈ [3.0, 5.0]（Sana 推荐 4.5）
  └─ Euler step: z_{t - Δt} = z_t + Δt · v_cfg
  │
  ▼
z₀ → AE decoder → pixel image [0, 1]
```

### Sana-Sprint 的特殊推理路径

Sana-Sprint 是 0.6B 的**蒸馏版本**，仅需 **2 步**推理：

```
noise z₁ ──→ 第1步 Denoising ──→ z_mid ──→ 第2步 Denoising ──→ z₀
```

两步之间没有中间 CFG 调节（guidance 已蒸馏内化），这意味着每步仅需一次 forward（不需要 cond+uncond 双 forward）。对于 受限显存场景，这是最快的出图路径。

---

## 5. 关键 Tensor Shape

### 5.1 Latent Shape（取决于 AE 配置）

| 分辨率 | Latent Shape（默认 32ch AE） | Image Tokens（p=2） | 说明 |
|--------|---------------------------|---------------------|------|
| 512×512 | `(1, 32, 64, 64)` | 32² = 1024 | |
| 1024×1024 | `(1, 32, 128, 128)` | 64² = 4096 | 主流分辨率 |
| 2048×2048 | `(1, 32, 256, 256)` | 128² = 16384 | linear attn 优势显现 |
| 4096×4096 | `(1, 32, 512, 512)` | 256² = 65536 | **仅 linear attn 可行** |

对比：4096px 下 softmax attention 需要 `65536² × 2B ≈ 8.6 GB` per attention layer，完全不可行。Linear attention 的中间矩阵仅 `d×d ≈ 1536×1536 ≈ 2.4M`，占用 < 20 MB。

### 5.2 Text Embedding Shape

| Encoder | Shape | 说明 |
|---------|-------|------|
| Gemma-2B tokens | `(1, 256, 2304)` | 可变长度，典型 256 |
| Gemma-2B pooled | `(1, 2304)` | last token hidden state |
| 投影后（送入 DiT） | `(1, 256, D_text)` | D_text = DiT hidden dim |

### 5.3 SVDQuant 4-bit 量化 Shape

| 组件 | fp16 权重 | int4 权重（SVDQuant） | 节省 |
|------|----------|---------------------|------|
| DiT (1.6B) | ~3.2 GB | **~0.8 GB** | 4× 压缩 |
| Gemma-2B | ~4 GB | **~1 GB**（可选量化） | 4× |
| AE | ~0.3 GB | 保持 fp16（AE 对量化敏感） | — |

**注意**：SVDQuant 不是简单 round-to-nearest int4，而是基于 SVD（奇异值分解）的量化方案，对 attention 的 weight 矩阵做了特定处理以保持推理质量。

---

## 6. 系统推理影响

### 6.1 显存瓶颈（按严重程度排序）

| 排序 | 瓶颈 | VRAM（fp16, 1024px） | 说明 |
|------|------|---------------------|------|
| 🟡 1 | Gemma-2B text encoder | ~4 GB | 比 T5-XXL (~11GB) 小 3× |
| 🟢 2 | Sana-DiT (1.6B) weights | ~3.2 GB | 参数量小，即使 fp16 也舒适 |
| 🟢 3 | Sana-DiT (0.6B) weights | ~1.2 GB | 非常轻量 |
| 🟢 4 | Attention activations | ~1 GB/step | linear attn 不产生 O(n²) 中间矩阵 |
| 🟢 5 | AE decoder | ~1.5 GB | 支持 tiling 进一步降 |

**为什么 Sana 比 SD3/FLUX 显存友好**：
1. Text encoder：4GB vs 11-16GB（T5/三 CLIP）
2. Attention：O(n) vs O(n²) → 4096 tokens 时 activation memory 少 ~100×
3. 参数量：0.6B/1.6B vs 2B(SD3 Medium)/4-5B(FLUX dev)

### 6.2 哪些可以 Cache

| 可 Cache | 收益 |
|---------|------|
| **Gemma-2B embeddings** | 一次 encode，全程复用（节省 4GB 常驻） |
| **AE encoder 输出** | image-to-image 场景（非文生图主场景） |
| Timestep embeddings | 所有 t ∈ [0,1] 可预计算 |

### 6.3 哪些不能 Cache

同 SD3/FLUX：denoiser 每步 K/V 不能 cache（latent 每步更新）。

但 linear attention 的"不能 cache"影响更小：因为 linear attention 本身就不需要存储完整的 O(n²) attention 矩阵，每步重算的 overhead 本来就少。

### 6.4 资源档位与运行边界

| 变体 | 判断 | VRAM | 推荐配置 |
|------|------|------|---------|
| **Sana-0.6B** | 🟢 非常舒适 | ~7-9 GB | `--model Efficient-Large-Model/Sana_600M_1024px_diffusers --dtype fp16` |
| **Sana-Sprint-0.6B** | 🟢 最舒适 | ~7 GB | 仅 2 步，latency < 1 秒 |
| **Sana-1.6B** | 🟢 舒适 | ~10-12 GB | fp16，无需 offload |
| **SVDQuant-int4-Sana-1.6B** | 🟢 首选 | **< 8 GB** | 质量损失 < 1%（SVDQuant 官方数据） |
| Sana-1.6B + 4096×4096 | 🟡 边界可跑 | ~15 GB | 极限分辨率需 AE tiling + CPU offload |

**一个受限显存示例命令**：

```bash
# 首选路径：SVDQuant int4 Sana-1.6B（质量+速度+VRAM 三赢）
# 注意：SVDQuant 需要特定的模型格式和加载方式
python -c "
from diffusers import SanaPipeline
pipe = SanaPipeline.from_pretrained(
    'Efficient-Large-Model/Sana_1600M_1024px_diffusers',
    torch_dtype=torch.float16
)
pipe = pipe.to('cuda')
# 不需要 offload！fp16 1.6B + Gemma-2B 在中等显存配置上完全可跑
image = pipe('一只柴犬在樱花树下', num_inference_steps=20, guidance_scale=4.5).images[0]
image.save('output.png')
"
# VRAM ≈ 10 GB（1024×1024，fp16，20 步）

# 极速方案：Sana-Sprint-0.6B（2 步推理）
python -c "
pipe = SanaPipeline.from_pretrained(
    'Efficient-Large-Model/Sana_Sprint_600M_1024px_diffusers',
    torch_dtype=torch.float16
)
pipe = pipe.to('cuda')
image = pipe('一只柴犬在樱花树下', num_inference_steps=2, guidance_scale=0.0).images[0]
# VRAM ≈ 7 GB，latency < 1 秒
"
```

**HF Model Card**：
- Sana-0.6B：`https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers`
- Sana-1.6B：`https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers`
- Sana-Sprint-0.6B：`https://huggingface.co/Efficient-Large-Model/Sana_Sprint_600M_1024px_diffusers`

**社区 SVDQuant 量化版本**：
- `https://huggingface.co/mit-han-lab/SVDQuant-int4-Sana-1.6B`（MIT HAN Lab，4-bit 量化，< 8GB VRAM）

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `attention.py`
- Sana 的 **linear attention** 是一个明确的扩展方向。当前 T11 的 `SelfAttention` 用的是 `F.scaled_dot_product_attention`（softmax full attention），未来可在 `attention.py` 中添加 `LinearAttention` 类作为替代实现
- **Linear attention 的接口**与 softmax attention 不同：不需要 `attn_mask`（无 O(n²) 矩阵），但需要有 `kernel_fn` 参数（如 `nn.ReLU()` 或 `lambda x: F.elu(x) + 1`）
- T18 总结时需明确标注：TinyDiT 使用 full attention（O(n²)），而 Sana 的 linear attention（O(n)）是高分辨率推理的正确方向

### 7.2 `text_conditioning.py`（T12 待实现）
- Sana 用 **Gemma-2B（LLM）** 替代 CLIP/T5，证明了 text encoder 不一定是 CLIP。这影响 T12 的 text conditioning 设计：
  - 接口不应写死为 CLIP，而应接受任意 encoder 的 hidden states
  - 我们的 v1 仍用 CLIP-ViT-L/14（最简单），但需在 `text_conditioning.py` 中声明"可替换为 Gemma-2B 等 LLM"
- LLM 的 hidden dim (2304) 与 DiT hidden dim 不匹配 → 需要投影层。T12 的 `TextConditioning` 应支持 `encoder_hidden_dim` 和 `projection_dim` 两个参数

### 7.3 `dit.py`
- Sana 使用 **cross-attention** 而非 MMDiT 的 joint attention 来处理 text-image 交互。这意味着我们的 TinyDiT 的 attention 接口需支持两种模式：
  - Self-attention（当前已实现）：image tokens 互 attend
  - Cross-attention（需扩展）：image tokens 作为 Q，text tokens 作为 K/V
- Sana 的 deep-narrow 设计（层数多、hidden dim 小）与 TinyDiT 的 toy 规模（2 层、hidden=256）不同，但设计理念一致：小而深 > 大而宽

### 7.4 `scheduler.py`
- 已实现的 `RectifiedFlowScheduler` 与 Sana 的 rectified flow 推理完全兼容
- Sana-Sprint 的 **2 步蒸馏推理**是一个重要的 benchmark：我们的 scheduler 能否在 2 步下生成合理结果？当前 toy rf 用 28 步才收敛，2 步需要特殊的蒸馏训练（Sana-Sprint 的实现证明可行）

### 7.5 `pipeline.py`（T12 待实现）
- Sana 的 denoising loop 结构（text encode 一次 → cross-attn 注入 → N 步 denoising → AE decode）是 T12 pipeline 的直接参考
- Sana-Sprint 的 **2 步推理**是最简 pipeline 的极致：仅需两个 denoising step + 一次 text encode + 一次 AE decode
- Cross-attention 方式比 joint attention 方式更适合 pipeline 的模块化设计（text encoder 和 DiT 的解耦更干净）

### 7.6 `memory_manager.py`（T12 待实现）
- Linear attention 的 memory tracking 与 softmax attention 完全不同：不能用 `n² × 2B × layers` 的计算公式，而应用 `d² × layers`（d = hidden dim）
- 这反过来证明 `memory_manager.py` 需要"attention type aware"的显存预估函数，不能假设所有 attention 都是 O(n²)

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 项目主页：`https://nvlabs.github.io/Sana/`
- arXiv：`https://arxiv.org/abs/2410.10629`
- GitHub：`https://github.com/NVlabs/Sana`
- HF model card (0.6B)：`https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers`
- HF model card (1.6B)：`https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers`
- HF model card (Sprint 0.6B)：`https://huggingface.co/Efficient-Large-Model/Sana_Sprint_600M_1024px_diffusers`
- SVDQuant 量化版：`https://huggingface.co/mit-han-lab/SVDQuant-int4-Sana-1.6B`

**读**：
- Paper 的 Section 3（Method）：重点看 linear attention 机制、Gemma text encoder 的接口、AE 的自训练方式
- Section 4（Experiments）：Sana vs SD3 vs FLUX 在 consumer GPU 上的 latency/VRAM 对比
- Official inference code（GitHub `demo.py`）：pipeline 的加载方式和推理参数
- SVDQuant 论文/技术报告：了解 4-bit 量化的原理和质量损失
- Diffusers pipeline 源码：`diffusers/pipelines/sana/pipeline_sana.py` 的 `__call__` 方法

**输出**：
- 本文档：`learning/papers/03_sana.md`（8 字段完整 + linear attention 细节 + SVDQuant 量化路径 + 资源档位判断）
- 不要求产出 HTML（HTML 留给 T8）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T7 (Wave 2)*
