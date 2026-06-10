# 01 — Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（SD3 / MMDiT）

> **论文全称**：Scaling Rectified Flow Transformers for High-Resolution Image Synthesis
> **arXiv**：[2403.03206](https://arxiv.org/abs/2403.03206)
> **作者**：Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach（Stability AI）
> **发表时间**：2024 年 3 月
> **分类**：文生图（image）— rectified flow + MMDiT 奠基
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

这是 **rectified flow + MMDiT（Multi-Modal Diffusion Transformer）** 的定义论文。它建立了现代 diffusion transformer 推理的标准范式：用 MMDiT 的 joint attention 同时处理 text token 和 image latent token，在 vector field 层面做 classifier-free guidance（CFG）。理解 SD3 推理是理解 FLUX、Sana、以及所有视频 DiT 的前提。不夸张地说，这篇论文定义了 2024 年以后开源文生图模型的推理骨架。

---

## 2. 模型类型

**文生图（text-to-image）**。SD3 系列有三个尺寸：
- **SD3 Medium**：2B 参数，适合消费级 GPU（12GB VRAM 可跑）
- **SD3 Large**：8B 参数，需高端 GPU
- **SD3 Large Turbo**：Large 的蒸馏版，4 步推理

所有变体共享同一架构，仅深度和宽度不同。

---

## 3. 核心架构

### 3.1 Denoiser：MMDiT（Multi-Modal Diffusion Transformer）

SD3 的核心创新是用 MMDiT 替代传统 U-Net。MMDiT 是一种**双流 transformer**：
- **Image stream**：latent 经 patchify（patch_size=2）后得到 image tokens，通过独立的 QKV 投影和 MLP 处理
- **Text stream**：text embeddings（来自 CLIP/T5）通过另一组 QKV 投影和 MLP 处理
- **Joint attention**：image tokens 和 text tokens 拼接后进入统一的注意力计算——所有 token 互相 attend（full attention，无 causal mask，与 LLM 的 causal/KV-cache 范式完全不同）
- 双流的调制参数（scale/shift/gate）由各自的 adaLN 模块根据 timestep embedding 独立生成

**为什么双流**：text token 和 image token 的语义空间不同，独立投影后再 joint attention 比直接拼接后统一投影效果好。这是 SD3 区别于 DiT（Peebles & Xie, 2023）的核心改进。

### 3.2 Latent 表示

- **VAE 编码器**：8× 空间下采样，16 通道 latent（与 SDXL 相同的 VAE，非 SD1.x/2.x 的 4 通道 VAE）
- **Latent shape**：对于 W×W 的方形输入图像，latent 大小为 `(B, 16, W/8, W/8)`
- **对比记忆**：16 通道 VAE 的 latent 容量是 4 通道的 4 倍，意味着 denoiser 在相同分辨率下有更多信息可处理。这也是为什么 SD3 比 SD1.5/SDXL 细节更好的原因之一。

### 3.3 Text Conditioning

**三路 text encoder**（SD3 Large 全用，Medium 可省略 T5）：

| Encoder | Hidden Dim | Max Length | 权重（SD3 Large） | 备注 |
|---------|-----------|------------|-------------------|------|
| CLIP-L (ViT-L/14) | 768d | 77 tokens | ~430M | 标准 CLIP |
| CLIP-G (ViT-bigG/14) | 1280d | 77 tokens | ~1.8B | 更大 CLIP，仅 SD3 用 |
| T5-XXL | 4096d | 77~512 tokens | ~11B | 最强长文本理解，但显存昂贵 |

**合并方式**：CLIP-L 和 CLIP-G 的输出在 token 维度拼接（concatenation），然后与 T5 输出一起送入 MMDiT。text stream 中不同 encoder 输出的 token 会被赋予不同的"encoder type embedding"以区分来源。

**Pooled embedding**：CLIP-G 的 pooled output（1280d）作为全局条件注入 adaLN，控制整体风格和构图。

### 3.4 Timestep / Sigma Conditioning

SD3 使用 **rectified flow** 的 timestep 参数化：
- 时间 `t ∈ [0, 1]`，`t=1` 为纯噪声，`t=0` 为目标数据
- Timestep `t` 经 Fourier features 编码后变为 256 维 embedding
- 该 embedding 经 2 层 MLP（SiLU 激活）投影到 adaLN 所需的 6 组调制参数：`shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp`
- 这 6 组参数分别调制 attention 和 FFN 的 LayerNorm + 残差路径

**与 DDPM sigma 参数化的区别**：SD3 的"时间"是 rectified flow 中的 `t`（线性插值参数），不是 DDPM 的扩散步数 `t=0..1000`。在推理时，`t` 从 1 到 0 线性递减。

### 3.5 Attention 结构

- **类型**：full attention（所有 token 互相 attend），无 GQA、无 causal mask、无 KV cache
- **Joint attention**：text tokens（~154 个）+ image tokens（~4096 个 for 1024px）→ 总计 ~4250 tokens，全部参与 QKV 计算
- **FlashAttention 兼容**：MMDiT 的注意力计算与 FlashAttention 2/3 兼容，实际推理中通常通过 `xformers` 或 `torch.nn.functional.scaled_dot_product_attention` 加速
- **无 RoPE**：SD3 不在 attention 中使用 Rotary Position Embedding——位置信息由 2D sinusoidal position embedding 在 patchify 阶段注入

### 3.6 VAE

- **来源**：沿袭 SDXL 的 16 通道 VAE
- **下采样因子**：8×（像素空间 → latent 空间）
- **Decoder 缩放因子**：latent 需乘以 `scaling_factor ≈ 0.13025` 后送入 decoder（VAE 训练时的归一化常数）
- **支持 tiling**：对大分辨率图像（2048px+），VAE decoder 可分块解码以避免 OOM

---

## 4. 推理数据流

### 完整路径

```
prompt
  │
  ├─→ tokenizer → CLIP-L (768d, 77 tokens)
  ├─→ tokenizer → CLIP-G (1280d, 77 tokens)        [Large 专有]
  └─→ tokenizer → T5-XXL (4096d, 77~512 tokens)    [可选]
  │
  ▼
text embeddings (concat, 按 encoder type embedding 标记)
  │
  ▼
noise latent z₁ ~ N(0, I)   shape: (1, 16, H/8, W/8)
  │
  ▼
denoising loop: t = 1 → 0 (28~50 步)
  ├─ conditional forward: MMDiT(z_t, text, t) → v_cond
  ├─ unconditional forward: MMDiT(z_t, ∅, t) → v_uncond
  ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)
  │         where s ∈ [3.0, 7.0]（典型值 4.5）
  └─ scheduler step: z_{t − Δt} = z_t + (t_{next} − t) · v_cfg
  │
  ▼
z₀ → multiply by 1/scaling_factor → VAE decoder → pixel image [0, 1]
```

### 关键步骤说明

1. **Text encoding**：在推理开始时一次性完成，text embeddings 在整个 denoising loop 中复用。这是最重要的 cache 机会（详见第 6 节）。
2. **CFG 插入位置**：在 **vector field 层面**做 CFG，不是 noise prediction 层面，也不是 latent 层面。每次 denoising step 需要**两次完整 forward**（conditional + unconditional），这是扩散推理的显存峰值时刻。
3. **Scheduler**：rectified flow 的 Euler step（一阶 ODE），`t` 从 1 到 0 线性递减。步数通常 28（SD3 Medium）或 50（SD3 Large）。更少步数会导致质量下降，更多步数收益递减。
4. **VAE decode**：最终一步，latent `z₀` 先除以 `scaling_factor`（反归一化），再送入 VAE decoder 得到像素空间输出。

---

## 5. 关键 Tensor Shape

### 5.1 输入输出 Shape

| 名称 | Shape | 说明 |
|------|-------|------|
| 输入图像 | `(B, 3, H, W)` | 如 `(1, 3, 1024, 1024)` |
| **Latent（VAE 编码后）** | `(B, 16, H/8, W/8)` | 如 1024px → `(1, 16, 128, 128)` |
| **Latent（512px）** | `(1, 16, 64, 64)` | 512×512 输入 |
| **Latent（2048px）** | `(1, 16, 256, 256)` | 2048×2048 输入，显存压力大 |
| VAE 输出 | `(B, 3, H, W)` | 像素空间，[-1, 1] → [0, 255] |

### 5.2 Patch / Token Shape

| 名称 | Shape | 说明 |
|------|-------|------|
| Patch size | `p = 2` | 空间上每 2×2 latent pixel 为 1 个 patch |
| Image tokens | `(B, N_img, D)` | N_img = (H/8/2)²，1024px → (128/2)² = 4096 tokens |
| Image tokens（512px） | `(1, 1024, D)` | 64/2=32，32²=1024 |
| Image tokens（2048px） | `(1, 16384, D)` | 256/2=128，128²=16384（attention O(n²) 严重） |

### 5.3 Text Embedding Shape

| 名称 | Shape | 说明 |
|------|-------|------|
| CLIP-L tokens | `(B, 77, 768)` | 固定长度 77，padding 填充 |
| CLIP-G tokens | `(B, 77, 1280)` | 固定长度 77 |
| T5-XXL tokens | `(B, L_t5, 4096)` | L_t5 = 77~512（可变），取决于 prompt 长度 |
| Pooled text（CLIP-G） | `(B, 1280)` | 全局语义向量，注入 adaLN |
| 合并后 text tokens | `(B, L_text, D_text)` | L_text = 77+77+L_t5，按 encoder type 拼接 |

### 5.4 Timestep Embedding Shape

| 名称 | Shape | 说明 |
|------|-------|------|
| Raw timestep | `(B, 1)` 或 `(B,)` | t ∈ [0, 1] |
| Fourier encoded | `(B, 256)` | 256 维正弦/余弦编码 |
| MLP 输出（adaLN 参数） | `(B, 6·D)` | 6 组 (shift, scale, gate)，每组 D 维（D = hidden_dim） |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

SD3 的显存瓶颈按严重程度排序：

| 排序 | 组件 | VRAM（fp16, 1024px） | 说明 |
|------|------|---------------------|------|
| 🔴 1 | T5-XXL text encoder | ~11 GB（单独加载） | 最大单组件，但可 omit（Medium） |
| 🟡 2 | MMDiT backbone (Large) | ~16 GB（8B 参数量） | fp16 权重 ~16GB |
| 🟢 3 | MMDiT backbone (Medium) | ~4-5 GB | 2B 参数量，12GB 友好 |
| 🟡 4 | Attention activations | ~3-4 GB/step | 4096² × 2B × layers |
| 🟢 5 | CLIP encoders | ~4.6 GB（两个 CLIP） | 加载后可 offload |
| 🟢 6 | VAE decoder | ~2 GB（峰值） | 支持 tiling 进一步降低 |

### 6.2 哪些可以 Cache

| 可 Cache | 为什么 | 收益 |
|---------|--------|------|
| **Text embeddings**（CLIP-L + CLIP-G + T5） | 推理开始时 encode 一次，整个 denoising loop 复用 | 消除 text encoder 的每步重算 |
| **Unconditional embedding** | 若使用空 prompt（""），其 text embedding 固定不变 | 减少一次 text encode |
| **Timestep embedding** | Fourier encoding 是确定性的，可预计算所有 t 的 embedding | 微小的性能提升 |
| VAE encoder 输出 | 仅 image-to-image 场景有用（文生图从噪声开始，不经过 VAE encoder） | 无（文生图场景） |

### 6.3 哪些不能 Cache

| 不能 Cache | 为什么 | 与 LLM 的差异 |
|-----------|--------|--------------|
| **Denoiser 内部 K/V 矩阵** | 每步 latent 都不同（被 scheduler 更新），上一步的 K/V 对下一步**完全无用** | LLM 的自回归生成中 K/V 随 token 追加而累积；扩散每步"重写全部 latent" |
| **Latent buffer** | 每步被 scheduler 原地覆盖更新 | — |

### 6.4 Buffer 复用策略

- **Latent**：可原地更新（不需要 ping-pong），直接 `z = z + dt * v` 覆盖
- **Text embeddings**：加载后常驻显存，不释放
- **Attention activation**：若使用 activation checkpointing，每层激活在 backward 时重算（forward 时释放），减少峰值显存约 40%，但增加 ~30% 计算时间

### 6.5 12GB RTX 5070 Ti 可行性判断

| 变体 | 判断 | VRAM 预估 | 推荐配置 |
|------|------|----------|---------|
| **SD3 Medium（no-T5）** | 🟢 适合 | ~4-5 GB | `--model stabilityai/stable-diffusion-3.5-medium --no_t5 --dtype fp16 --enable_model_cpu_offload` |
| SD3 Medium（含 T5） | 🟡 极限可跑 | ~9-10 GB | 需开 sequential offload，T5 占 ~5GB |
| SD3 Large（含 T5） | 🔴 不适合 | ~22 GB+ | 8B 权重 + T5 已超 12GB，即使 offload 也需频繁换页 |
| SD3 Large Turbo（4 步，含 T5） | 🟡 极限可跑 | ~12 GB（边界） | 4 步推理缩短了 peak time，但权重加载仍受限 |

**推荐 12GB fallback 命令**：

```bash
# 最稳妥：SD3 Medium，去 T5，开 offload
python -c "
from diffusers import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    'stabilityai/stable-diffusion-3.5-medium',
    torch_dtype=torch.float16
)
pipe = pipe.to('cuda')
pipe.enable_model_cpu_offload()
image = pipe('一只柴犬在樱花树下', num_inference_steps=28, guidance_scale=4.5).images[0]
image.save('output.png')
"
# VRAM ≈ 4.3 GB（实测社区报告）
```

**HF Model Card**：
- SD3.5 Medium：`https://huggingface.co/stabilityai/stable-diffusion-3.5-medium`
- SD3.5 Large：`https://huggingface.co/stabilityai/stable-diffusion-3.5-large`
- SD3.5 Large Turbo：`https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo`

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `scheduler.py`
- 已实现 `RectifiedFlowScheduler`（T10），与 SD3 的 Euler rectified flow step 一致：`z_next = z_t + dt * v`
- SD3 的 timestep 序列使用对数间距更优（低频步更多，高频步更少），当前 scheduler 使用线性间距。需在 T18 总结时评估是否添加对数间距选项。

### 7.2 `attention.py`
- MMDiT 的 **joint attention**（text token + image token 统一 QKV）是核心参考。当前 T11 实现的 `JointAttention` 是 toy 简化版（拼接后 unified attention），距离真正的双流 MMDiT 有差距
- 真正的 MMDiT 需要：text stream 和 image stream 分别有独立的 pre-attention LayerNorm + QKV 投影，然后在 attention 中统一计算，最后分别输出
- T18 总结时需明确标注"MMDiT 双流 vs toy 拼接式 joint attention"的差异

### 7.3 `dit.py`
- SD3 的 `patch_size=2` 是工业实际值（不是 toy 的 4 或 8），我们的 TinyDiT 应该保持 patch_size=2
- SD3 使用可学习的位置编码（`nn.Parameter`）而非 2D sinusoidal——这与 T11 TinyDiT 的选择一致（都是 toy 简化）
- 真实 SD3 的 patchify 还注入了 2D sin/cos position embedding，TinyDiT 暂未实现

### 7.4 `text_conditioning.py`（T12 待实现）
- **三路 encoder 合并**是关键挑战：CLIP-L (768d) + CLIP-G (1280d) + T5 (4096d) 的维度不同，需要在送入 DiT 前统一到 hidden_dim
- 我们的 v1 text conditioning 可以先用**单一 encoder 的 last hidden**（如 CLIP-ViT-L/14 的 768d），后续扩展多 encoder 接口
- Pooled embedding 注入 adaLN 的路径需在 `transformer_block.py` 中预留接口

### 7.5 `pipeline.py`（T12 待实现）
- CFG 插入位置：在 **vector field 层面**（`v_cfg = v_uncond + s * (v_cond - v_uncond)`），不是 noise prediction 层面
- Denoising loop 结构：一次 text encode → N 步 denoising → VAE decode。Text encoding 只需一次
- 双 forward（cond + uncond）是每步的显存峰值，T16 的 CFG batching 实验应探索"合并 batch"方式减少显存

### 7.6 `memory_manager.py`（T12 待实现）
- Text embedding 固定不变 → 可作为"静态 buffer"管理
- Latent buffer 每步更新 → "动态 buffer"
- 不需要 KV cache → 与 LLM 的 memory manager 设计完全不同

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 官方 repo URL：`https://github.com/Stability-AI/generative-models`
- arXiv URL：`https://arxiv.org/abs/2403.03206`
- HF model card（Medium）：`https://huggingface.co/stabilityai/stable-diffusion-3.5-medium`
- HF model card（Large）：`https://huggingface.co/stabilityai/stable-diffusion-3.5-large`
- HF model card（Large Turbo）：`https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo`
- Stability AI 官方 blog：`https://stability.ai/news/stable-diffusion-3`
- Diffusers pipeline 源码：`diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`

**读**：
- Section 2（Rectified Flow formulation）：理解 RF 的数学定义和与 score-based 的区别
- Section 3（MMDiT architecture）：重点看双流设计、joint attention 的 QKV 投影方式、adaLN 调制路径
- Section 4（Scaling study）：了解模型大小（Medium 2B vs Large 8B）和推理成本的关系
- Appendix 中的 inference 细节：sampling steps、CFG guidance scale 推荐值、scheduler choice
- Diffusers pipeline 源码的 `__call__` 方法：实际推理循环的代码结构

**输出**：
- 本文档：`learning/papers/01_scaling_rectified_flow_transformers_sd3.md`（8 字段完整 + 12GB 判断）
- 不要求产出 HTML（HTML 留给 T8）
- 本文档将作为 T13（reference image inference 脚手架）和 T18（最终报告）的 SD3 部分输入

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T7 (Wave 2)*
