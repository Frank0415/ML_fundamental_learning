# 07 — CogVideoX：Expert Transformer 文生视频

> **模型名称**：CogVideoX（智谱 AI）
> **官方仓库**：[github.com/THUDM/CogVideo](https://github.com/THUDM/CogVideo)
> **HF Model Card (2B)**：[huggingface.co/THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b)
> **HF Model Card (5B)**：[huggingface.co/THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
> **分类**：文生视频 + image-to-video — 最易尝试的视频模型
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

CogVideoX-2B 是 中等显存配置 上**最现实的开源视频模型**。仅 2B 参数，Apache 2.0 许可，官方最低要求仅 4GB 显存。它与 Wan2.1-1.3B 和 LTX-Video 2B 并列 开源小型视频推理的代表组合。在架构上，CogVideoX 采用了独特的 **3D Causal VAE + Expert Transformer** 设计，其 3D causal attention 虽然限制了并行性但提供了更稳定的帧间一致性。对于 `diffusion_engine` 项目，CogVideoX 是 video reference inference 的首选。

---

## 2. 模型类型

**文生视频（text-to-video）+ image-to-video**。CogVideoX 系列有两个尺寸：

| 变体 | 参数量 | 最低 VRAM | 典型帧数/分辨率 | 许可 | 资源档位 |
|------|--------|----------|---------------|------|-----------|
| **CogVideoX-2B** | 2B | **4 GB** | 49f × 720×480 | Apache 2.0 | 🟢 非常舒适 |
| CogVideoX-5B | 5B | ~10 GB | 49f × 720×480 | Apache 2.0（？） | 🟡 极限可跑 |

---

## 3. 核心架构

### 3.1 Denoiser：Expert Transformer

CogVideoX 的论文标题中的 "Expert Transformer" 指的是为视频专门设计的 transformer，而非传统 MoE（Mixture of Experts）。核心设计特点：

- **3D attention**：attention 在 `(T, H, W)` 三个维度上计算，所有 spacetime patches 参与
- **Causal in time**：时间维度上是 causal 的——当生成第 t 帧时，只能 attend 第 1~t 帧和当前帧的 patches，不能"看到"未来的帧
- **空间维度 full attention**：每个帧内部的所有 patches 互 attend（non-causal）
- **Attention 公式**：QK^T 计算后，时间维度超过当前帧的那些位置被 mask 掉（设置为 -inf）

**3D Causal Attention 的含义**（推理视角）：

```
对于第 t 帧的某个 patch：
  - 可以 attend：帧 1 ~ t 的所有 patches（T×H×W 的子集）
  - 不能 attend：帧 t+1 ~ T 的 patches
  - 这保证了逐帧生成的连贯性——生成当前帧时不会"偷看"未来帧
```

**与 Wan 的区别**：Wan 使用 full attention（所有帧互相 attend，无 causal mask）。CogVideoX 的 causal attention 在理论上有两个优势：① 更稳定的帧间过渡（因为每帧生成时只能基于已生成的帧）；② 在推理时，可以逐帧生成而非一次性生成所有帧（虽然实践中通常仍是一次性 denoising）。

### 3.2 3D Causal VAE

CogVideoX 使用自训练的 **3D Causal VAE**：
- **空间压缩**：8×
- **时间压缩**：4×
- **通道数**：4（与 SD1.x VAE 相同，容量低于 SD3/FLUX 的 16ch）
- **Causal 特性**：VAE 的 3D 卷积也是 causal 的（时间维度上只向前看）

**与 Wan VAE 的对比**：
| 维度 | CogVideoX VAE | Wan2.1 VAE |
|------|-------------|-----------|
| 通道数 | 4 | **16** |
| Causal | 是（时间维度） | 否 |
| 空间压缩 | 8× | 8× |
| 时间压缩 | 4× | 4× |

CogVideoX 选择 4 通道 VAE 而非 16 通道是为了降低计算量——patch 展平后的维度 = p²C = 2²×4 = 16（vs Wan 的 2²×16 = 64），每个 token 输入维度更小，transformer 的 hidden dim 可以更小，参数量更低。

### 3.3 Expert Adaptive LayerNorm

CogVideoX 引入了 **expert adaptive LayerNorm**，这是一种条件注入机制，类似于 SD3/MMDiT 中的 AdaLN，但针对视频做了适配：
- Timestep embedding → 生成 scale/shift/gate
- 不同的 transformer 层可能使用不同的"expert"调制参数
- 这类似于 MoE 的"不同专家处理不同情况"思想，但应用在 LayerNorm 层面而非 FFN 层面

### 3.4 Progressive Training

CogVideoX 采用 progressive training 策略——训练过程不是一次性在所有分辨率/帧数上进行，而是由低到高逐步增加：
1. 先在低分辨率低帧数上训练（如 256×256, 17f）
2. 逐步增加到中等规格（如 480×480, 33f）
3. 最终在高规格上 fine-tune（如 720×480, 49f）

**对推理的影响**：这解释了为什么 CogVideoX-2B 在不同规格下的表现不一——它的训练分布决定了推理时的推荐参数。

### 3.5 Multi-Resolution Frame Pack

CogVideoX 支持在训练时使用不同宽高比和不同帧数的视频，并将不同分辨率的视频"pack"到同一个 batch 中（通过 padding 或 dynamic batch）。这使得模型在推理时对不同的输入规格有更好的泛化能力。

---

## 4. 推理数据流

```
prompt ("一只猫在草地上奔跑")
   │
   └─→ T5 tokenizer → T5-XXL (4096d) → text embeddings
   │
   ▼
noise latent z_T ~ N(0, I)  shape: (1, 4, T_latent, H_latent, W_latent)
   典型：49f×720×480 → latent (1, 4, 13, 90, 60)
   │
   ▼
denoising loop（50 步）
   ├─ patchify: (1,4,13,90,60) → p=(1,2,2) → N_st = 13 × 45 × 30 = 17,550
   ├─ Causal attention: 每帧只能看 ≤ 当前帧
   ├─ Expert Adaptive LayerNorm: timestep → scale/shift
   ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)   s ≈ 6.0
   └─ Euler step
   │
   ▼
3D VAE decoder: (1, 4, 13, 90, 60) → (1, 3, 49, 720, 480)
```

**Causal mask 的实际效果**：对于 17,550 tokens，causal mask 将注意力计算限制在"当前帧及之前的 tokens"范围内，这意味着实际参与计算的 token 数约等于 `(总 tokens) / 2`（平均而言）。这比 full attention 节省了约 50% 的 attention 计算量，但并不改变 O(n²) 的复杂度。

---

## 5. 关键 Tensor Shape

### 5.1 不同规格下的 Token 数

| 规格 | 帧数 | 分辨率 | Latent Shape (C=4) | Spacetime Tokens |
|------|------|--------|---------------------|-----------------|
| **最小** | 17 | 256×256 | `(4, 5, 32, 32)` | `5 × 16 × 16 = 1,280` |
| **推荐** | 49 | 480×720 | `(4, 13, 60, 90)` | `13 × 30 × 45 = 17,550` |
| **中等显存极限** | 49 | 576×1024 | `(4, 13, 72, 128)` | `13 × 36 × 64 = 29,952` |

### 5.2 与同类模型 Token 数对比（相似帧数和分辨率）

| 模型 | 帧数 | 分辨率 | C | Latent Shape | Tokens |
|------|------|--------|---|-------------|--------|
| **CogVideoX-2B** | 49 | 480×720 | 4 | `(4, 13, 60, 90)` | 17,550 |
| **Wan2.1-1.3B** | 33 | 480×832 | 16 | `(16, 9, 60, 104)` | 14,040 |
| **LTX-Video 2B** | 121 | 480×720 | 4 | `(4, 16, 60, 90)` | ~8,640 |

CogVideoX 的 token 数处于中等水平。因为 C=4（VAE 通道少），patch 展平维度小，transformer hidden dim 小，总体显存需求低。

### 5.3 Text Embedding

| 名称 | Shape | 说明 |
|------|-------|------|
| T5 tokens | `(1, 226, 4096)` | CogVideoX 默认 max_length=226 |
| 投影后 | `(1, 226, D_dit)` | D_dit 约 1920（2B 模型） |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

| 排序 | 组件 | VRAM（fp16, 49f×480×720） | 说明 |
|------|------|--------------------------|------|
| 🟡 1 | T5-XXL | ~5 GB | 常驻 |
| 🟡 2 | DiT attention activations | ~2-3 GB/step | 17.5K tokens, causal attn 减少 ~50% |
| 🟢 3 | DiT 权重（2B） | ~4 GB | fp16 参数 |
| 🟢 4 | 3D VAE decoder | ~1-2 GB | |

### 6.2 资源档位与运行边界

| 配置 | 判断 | VRAM 估算 | 说明 |
|------|------|----------|------|
| **CogVideoX-2B, 49f×480p** | 🟢 适合 | ~8-9 GB | 官方最低 4GB，在中等显存配置下仍较从容 |
| **CogVideoX-2B, 49f×576p** | 🟡 极限可跑 | ~10-11 GB | 边界，需 offload |
| CogVideoX-5B, 49f×480p | 🟡 极限可跑 | ~10-12 GB | 5B 权重 ~10GB，需 offload |

**一个受限显存示例命令**：

```bash
# CogVideoX-2B：最舒适
python -c "
from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained(
    'THUDM/CogVideoX-2b',
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
video = pipe(
    '一只猫在草地上奔跑',
    num_frames=49,
    width=720,
    height=480,
    num_inference_steps=50,
    guidance_scale=6.0
).frames[0]
# VRAM ≈ 8.5 GB
"

# 降帧数（更稳）
video = pipe(prompt, num_frames=25, width=576, height=384).frames[0]
# VRAM ≈ 5-6 GB
```

**HF Model Card**：
- 2B：`https://huggingface.co/THUDM/CogVideoX-2b`
- 5B：`https://huggingface.co/THUDM/CogVideoX-5b`

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `attention.py`
- **Causal attention with 3D mask** 是一个重要的变体。当前 `SelfAttention` 只支持 2D full attention。CogVideoX 的 causal mask 需要根据 3D spacetime 坐标判断"哪些 tokens 是当前帧及之前的"。
- Causal attention 的 mask 计算：对于 spacetime token 位置 `(t_i, y_i, x_i)` 和 `(t_j, y_j, x_j)`，如果 t_j > t_i，则 mask=False。

### 7.2 `dit.py`
- 3D patchify 需要支持 `patch_size = (1, 2, 2)` 的三维参数（时间 patch=1，空间 patch=2）。
- Patch 展平维度 = `t_p × h_p × w_p × C`，其中 C 取决于 VAE。

### 7.3 `pipeline.py`
- CogVideoX 的 denoising loop 50 步明显高于图像（28 步），意味着视频推理的 wall time 是图像推理的 ~2 倍（相同参数规模下）。
- CFG scale=6.0 高于文生图的 4.5，意味着视频需要更强的文本引导（每步 cond+uncond 双 forward 不可避免）。

### 7.4 `vae_stub.py`
- C=4 的 3D VAE 比 C=16 的 VAE 更轻量，对 中等显存配置 更友好。在 vae_stub 中应预留 "channels" 参数以支持不同 VAE。

### 7.5 `memory_manager.py`
- Causal attention 的显存预估不同于 full attention：平均约 50% 的 token 参与实际 attention（因为有 causal mask），但存储的 activation 矩阵仍然是 n×n（只是 masked 部分为 0）。如果使用 FlashAttention 或 xformers 的 block-sparse 实现，可以真正节省显存。

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 官方 GitHub：`https://github.com/THUDM/CogVideo`
- HF 2B：`https://huggingface.co/THUDM/CogVideoX-2b`
- HF 5B：`https://huggingface.co/THUDM/CogVideoX-5b`
- arXiv：搜索 "CogVideoX text-to-video expert transformer"

**读**：
- Paper 的 architecture section（3D causal VAE、expert transformer、causal attention 机制）
- GitHub 推理脚本中的参数和 VRAM 建议
- Diffusers pipeline 源码：`diffusers/pipelines/cogvideo/pipeline_cogvideox.py`

**输出**：
- 本文档：`learning/papers/07_cogvideox.md`（8 字段完整 + causal attention 对比 + 资源档位判断）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
