# 05 - Wan2.1：开源大规模视频生成模型

> **模型名称**：Wan2.1（阿里通义万相）
> **官方仓库**：[github.com/Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
> **HF Model Card (T2V-1.3B)**：[huggingface.co/Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
> **HF Model Card (T2V-14B)**：[huggingface.co/Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)
> **分类**：文生视频 + image-to-video - 开源视频 DiT
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

Wan2.1 是阿里开源的视频生成模型，公开了完整的 3D VAE 和 DiT 推理代码。它的 **Wan2.1-T2V-1.3B** 变体（仅 1.3B 参数）是 中等显存配置 上可以跑通的实用视频模型之一，官方报告 480p 视频在 8.19GB 显存即可运行。对比 SD3 Medium 的 2B 文生图模型，1.3B 视频 DiT 能在相近参数规模下生成有意义的视频，这证明了 DiT 架构在视频模态上的参数效率。理解 Wan 的推理路径是完成 video reference inference 的关键技术准备。

---

## 2. 模型类型

**文生视频（text-to-video）+ image-to-video**。Wan2.1 系列包含：

| 变体 | 参数量 | 定位 | 资源档位 |
|------|--------|------|-----------|
| **Wan2.1-T2V-1.3B** | 1.3B | 轻量快速，480p 可跑 | 🟢 适合（~8.2 GB） |
| **Wan2.1-T2V-14B** | 14B | 高质量大模型 | 🔴 不适合（权重就 ~28GB fp16） |
| Wan2.1-I2V-14B | 14B | image-to-video | 🔴 不适合 |

**注意**：Wan2.1-1.3B 可以直接在 HuggingFace 上获取，协议为 Apache 2.0（部分组件可能有不同许可）。推理需要 Diffusers 库支持（`diffusers>=0.31.0`）。

---

## 3. 核心架构

### 3.1 Denoiser：Video DiT（3D Full Attention）

Wan2.1 的 denoiser 是 **Video DiT**（基于 DiT 的视频版本）：
- **Full attention over all spacetime tokens**：所有帧的所有 patches 互相 attend（非 causal）
- **Patch size**：`(1, 2, 2)`，时间维度上每 1 帧一个 patch（不做时间 patch），空间维度上每 2×2 latent pixel 一个 patch
- **Transformer block 结构**：标准的 AdaLN + Self-Attention + FFN

由于没有 causal mask（不限制当前帧只能向前看），Wan 的 attention 是"并行"的，所有 tokens 同时 attend。这保证了全局一致性，但也意味着 token 数 = `T_latent × (H_latent/2) × (W_latent/2)`，总 token 数随帧数增加而线性增长。

### 3.2 Latent 表示：3D VAE

Wan2.1 使用自训练的 **3D VAE**（公开可下载）：

- **空间压缩**：8×（如 480×832 像素 → 60×104 latent）
- **时间压缩**：4×（如 81 帧 → 21 帧 latent）
- **通道数**：**16**（与 SD3/FLUX VAE 相同，是 4 通道的两倍容量）

**Wan 的 3D VAE 是一个关键公开资产**：它不仅是 Wan 自己使用，也被其他项目和研究者分析和复用。对于 `diffusion_engine` 项目，Wan 的 3D VAE 提供了 video VAE 接口设计的具体参考。

### 3.3 Text Conditioning

Wan2.1 使用 **T5 作为 text encoder**（推测为 T5-XXL，即 11B 参数版本），通过 cross-attention 注入 DiT。具体方式：
- T5 的 last hidden states 作为 text tokens
- Text tokens 通过 cross-attention 注入每层 DiT block（text tokens 作为 K/V，image tokens 作为 Q）
- 另外，T5 的 pooled output（或特殊设计的全局条件向量）通过 adaLN 注入

**对受限显存配置的影响**：T5-XXL 约占 5GB fp16，在 1.3B DiT + T5 组合下总显存约 8.2GB，这意味着即使全加载，中等显存配置 也是舒适的。

### 3.4 Timestep / Sigma Conditioning

- 使用 **Flow Matching** 框架（与 rectified flow 类似）：`t ∈ [0, 1]`
- Fourier features + MLP → adaLN 调制参数（shift, scale, gate）
- 与 SD3/FLUX 的 adaLN 路径一致

### 3.5 Attention 结构

| 特性 | Wan2.1 (1.3B) | Wan2.1 (14B) |
|------|--------------|-------------|
| Attention 类型 | Full attention (all spacetime) | Full attention |
| Causal mask | 无（所有 tokens 互 attend） | 无 |
| Token 数（81f×480p） | ~6,500 | ~6,500 |
| FlashAttention 兼容 | 是（通过 xformers/SDPA） | 是 |

### 3.6 VAE

- **类型**：3D VAE（因果卷积？非因果？，官方文档未明确因果性，但从实现看可能是非因果的 3D Conv）
- **压缩因子**：时间 4×，空间 8×
- **Decoder**：一次性 decode 所有帧（不是逐帧 decode），意味着 VAE decoder 的显存随帧数线性增长
- **支持 tiling**：对于超长视频，社区已实现分块 decode 方案

---

## 4. 推理数据流

```
prompt ("一只柴犬在草地上奔跑，阳光明媚")
   │
   └─→ T5 tokenizer → T5-XXL (4096d, 可变长度) → text embeddings
   │
   ▼
noise latent z_T ~ N(0, I)  shape: (1, 16, T_latent, H_latent, W_latent)
   1.3B 典型规格：81f×480×832px → latent (1, 16, 21, 60, 104)
   │
   ▼
denoising loop: t = 1 → 0（通常 50 步）
   ├─ patchify: (1,16,21,60,104) → (1, N_st, D)    N_st = 21 × 30 × 52 = 32,760
   │   注：p=2 空间 patch（H_latent/2=30, W_latent/2=52），p=1 时间 patch（T_latent/1=21）
   ├─ DiT forward: AdaLN(attn(patches + timestep_emb) + cross_attn(patches, text)) + FFN
   ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)     s ≈ 5.0~7.0
   └─ Flow Match Euler step: z_{t-Δt} = z_t + Δt · v_cfg
   │
   ▼
3D VAE decoder: (1, 16, 21, 60, 104) → (1, 3, 81, 480, 832)
```

**关键的 Token 数计算**（以 1.3B 规格为例）：
- 原始：81 帧 × 480×832 RGB → VAE 编码 → latent `(16, 21, 60, 104)`
- Patchify：时间 p_t=1, 空间 p_h=p_w=2 → N_st = 21 × 30 × 52 = **32,760 tokens**
- Full attention 矩阵：32,760² ≈ 1.07B 元素 → fp16 约 2.1 GB per attention layer
- 对于 1.3B 模型（~30 层 DiT），attention 中间激活的内存需求非常大

**这就是为什么 1.3B 模型在 480p 上"刚好"能跑**，32K tokens 的 full attention 已经接近了 中等显存配置的上限。如果升到 720p，token 数再翻倍（~131K），attention 就完全不可行了。

---

## 5. 关键 Tensor Shape

### 5.1 不同规格下的 Latent Shape

| 规格 | 帧数 | 分辨率 | 原始像素 Shape | Latent Shape | Spacetime Tokens (p=(1,2,2)) |
|------|------|--------|---------------|-------------|-------------------------------|
| 最小规格 | 17 | 256×256 | `(3, 17, 256, 256)` | `(16, 5, 32, 32)` | `5×16×16 = 1280` |
| 推荐规格 | 33 | 480×832 | `(3, 33, 480, 832)` | `(16, 9, 60, 104)` | `9×30×52 = 14,040` |
| 标准规格 | 81 | 480×832 | `(3, 81, 480, 832)` | `(16, 21, 60, 104)` | `21×30×52 = 32,760` |
| 高质量 | 81 | 720×1280 | `(3, 81, 720, 1280)` | `(16, 21, 90, 160)` | `21×45×80 = 75,600` |

### 5.2 显存对应关系

| 规格 | Token 数 | Full Attn 矩阵（fp16） | 估算总 VRAM（1.3B+T5） |
|------|---------|---------------------|----------------------|
| 1280 tokens | 1,280 | ~3.3 MB | ~4 GB |
| 14,040 tokens | 14,040 | ~394 MB | ~6 GB |
| 32,760 tokens | 32,760 | ~2.1 GB | ~8.2 GB |
| 75,600 tokens | 75,600 | ~11.4 GB | ~16 GB（超 中等显存配置） |

### 5.3 Text Embedding Shape

| 名称 | Shape | 说明 |
|------|-------|------|
| T5 tokens | `(1, L_text, 4096)` | L_text 可变（通常 128~256） |
| Cross-attn 投影后 | `(1, L_text, D_dit)` | D_dit 取决于 Wan 版本（1.3B 可能 ~1152） |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

| 排序 | 组件 | VRAM（81f×480p, fp16） | 说明 |
|------|------|---------------------|------|
| 🔴 1 | DiT full attention activations | ~4-5 GB per step | 32K tokens 的 O(n²) attention |
| 🟡 2 | T5-XXL text encoder | ~5 GB | 常驻显存 |
| 🟢 3 | DiT 权重（1.3B） | ~2.6 GB | fp16 参数 |
| 🟡 4 | 3D VAE decoder | ~2-3 GB（峰值） | 一次性 decode 81 帧 |

### 6.2 哪些可以 Cache

| 可 Cache | 收益 |
|---------|------|
| **T5 text embeddings** | 一次 encode，全程复用（~5GB 常驻但只 load 一次） |
| **Timestep embeddings** | Fourier encoding 确定，可预计算 |
| **Image-to-video 初始 latent** | 输入图像经 VAE encoder 的 latent 可 cache |

### 6.3 哪些不能 Cache

- **Denoiser K/V**：每步 latent 全刷新（diffusion 本质特征）
- **Attention activation**：每步 attention pattern 随 latent 变化而变化

### 6.4 资源档位与运行边界

| 变体 + 规格 | 判断 | VRAM 估算 | 备注 |
|-------------|------|----------|------|
| **Wan2.1-1.3B, 33f×480p** | 🟢 适合 | ~6-7 GB | 降帧数，最稳妥 |
| **Wan2.1-1.3B, 81f×480p** | 🟡 极限可跑 | ~8-10 GB | 标准配置，需开 offload |
| Wan2.1-1.3B, 81f×720p | 🔴 不适合 | ~14-16 GB | attention 显存爆炸 |
| Wan2.1-14B | 🔴 不适合 | ~30+ GB | 权重就超了 |

**一个受限显存示例命令**：

```bash
# 最稳妥：1.3B + 33 帧 + 480p + offload
python -c "
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained(
    'Wan-AI/Wan2.1-T2V-1.3B',
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
video = pipe(
    '一只柴犬在草地上奔跑',
    num_frames=33,
    width=832,
    height=480,
    num_inference_steps=50,
    guidance_scale=5.0
).frames[0]
# VRAM ≈ 6.5 GB（实测社区报告）
"

# 标准规格（可能接近上限）：81 帧 + 480p + sequential offload
pipe.enable_sequential_cpu_offload()
# VRAM ≈ 8.2 GB（官方数据）
```

**HF Model Card**：
- T2V-1.3B：`https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B`
- T2V-14B：`https://huggingface.co/Wan-AI/Wan2.1-T2V-14B`

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `dit.py`
- 3D patchify：当前只支持 2D `(B,C,H,W) → (B,N,D)`。Wan 的 3D patchify（时间 patch=1，空间 patch=2）是实际的工业方案。需扩展 `patchify` 支持 5D 输入 `(B,C,T,H,W)`。

### 7.2 `attention.py`
- Wan 的 full attention over all spacetime tokens 展现了视频 attention 的真实压力：32K tokens 的 O(n²) attention 在中等显存配置上已接近极限。这验证了 linear attention 或 sparse attention 的必要性。
- 当前 `SelfAttention` 使用 `F.scaled_dot_product_attention`，本身就支持 FlashAttention，这是能跑 32K tokens 的原因。

### 7.3 `vae_stub.py`
- Wan 的 3D VAE 提供了 video VAE 接口的具体参考：`encode(video_pixels) → (B,C,T,H,W)`，`decode(latent_3d) → video_pixels`
- Channel=16 而非 4，与 image VAE (SD3/FLUX) 一致，这意味着视频 VAE 的 latent 容量比 SD1.x 的 4ch VAE 大 4 倍

### 7.4 `pipeline.py`
- 视频 denoising loop 结构：text_encode() → noise init → for t in steps: DiT forward → CFG → Euler step → VAE decode
- 与图像 pipeline 结构完全相同，差异仅在 shape 从 4D→5D。这支持 pipeline 的 shape-agnostic 设计。

### 7.5 `memory_manager.py`
- Wan 的显存分配模式：DiT 权重（2.6GB）+ T5（5GB）+ attention activations（动态，随 token 数变化）+ VAE decode 临时 buffer
- 需要支持 "max_token_budget" 参数，超标时降级（减少 frame 数或降低分辨率）

### 7.6 `scheduler.py`
- Flow Matching Euler scheduler 与已实现的 `RectifiedFlowScheduler` 兼容（t∈[0,1]，Euler step）
- Wan 默认 50 步，更少步数（如 28 步）可降级使用但质量会降低

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 官方 GitHub：`https://github.com/Wan-Video/Wan2.1`
- HF T2V-1.3B：`https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B`
- HF T2V-14B：`https://huggingface.co/Wan-AI/Wan2.1-T2V-14B`

**读**：
- GitHub README 中的推理指引（`python generate.py` 参数和 VRAM 建议）
- 3D VAE 的 encode/decode 接口（理解 shape 约定和 scaling factor）
- DiT 的 patchify 实现（确认时间 patch 是否为 1）

**输出**：
- 本文档：`learning/papers/05_wan_video.md`（8 字段完整 + 不同规格 token 数对比表 + 资源档位判断）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
