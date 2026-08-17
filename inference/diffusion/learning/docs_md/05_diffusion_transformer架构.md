# 05 - Diffusion Transformer (DiT) 架构

DiT（*Scalable Diffusion Models with Transformers*，Peebles & Xie, ICCV 2023）用 Transformer 取代 U-Net 作为扩散模型的去噪骨干。本页按数据形状拆解它的结构。

## 1. 为什么用 Transformer 替代 U-Net？

传统扩散模型（如 Stable Diffusion 1.x/2.x）使用 **U-Net** 作为去噪器。U-Net 的瓶颈：

- **难以 scaling**：U-Net 参数增大会导致显存爆炸（跨分辨率特征图）
- **文本条件注入受限**：cross-attention 仅在少数层注入，文本对齐能力有限
- **无法统一模态**：图像和视频需要不同的架构适配，而非统一 backbone

DiT 提出的答案：**把 latent 切成 patches，用标准 Transformer（ViT 风格）处理**。这带来了：

- 可预测的 scaling 行为（参数翻倍 → 性能变好，不爆炸）
- 自然的文本条件注入（所有层都能 attend 文本 embedding）
- 与 NLP 研究生态对齐（可使用成熟的 Transformer 优化技术）

## 2. DiT Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────────────────┐    ┌──────────┐    ┌──────────┐
│ 噪声画像  │ -> │ Patchify │ -> │ DiT Transformer      │ -> │ Unpatch  │ -> │ 预测噪声  │
│ (B,C,H,W)│    │ (B,N,D)  │    │ Blocks × depth       │    │ (B,C,H,W)│    │ ε_θ 或 v_θ│
└──────────┘    └──────────┘    │ + Timestep + Text    │    └──────────┘    └──────────┘
                                └──────────────────────┘
```

<figure style="margin: 1rem 0 1.5rem;">
  <img src="../../docs/assets/architecture/sd3_mmdit_architecture.png" alt="以 SD3 为代表的 DiT / MMDiT 组件总览图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：SD3 论文 Figure 2。虽然它对应的是 MMDiT 而不是最原始的 DiT，但左侧组件图完整覆盖了 patchify、position embedding、timestep conditioning 和 transformer 主干这条标准推理路径。
  </figcaption>
</figure>

### Step 1: Patchify

将 `(B, C, H, W)` 切割为 \(N = (H / p) \times (W / p)\) 个 patch，每个 patch 展平为 \(p \times p \times C\) 维。

| 参数 | 示例 | 结果 |
|------|------|------|
| latent=(1,4,32,32), p=2 | \(16 \times 16 = 256\) patches | (1, 256, 16) |
| latent=(1,4,64,64), p=2 | \(32 \times 32 = 1024\) patches | (1, 1024, 16) |
| latent=(1,16,32,32), p=2 | \(16 \times 16 = 256\) patches | (1, 256, 64) |

### Step 2: Position Embedding

DiT 使用 **2D sinusoidal position embedding**：每个 patch 的 (row, col) 映射为正弦/余弦编码并相加。这与 NLP 的 1D position encoding 不同，因为图像有二维空间结构。

*toy 简化*：我们的 TinyDiT 使用可学习的 `nn.Parameter` 替代。

### Step 3: Timestep Conditioning (AdaLN)

timestep t → sinusoidal encoding → SiLU MLP → 调制参数 (shift, scale, gate)。

这些参数被注入到 **每个 DiT block 的两次 LayerNorm 之前**：

```python
modulated = LayerNorm(x) * (1 + scale) + shift
```

gate 用于控制残差连接的强度：

```python
x = x + gate * attention_output
x = x + gate * ffn_output
```

### Step 4: DiTBlock 内部

每个 Block 包含：

- **Attention 子层**：AdaLN → MultiHeadSelfAttention → gate 残差
- **FFN 子层**：AdaLN → Linear → GELU → Linear → gate 残差

Attention 是 **non-causal full self-attention**：所有 patches 互相 attend，无因果掩码。

### Step 5: Unpatchify + Output

最终 Linear 层将 `(B, N, hidden)` 转换为 \((B, N, p^2C)\)，再通过 reshape/transpose 重组为 `(B, C, H, W)`。

输出可以是 **\(\epsilon_\theta\)**（预测噪声）或 **\(v_\theta\)**（预测速度/矢量场），取决于训练目标。

## 3. Shape 对照

| 阶段 | 输入 Shape | 输出 Shape |
|------|-----------|-----------|
| Patchify | (B, C, H, W) | (B, N, p²C) |
| Input Proj | (B, N, p²C) | (B, N, D) |
| + Pos Embed | (B, N, D) | (B, N, D) |
| Timestep Emb | (B,) | (B, D) |
| Text Proj | (B, L, D_text) | (B, L, D) |
| DiTBlock × depth | (B, N, D) + (B, D) + (B, L, D)? | (B, N, D) |
| Output Head | (B, N, D) | (B, N, p²C) |
| Unpatchify | (B, N, p²C) | (B, C, H, W) |

## 4. 与 U-Net 的差异

| 维度 | U-Net (SD1.x/2.x) | DiT (SD3/FLUX) |
|------|------------------|----------------|
| 骨干 | 卷积 + 下采样/上采样 | Transformer blocks |
| 分辨率处理 | 多尺度特征图 | 固定分辨率 token 序列 |
| 文本注入 | Cross-attention（仅特定分辨率层） | 所有层均可访问 text tokens |
| 位置编码 | 卷积隐式编码 | 显式 2D sinusoidal |
| Scaling | 参数×10 → 显存×20+ | 参数×10 → 可预测提升 |
| 多模态 | 图像专用 | 可扩展至视频（spacetime patch） |

## 5. AdaLN-Zero 机制（真实 DiT）

真实 DiT 论文中的 AdaLN-Zero 使用零初始化：

- 最后输出层的权重初始化为零
- 训练开始时，\(\mathrm{gate}_{\mathrm{attn}} = \mathrm{gate}_{\mathrm{ffn}} = 0\)，所以 \(x = x + 0 = x\)
- 模型退化为恒等映射，逐步学习非零 gate
- **toy 简化**：我们的 TinyDiT 使用标准 Xavier 初始化，未实现零初始化

## 6. 与 LLM Attention 的差异

| 维度 | LLM (如 GPT) | DiT |
|------|-------------|-----|
| 注意范围 | 因果（causal，只看向前） | 全连接（所有 patches 互相看） |
| KV Cache | 必须（自回归生成需要） | 不需要（每步 latent 全刷新） |
| 位置编码 | 1D RoPE | 2D sinusoidal / learnable |
| 归一化 | RMSNorm（静态） | AdaLN（动态，依赖 timestep） |
| 文本交互 | 无（纯文本模型） | 有（text tokens 参与 attention） |

## 7. DiT 在视频生成中的扩展：Spacetime Patch

图像 DiT 的 patchify 只在 (H, W) 二维上切分。视频 DiT 将这一思想扩展到三维，**spacetime patch** 在 (T, H, W) 三个维度上同时切分视频 latent。

### Spacetime Patch 的计算

给定视频 latent `(B, C, T_latent, H_latent, W_latent)` 和 patch 参数 `(t_p, h_p, w_p)`：

\[
N_{\mathrm{spacetime}} = \left(T_{\mathrm{latent}} / t_p\right) \times \left(H_{\mathrm{latent}} / h_p\right) \times \left(W_{\mathrm{latent}} / w_p\right)
\]

每个 patch 展平为 \(t_p \times h_p \times w_p \times C\) 维。

| 视频规格 | VAE 压缩后 Latent | Patch (1,2,2) | Spacetime Tokens | Full Attn 矩阵 |
|---------|------------------|--------------|------------------|----------------|
| Sora 推测：16f×256² | `(C, 4, 32, 32)` | \(4 \times 16 \times 16\) | \(= 1{,}024\) | ~2 MB |
| CogVideoX：49f×720×480 | `(4, 13, 90, 60)` | \(13 \times 45 \times 30\) | \(= 17{,}550\) | ~616 MB |
| Wan2.1-1.3B：81f×480×832 | `(16, 21, 60, 104)` | \(21 \times 30 \times 52\) | \(= 32{,}760\) | ~2.1 GB |
| LTX-Video (32×VAE)：121f×720×480 | `(4, 15, 22, 15)` | \(15 \times 11 \times 8\) | \(= 1{,}320\) | ~3.5 MB |

**Token 数对 attention 的影响**：LTX-Video 的 VAE 使用 32× 空间压缩而不是 8×，将 token 数从 40,500+ 降到 1,320，减少了 30×。按 \(O(N^2)\) 计算，attention 成本降低近 1000×。在中等显存配置下，减少 token 数会同时缩小 attention 矩阵的两个维度。

### 3D Position Encoding

视频 DiT 的位置编码从 2D 扩展到 3D：每个 spacetime patch 的 `(t, y, x)` 坐标分别编码后相加或拼接。Wan2.1 和 CogVideoX 在 patchify 阶段注入 3D positional encoding，确保模型知道每个 token 在视频时空中的时空位置。

## 结论

DiT 将 image latent 切割为 patch tokens，用带 timestep 调制 AdaLN 的 Transformer blocks 替代 U-Net。论文证明了 Transformer 可以承担扩散模型的去噪主干，并随模型规模增长改善生成结果。它与 LLM 的主要区别是 non-causal full attention、无 KV cache，以及由 timestep 动态控制的归一化。

## 和我的 diffusion_engine 的关系

`diffusion_engine/core/dit.py` 中的 `TinyDiT` 是 DiT 的 toy 实现。它包含 patchify/unpatchify、sinusoidal timestep embedding（torch 版）、AdaLN 调制的 DiTBlock、以及可选的 text conditioning。这是一个简化的验证性实现，用于测试 shape 系统和 pipeline 集成，不用于真实图像生成。真实 DiT（如 SD3/FLUX 所用的版本）参数规模可达 2B~12B，特征维度 1536~3072。
