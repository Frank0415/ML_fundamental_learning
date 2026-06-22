# 02 — FLUX.1 / FLUX 系列架构笔记

> **模型全称**：FLUX.1（schnell / dev / pro 三变体）
> **开发者**：Black Forest Labs（原 Stability AI 核心团队）
> **官方 blog**：[Announcing Black Forest Labs](https://blackforestlabs.ai/announcing-black-forest-labs/)
> **开源仓库**：[github.com/black-forest-labs/flux](https://github.com/black-forest-labs/flux)
> **分类**：文生图（image）— open-weight flow matching transformer
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

FLUX 是目前最强的 **open-weight flow transformer**，并且提供 **schnell（4 步推理）** 变体。它在 SD3 的 rectified flow + MMDiT 范式上做了关键工程改进：parallel attention block 减少 kernel launch、RoPE 替代 2D sin/cos 位置编码、以及蒸馏训练实现 few-step 推理。理解 FLUX 等于在 SD3 基线上验证 rectified flow 范式的实际落地效果，并观察 engineering 层面如何把推理门槛降到 消费级 GPU 可达。

---

## 2. 模型类型

**文生图（text-to-image）**。FLUX.1 系列三个变体：

| 变体 | 步数 | 许可 | 定位 | 资源档位 |
|------|------|------|------|-----------|
| **schnell** | 4 步 | Apache 2.0 | 开源最快，distilled few-step | 🟢 适合 |
| **dev** | 50 步 | 非商用 | 开发/研究，质量最高 | 🟡 极限可跑 |
| **pro** | 闭源 | 商业 API | 商业生产级 | 🔴 不可本地跑 |

---

## 3. 核心架构

### 3.1 Denoiser：Single-Stream DiT（与 SD3 MMDiT 的关键差异）

FLUX 的 denoiser 是 **single-stream DiT**，不同于 SD3 的双流 MMDiT：

**SD3 MMDiT（双流）**：
```
Text tokens ──→ [QKV_text] ──┐
Image tokens ──→ [QKV_img] ──┤
                              ├──→ Joint Attention ──→ [Text_out, Image_out]
```
Text stream 和 Image stream 有独立的 QKV 投影和 adaLN 调制。

**FLUX DiT（单流）**：
```
[Text, Image] tokens concat ──→ [QKV] ──→ Attention ──→ MLP ──→ output
```
所有 token（text + image）拼接后进入统一的 QKV 投影和 attention。换句话说，text 和 image token 在 transformer block 中共享同一组参数，没有"双流"的概念。

**为什么这对推理重要**：单流意味着参数更少（总参数量相近但无重复的 QKV 投影），Kernel launch 更少（一次 QKV 投影 vs 两次），但 text-image token 的交互可能不如双流精确（这通过更大的模型宽度和更长的训练来弥补）。

### 3.2 Parallel Attention Block（工程优化）

FLUX 的 transformer block 使用 **parallel attention + MLP** 结构：

```
输入 → ┌─ Self-Attention (QKV) ─┐
       │                          ├→ 相加 → output
       └─ MLP (FFN) ─────────────┘
```

即 attention 和 FFN 在**同一个 block 内并行计算**后相加，而非 SD3/标准 transformer 的"先 attention 后 FFN"串行结构。这种设计的工程收益：
- **减少 kernel launch 次数**：attention 和 FFN 的 QKV 和 gate/up 投影可以融合为一个 GEMM
- **提高 GPU 利用率**：更少的同步点，SM 空闲时间减少
- **对推理的影响**：在相同参数规模下，FLUX 的每步 forward 比 SD3 快约 15-20%（按社区报告）

但代价是：parallel attention 使训练更不稳定（attention 和 FFN 同时修改 hidden state），需要更仔细的初始化和 normalization。

### 3.3 Latent 表示

- **VAE 编码器**：8× 空间下采样，**16 通道** latent
- **与 SD3 VAE 的关系**：FLUX 使用自训练的 VAE（非直接复用 SD VAE），但编码参数（8×/16ch）与 SDXL/SD3 VAE 相同
- **Latent shape**：对 1024×1024 输入 → `(B, 16, 128, 128)`
- **为什么保持 16ch**：16 通道 VAE 是 SDXL 以来的经验共识，在 latent 容量和编码效率间取得了平衡。4ch（SD1.x）容量不足，更多通道导致 VAE 训练不稳定

### 3.4 Text Conditioning

**双路 text encoder**，与 SD3 不同，FLUX 没有用 CLIP-G：

| Encoder | Hidden Dim | Max Length | 权重 | 备注 |
|---------|-----------|------------|------|------|
| CLIP-L (ViT-L/14) | 768d | 77 tokens | ~430M | 标准 CLIP |
| T5-XXL | 4096d | 256~512 tokens | ~11B | 长文本理解关键 |

**FLUX 的 text conditioning 特点**：
- **更依赖 T5**：FLUX 对 T5 的依赖比 SD3 更强。schnell 即使去 T5 也能生成，但长文本理解和细节会明显下降
- **Pooled embedding**：CLIP-L 的 pooled output 注入到 adaLN，控制整体风格
- **无 CLIP-G**：FLUX 团队认为 CLIP-G 的收益不大（参数量增加 4× 但质量提升有限），删除了这个 encoder

### 3.5 Timestep / Sigma Conditioning

- **Rectified flow 风格**：`t ∈ [0, 1]`，与 SD3 相同的参数化
- **Fourier features + MLP** → adaLN 调制参数
- **与 SD3 的差异**：FLUX 使用 **QK-normalization**（对 attention 的 Q 和 K 做 RMSNorm），提高训练稳定性和推理时的数值精度。这在 large-scale flow matching transformer 中是一个重要但常被忽略的细节。

### 3.6 Attention 结构

**关键差异：FLUX 使用 RoPE（Rotary Position Embedding）**

| 特性 | SD3 / MMDiT | FLUX |
|------|-----------|------|
| Position encoding | 2D sinusoidal（patchify 阶段注入） | **2D RoPE**（attention 内旋转编码） |
| QK normalization | 无 | **有**（QK-LayerNorm） |
| Attention 类型 | Full attention（joint） | Full attention（单流） |
| FlashAttention 兼容 | 是 | 是 |

**RoPE 在 diffusion 中的角色**：RoPE 原本是 LLM 的位置编码方案（Su et al., 2021）。FLUX 将其扩展到 2D，对 image token 的 (x, y) 坐标分别应用频率旋转。这使得 attention 能够感知 token 之间的相对位置关系，在相同 token 数下比 2D sin/cos 提供更好的位置感知。

但这**偏离了 rectified flow 的主流**：SD3、Sana、大多数视频 DiT 使用 2D sinusoidal 或可学习位置编码。FLUX 的 RoPE 选择是一个"工程实用主义"的决策——不是因为 RoPE 比 sin/cos 理论上更好，而是因为 FlashAttention 对 RoPE 有原生 kernel 加速，实测速度更快。

### 3.7 VAE

- **自训练 VAE**：8× 空间下采样，16 通道 latent
- **支持 tiling**：大分辨率（2048px+）时可分块 decode 以避免 decoder OOM
- **scaling_factor**：与 SD3 VAE 不同的归一化常数，加载模型时自动处理

---

## 4. 推理数据流

### 完整路径（schnell 为例）

```
prompt
  │
  ├─→ tokenizer → CLIP-L (768d, 77 tokens) + pooled (768d)
  └─→ tokenizer → T5-XXL (4096d, 256 tokens)            [schnell 可 omit]
  │
  ▼
text embeddings: concat + pooled → adaLN
  │
  ▼
noise latent z₁ ~ N(0, I)   shape: (1, 16, H/8, W/8)
  │
  ▼
denoising loop（schnell: 仅 4 步！）
  ├─ forward: DiT(z_t, text, t) → v_cond
  │  内部：patchify(p=2) → RoPE → parallel attn+MLP → unpatchify
  ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)   [schnell: s≈0~1.5]
  │   注：schnell 蒸馏后 guidance 已内化，cfg=0 也可生成
  └─ Euler step: z_{t - Δt} = z_t + Δt · v_cfg
  │
  ▼
z₀ → VAE decoder → pixel image
```

### FLUX schnell vs dev 的关键推理差异

| 维度 | schnell | dev |
|------|---------|-----|
| Denoising 步数 | **4 步** | 50 步 |
| CFG scale | 0~1.5（guidance 内化） | 3.5~7.0 |
| 加载 T5 | 可选（omit 可节省 ~5GB） | 强烈建议 |
| 单次推理 latency | ~2-5 秒（可用的 CUDA GPU） | ~30-60 秒 |
| Peak VRAM | ~6-8 GB | ~12-14 GB |

---

## 5. 关键 Tensor Shape

### 5.1 Latent Shape

| 分辨率 | Latent Shape | Image Tokens（p=2） | 说明 |
|--------|-------------|---------------------|------|
| 512×512 | `(1, 16, 64, 64)` | 32² = 1024 | |
| 1024×1024 | `(1, 16, 128, 128)` | 64² = 4096 | 主流分辨率 |
| 2048×2048 | `(1, 16, 256, 256)` | 128² = 16384 | attention O(n²) 压力大 |

### 5.2 Text Embedding Shape

| Encoder | Shape | 说明 |
|---------|-------|------|
| CLIP-L sequence | `(1, 77, 768)` | 固定填充 |
| CLIP-L pooled | `(1, 768)` | 全局语义 |
| T5-XXL sequence | `(1, 256, 4096)` | 可变长度 |
| 合并后 text tokens | `(1, 333, D_text)` | 77+256=333（拼接后投影到 D_text） |

### 5.3 RoPE 位置编码 Shape

FLUX 的 2D RoPE 对 image token 的 (x, y) 坐标分别编码。对 128×128 latent（4096 tokens），每个 token 的位置 `(row, col)` ∈ [0,127]×[0,127] 被映射到 D/2 维的旋转频率。

---

## 6. 系统推理影响

### 6.1 显存瓶颈（按严重程度排序）

| 排序 | 瓶颈 | VRAM（fp16, 1024px） | 说明 |
|------|------|---------------------|------|
| 🔴 1 | T5-XXL | ~11 GB | 最大单组件。schnell 去 T5 可释放 ~5GB |
| 🟡 2 | DiT 权重 (dev) | ~7-8 GB | dev 的 transformer 参数量 ~4-5B |
| 🟢 3 | DiT 权重 (schnell) | ~5-6 GB | schnell 和 dev 同架构，仅训练方式不同 |
| 🟡 4 | Attention activations | ~3 GB/step | 4096 tokens 的 full attention |
| 🟢 5 | CLIP-L | ~0.9 GB | 小 |

### 6.2 哪些可以 Cache

| 可 Cache | 收益 |
|---------|------|
| **CLIP-L embeddings** | 一次 encode，全程复用 |
| **T5 embeddings**（若加载） | 一次 encode，全程复用 |
| **Unconditional embedding**（schnell 用） | schnell 的 cfg≈0 时不需要 unconditional forward |

### 6.3 哪些不能 Cache

与 SD3 完全相同：denoiser 每步 K/V 不能 cache（latent 每步刷新）。

### 6.4 资源档位与运行边界

| 变体 | 判断 | VRAM | 推荐配置 |
|------|------|------|---------|
| **FLUX.1-schnell（no-T5）** | 🟢 适合 | ~6-7 GB | `--model black-forest-labs/FLUX.1-schnell --dtype fp16 --enable_model_cpu_offload` |
| FLUX.1-schnell（含 T5） | 🟡 极限可跑 | ~10-11 GB | 需 sequential offload，T5 占 ~5GB |
| **FLUX.1-dev** | 🔴 不适合（默认） | ~14 GB+ | 50 步 + 4-5B 参数 + T5 已超预算 |
| FLUX.1-dev（Q4 量化 + offload） | 🟡 极限可跑 | ~8-10 GB | GGUF/NF4 量化，但质量有损 |

**一个受限显存示例命令**：

```bash
# schnell + 去 T5：最舒适
python -c "
from diffusers import FluxPipeline
pipe = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-schnell',
    torch_dtype=torch.float16
)
pipe = pipe.to('cuda')
pipe.enable_model_cpu_offload()
# schnell 默认 guidance_scale=0.0（蒸馏内化）
image = pipe('一只柴犬在樱花树下', num_inference_steps=4, guidance_scale=0.0).images[0]
image.save('output.png')
"
# VRAM ≈ 6.5 GB（实测社区报告，1024×1024，fp16，schnell，no-T5）

# schnell + T5 + 1024px：边界可跑
pipe = FluxPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-schnell',
    torch_dtype=torch.float16
)
pipe.enable_sequential_cpu_offload()  # 比 model_cpu_offload 更激进
# VRAM ≈ 10 GB
```

**HF Model Card**：
- FLUX.1-schnell：`https://huggingface.co/black-forest-labs/FLUX.1-schnell`
- FLUX.1-dev：`https://huggingface.co/black-forest-labs/FLUX.1-dev`

**社区量化版本**（更友好的 中等显存配置 路径）：
- `https://huggingface.co/city96/FLUX.1-schnell-gguf`（GGUF Q4 量化，schnell ~3GB 权重）
- `https://huggingface.co/Kijai/flux-fp8`（FP8 量化，社区维护）

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `scheduler.py`
- 已实现的 `RectifiedFlowScheduler` 可直接用于 FLUX 风格推理（t∈[0,1]，Euler step）
- Few-step（4 步）scheduler 的接口需支持"非均匀 timestep 间距"，因为 distilled 模型的步数分布非常不均匀（初始几步和最后几步的间隔可能很大）
- T16 的 scheduler benchmark 应分别测试 4/8/16/28/50 步的 latency-quality tradeoff

### 7.2 `attention.py`
- FLUX 使用 **RoPE**，这是对 SD3 的 2D sin/cos 位置编码的偏离。当前的 `attention.py`（T11）使用可学习 position embedding，与 FLUX 和 SD3 都不同
- QK-normalization（注意力中对 Q 和 K 做 LayerNorm）是 FLUX 的独特设计，用于提高推理数值精度——TinyDiT 的 toy 实现不需要，但 T18 总结时应标注"真实 DiT 通常会加入 QK-norm"
- Parallel attention + MLP 是工程优化，TinyDiT 保持串行（attn → FFN）以简化实现

### 7.3 `dit.py`
- FLUX 的单流 DiT（text+image concat 后统一处理）比 SD3 的双流 MMDiT 更接近我们 T11 的 toy 简化——我们的 TinyDiT 也是拼接式处理
- 这意味着 TinyDiT 实际上更像 FLUX 的单流架构而非 SD3 的 MMDiT 双流。T18 总结时应明确这个对应关系
- Patch_size=2（与 SD3 相同）仍是推荐值

### 7.4 `text_conditioning.py`（T12 待实现）
- FLUX 的双 encoder（CLIP-L + T5）提供了比 SD3 三 encoder 更简单的参考——我们的 v1 可以先实现 CLIP-L + pooled 注入 adaLN
- T5 可选的概念需在接口设计中体现：`TextConditioning` 应支持"加载 T5 / 不加载 T5"两种路径

### 7.5 `pipeline.py`（T12 待实现）
- FLUX schnell 的 few-step loop（4 步出图）是我们在 受限显存场景下的**目标 benchmark**
- Schnell 的 CFG=0（guidance 蒸馏内化）意味着只需一次 forward per step（不需要 cond+uncond 双 forward），这对 受限显存场景极其友好
- Dev 的 50 步 + CFG>3 → 双 forward → VRAM 和时间压力——这反过来证明蒸馏对推理的重要性

### 7.6 `memory_manager.py`（T12 待实现）
- Parallel attention block 的 activation memory 模式不同于串行 attention+FFN（并行模式下 attention 和 FFN 的中间结果同时存在），需在 T16 的 `attention_memory_benchmark.py` 中考虑

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 官方 repo：`https://github.com/black-forest-labs/flux`
- 官方 blog（架构说明）：`https://blackforestlabs.ai/announcing-black-forest-labs/`
- HF model card（schnell）：`https://huggingface.co/black-forest-labs/FLUX.1-schnell`
- HF model card（dev）：`https://huggingface.co/black-forest-labs/FLUX.1-dev`
- Diffusers pipeline 源码：`diffusers/pipelines/flux/pipeline_flux.py`
- 社区逆向工程笔记：GitHub 搜索 "FLUX architecture deep dive" 或 "FLUX internals"

**读**：
- 官方 blog 的 architecture 部分：了解 parallel attention、QK-norm、RoPE 的设计决策
- Diffusers pipeline 源码的 `__call__` 方法：重点看 schnell vs dev 的分支差异（CFG 处理、T5 加载、scheduler step）
- 社区 benchmark：FLUX schnell vs SD3 Medium vs Sana 在 中低显存 GPU 上的 latency/VRAM 对比
- 社区量化报告：GGUF Q4 / NF4 / FP8 量化后的 VRAM 降低和质量损失

**输出**：
- 本文档：`learning/papers/02_flux_architecture_notes.md`（8 字段完整 + schnell/dev 对比 + 资源档位判断）
- 不要求产出 HTML（HTML 留给 T8）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T7 (Wave 2)*
