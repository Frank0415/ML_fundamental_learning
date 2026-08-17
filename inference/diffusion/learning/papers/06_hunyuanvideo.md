# 06 - HunyuanVideo 1.5：系统化视频生成框架

> **模型名称**：HunyuanVideo 1.5（腾讯混元视频）
> **官方仓库**：[github.com/Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo)
> **HF Model Card**：[huggingface.co/tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
> **分类**：文生视频 - 大规模视频 DiT（bonus 路线）
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

HunyuanVideo 是目前最系统的开源视频生成框架。它不只是发布一个模型，而是覆盖了 data curation、3D VAE、DiT 设计、training recipe 的**全链路方法论**。尽管它的主力模型（8.3B 参数）在中等显存配置上偏紧，但它的系统化文档和经验对于理解"工业级视频推理需要什么"有不可替代的参考价值。另外，HunyuanVideo 1.5 引入了 step distillation，使得推理步数大幅减少，这是 受限显存场景下"边界可跑"的关键利好。

---

## 2. 模型类型

**文生视频（text-to-video）**。HunyuanVideo 1.5 的核心规格：

| 属性 | 值 |
|------|-----|
| 参数量 | **8.3B**（DiT backbone） |
| VAE | 3D VAE（C=16, 时间 4×, 空间 8×） |
| Text encoder | 双编码器：双语 CLIP + 多语言 T5（推测） |
| 推理步数（原始） | 50 步 |
| 推理步数（step-distilled） | 10~20 步（1.5 版本新增） |
| 官方最低 VRAM | **14GB**（标准推理） |
| 社区优化 VRAM | **6GB**（GGUF 量化 + CPU offload） |

---

## 3. 核心架构

### 3.1 Denoiser：Dual-Stream Video DiT

HunyuanVideo 的 denoiser 是 **双流 Video DiT**，结合了 SD3 MMDiT 的双流设计和视频 DiT 的 spacetime 处理：

- **Image stream**：视频 latent tokens（spacetime patches → tokens）
- **Text stream**：text encoder 输出的 text tokens（CLIP + T5）
- **Joint attention**：两流各自有独立的 QKV 投影和 AdaLN 调制，在 attention 中统一计算（与 SD3 MMDiT 的设计同源）

**双流 vs 单流**：与 Wan（单流，text+image concat 后统一处理）不同，HunyuanVideo 的双流使 text tokens 和 image tokens 的关系处理更精细，但也增加了参数（两组 QKV 和 AdaLN）。

### 3.2 Latent 表示：3D VAE

- **空间压缩**：8×（同 image VAE）
- **时间压缩**：4×
- **通道数**：**16**（与 SD3/FLUX VAE 通道数一致，提供高 latent 容量）
- **VAE 类型**：3D CNN-based VAE（非 causal）

**关键特性**：HunyuanVideo 的 3D VAE 在质量和压缩效率之间做了精心平衡。16 通道比 4 通道的视频 VAE（如 CogVideoX）存储更多信息，但也意味着 denoiser 的输入 token 维度 p²C 更大（patch_size=2 时，每个 token 的输入维度 = 2×2×16 = 64）。

### 3.3 Text Conditioning

HunyuanVideo 使用多语言双向 text encoder 系统：
- **CLIP**：提供视觉-文本对齐
- **T5**：提供长文本理解
- 双 encoder 输出拼接后通过投影层送入 DiT

**对显存的影响**：双 encoder 在加载时占用更多显存（CLIP ~2GB + T5 ~5GB = ~7GB），但都只 encode 一次，后续 denoising 循环中复用。

### 3.4 Step Distillation（1.5 版本关键特性）

HunyuanVideo 1.5 引入了 **step distillation**，这是对 受限显存场景最重要的利好：
- 原始模型：50 步推理
- 蒸馏后：**10~20 步**推理即可达到类似质量
- 蒸馏方式：推测使用 progressive distillation（teacher 50 步 → student 20 步 → 10 步）
- 对显存的影响：每步 peak VRAM 不变，但总步数从 50→10 意味着 wall time 缩短 5×，且由于步数少，offload 的上下文切换次数也少

### 3.5 Attention 结构

| 特性 | HunyuanVideo |
|------|-------------|
| 自注意力类型 | Full attention（双流 joint） |
| Text-image 交互 | Joint attention（双流各自 QKV） |
| 位置编码 | 3D sinusoidal（T, H, W 分别编码） |
| Causal mask | 无（所有 spacetime tokens 互 attend） |
| FlashAttention 兼容 | 是 |

**与 CogVideoX 的关键差异**：HunyuanVideo 的 attention **不 causal**（所有帧互相 attend），而 CogVideoX 使用 3D causal attention（只能向前看）。非 causal 的优点是全局一致性更好，缺点是 token 数相同的条件下 attention 计算量一样。

---

## 4. 推理数据流

```
prompt ("一只猫在草地上奔跑")
   │
   ├─→ CLIP tokenizer → CLIP (768d) → text tokens
   └─→ T5 tokenizer → T5 (4096d) → text tokens
   │
   ▼
text tokens concat + projection → (B, L_text, D_dit)
   │
   ▼
noise latent z_T ~ N(0, I)  shape: (1, 16, T_latent, H_latent, W_latent)
   典型：129f×720×1280px → latent (1, 16, 33, 90, 160)
   │
   ▼
denoising loop（step-distilled: 10~20 步；原始: 50 步）
   ├─ patchify: (1,16,33,90,160) → (1, N_st, D)   N_st = 33 × 45 × 80 = 118,800
   ├─ Dual-stream DiT: img_adain → qkv_img, text_adain → qkv_text
   ├─ Joint Attention: concat [Q_img, Q_text] · concat [K_all, V_all]
   ├─ CFG: v_cfg = v_uncond + s · (v_cond − v_uncond)
   └─ Euler step
   │
   ▼
3D VAE decoder: (1, 16, 33, 90, 160) → (1, 3, 129, 720, 1280)
```

**Token 数爆炸**：标准 129f×720p 规格下，token 数 = 33 × 45 × 80 = **118,800**。这意味着单层 full attention 的矩阵是 118800² ≈ **14B 元素**。在 fp16 下约 28 GB per attention layer，对于任何消费级 GPU 都完全不可能。

这也是为什么 HunyuanVideo 默认规格不建议在消费级 GPU 上跑，**必须大幅降规格**（减少帧数/分辨率）或 **使用社区量化/CPU offload 方案**。

---

## 5. 关键 Tensor Shape

### 5.1 不同规格下 Token 数

| 规格 | 帧数 | 分辨率 | Latent Shape | Spacetime Tokens | Full Attn 矩阵（fp16） |
|------|------|--------|-------------|-----------------|----------------------|
| **中等显存极限** | 9 | 256×256 | `(16, 3, 32, 32)` | `3×16×16 = 768` | ~1.2 MB |
| **中等显存配置 现实** | 17 | 384×384 | `(16, 5, 48, 48)` | `5×24×24 = 2,880` | ~16.6 MB |
| **边界** | 33 | 480×640 | `(16, 9, 60, 80)` | `9×30×40 = 10,800` | ~233 MB |
| **官方默认** | 129 | 720×1280 | `(16, 33, 90, 160)` | `33×45×80 = 118,800` | ~28 GB ❌ |

### 5.2 降级策略

```
官方规格：129f × 720p → 118,800 tokens → 28GB attention（完全不可行）
          ↓
降帧数：   33f × 720p → 30,240 tokens  → 1.8GB attention
          ↓
降分辨率： 33f × 480p → 13,500 tokens  → 364MB attention
          ↓
受限显存目标：17f × 384p → 2,880 tokens   → 16MB attention ✅
```

**关键**：每次降低分辨率或帧数，token 数都按乘积关系下降。从 118,800 → 2,880（41× 减少），attention 矩阵从 28GB → 16MB（1750× 减少）。这就是为什么"降规格"在视频推理中效果如此显著。

### 5.3 Text Embedding Shape

| 名称 | Shape | 说明 |
|------|-------|------|
| CLIP tokens | `(1, 77, 768)` | 固定长度 |
| T5 tokens | `(1, L_t5, 4096)` | 可变长度 |
| 合并后 | `(1, L_total, D_dit)` | 拼接后投影到 DiT hidden dim |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

| 排序 | 组件 | VRAM | 说明 |
|------|------|------|------|
| 🔴 1 | DiT attention activations | 随 token 数 O(n²) 增长 | 最大变量 |
| 🔴 2 | DiT 权重（8.3B） | ~16.6 GB fp16 | 单权重就超过 中等显存配置 |
| 🟡 3 | T5 text encoder | ~5 GB | |
| 🟡 4 | CLIP text encoder | ~2 GB | |
| 🟡 5 | 3D VAE decoder | ~3 GB | |

### 6.2 中等显存配置 可行路径

由于 8.3B DiT 权重本身就是 16.6GB fp16，**纯 fp16 加载必然 OOM**。要在中等显存配置上运行，必须采用以下组合：

| 方案 | 效果 | 代价 |
|------|------|------|
| **GGUF/NF4 量化**（社区版） | 权重从 16.6GB → ~4-5GB | 质量略降（<5%），但推理可用 |
| **CPU offload** | 逐模块加载/卸载 | 推理极慢（15~30 min） |
| **降规格到 17f×256p** | token 数从 118K → 768 | 视频极短、极低分辨率 |
| **Step distillation（10 步）** | wall time 缩短 | 每步 peak VRAM 不变 |

**社区实测（中等显存配置）**：GGUF Q4 量化版 + 17f×384p + 10 步 + CPU offload → VRAM ≈ 6-7 GB，推理时间 ~10-15 分钟。

### 6.3 资源档位与运行边界

| 配置 | 判断 | 说明 |
|------|------|------|
| **纯 fp16 + 官方默认规格** | 🔴 不适合 | 权重 + attention 共 25+ GB |
| **GGUF Q4 + 降规格 + CPU offload** | 🟡 极限可跑 | ~6-7 GB，但 slow |
| **GGUF Q4 + 17f×384p + 10 steps** | 🟡 极限可跑 | 小而短的视频可尝试 |

**结论：HunyuanVideo 在中等显存配置上偏紧，不属于本项目的主力路线。仅作为 bonus 了解（理解工业级视频推理的全链路），实际尝试以 Wan2.1-1.3B / CogVideoX-2B / LTX-Video 为主。**

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `dit.py`
- 双流 Video DiT 是 MMDiT 在视频模态的延伸。当前 TinyDiT 是单流（image-only），双流需要分别处理 image stream 和 text stream 的 AdaLN 调制，这对 T12 的 text conditioning 设计有直接影响。

### 7.2 `attention.py`
- HunyuanVideo 的 token 爆炸问题（118K tokens → 28GB attention）是一个重要的反面教材：**在中等显存配置上做 video attention，token 数必须控制在 ~3K 以下**，否则 O(n²) 无法承受。
- 这反过来证明了 attention.py 需要 `max_tokens` 参数和 linear attention 替代方案。

### 7.3 `memory_manager.py`
- HunyuanVideo 证明了"模型权重量化"（GGUF/NF4）是 受限显存视频推理的常见路径。memory_manager 应预留 "weight_quantization" 和 "offload_strategy" 的配置项。
- 注意力显存的公式：`n² × 2bytes × num_attention_layers × 2(QK + AV)`，这个公式对 video DiT 的显存预估算很重要。

### 7.4 `pipeline.py`
- Step distillation（50→10 步）对 pipeline 的启发：pipeline 应支持 `num_steps` 的灵活配置，并且 scheduler 在低步数时自动调整 timestep 分布（非均匀间距）。

### 7.5 `scheduler.py`
- Distilled scheduler 的 timestep 分布不是线性或对数均匀的，它是由 teacher-student 蒸馏学习到的。scheduler 接口需要支持自定义 `timestep_list`。

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- 官方 GitHub：`https://github.com/Tencent/HunyuanVideo`
- HF Model Card：`https://huggingface.co/tencent/HunyuanVideo-1.5`
- arXiv：搜索 "HunyuanVideo 1.5 technical report"
- 社区 GGUF 量化版：HF 搜索 "HunyuanVideo-GGUF"

**读**：
- 官方 README 中的系统要求和推理命令
- Architecture 文档（了解双流 DiT 和 3D VAE 的设计决策）
- 社区的 中等显存配置 尝试报告（GGUF 量化效果、offload 策略）

**输出**：
- 本文档：`learning/papers/06_hunyuanvideo.md`（8 字段完整 + 降级策略表 + 中等显存配置 复杂度分析）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
