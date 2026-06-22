# 09 — Pyramid Flow Matching：多尺度高效视频生成

> **方法名称**：Pyramid Flow Matching for Efficient Video Generative Modeling
> **arXiv**：搜索 "Pyramid Flow Matching efficient video generation"
> **分类**：文生视频 — 多尺度 flow matching（效率方向）
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

Pyramid Flow Matching 提出了用 **"金字塔式 flow"（多分辨率/多尺度 flow matching）** 提高视频生成效率的方法。它的核心思想——用低分辨率 latent 计算粗粒度 flow，再用高分辨率 latent 细化细节——直接降低了大分辨率 latent 下的 attention 开销。在 中等显存配置 约束下，multi-scale 策略比单纯降分辨率更智能：低分辨率阶段快速生成结构，高分辨率阶段只做局部精修。

---

## 2. 模型类型

**文生视频**。Pyramid Flow Matching 不是特定的模型实现，而是一种**方法论/框架**，可应用于任何基于 flow matching 的视频 DiT 架构。

---

## 3. 核心架构

### 3.1 Pyramid Flow 的核心思想

传统视频 DiT 在所有 denoising 步数中使用固定的 latent 分辨率（如 `(C, 16, 64, 64)`），这意味着从头到尾每一帧的 attention 都在相同的（通常很大的）token 数上执行。结果是：低质量阶段（噪声多）用高分辨率是浪费，高质量阶段（接近清洁）不得不继续付出高分辨率 attention 的代价。

**Pyramid flow 的解决方案**：在过程中逐步提升 latent 的分辨率：

```
Step 1-10（噪声阶段）：latent=(C, 2, 8, 8)     # 极低分辨率 → token 极少
Step 11-20（轮廓阶段）：latent=(C, 4, 16, 16)    # 低分辨率
Step 21-30（纹理阶段）：latent=(C, 8, 32, 32)    # 中分辨率
Step 31-40（细节阶段）：latent=(C, 16, 64, 64)  # 目标分辨率 → 仅最后精修
```

**每一级的计算量对比**：

| Level | Latent (T,H,W) | Patch(1,2,2) → Tokens | Attention 矩阵大小 |
|-------|----------------|----------------------|-------------------|
| Lev 1 | (2, 8, 8) | 2×4×4 = 32 | 32² ≈ 1K |
| Lev 2 | (4, 16, 16) | 4×8×8 = 256 | 256² ≈ 65K |
| Lev 3 | (8, 32, 32) | 8×16×16 = 2,048 | 2K² ≈ 4M |
| Lev 4 | (16, 64, 64) | 16×32×32 = 16,384 | 16K² ≈ 268M |

与全部 40 步在 Level 4 上做（40 × 268M = 10.7B attention ops）相比，pyramid 方案（10×1K + 10×65K + 10×4M + 10×268M ≈ 2.72B）节省了约 **75%** 的 attention 计算量。

### 3.2 各 Level 间的过渡

从粗 level 到细 level 需要 **upsample** latent：
- 方式 1：简单 bicubic/trilinear 插值
- 方式 2：可学习的 upsample 层（如 Conv3DTranspose）
- 方式 3：在 upsampled latent 上再跑 1-2 步 denoising 以恢复 quality

大多数 pyramid 实现在 level 切换时使用方式 1（插值 + 几步 refinement）。

### 3.3 Denoiser 的共享策略

Pyramid 框架的 denoiser 可以有多种组织方式：
- **Shared denoiser**：所有 level 使用同一个 DiT（因为 latent 形状不同，需要先 resize proj 层或使用 adaptive 结构）
- **Separate denoisers**：每个 level 有一个专用的小 DiT（参数更多但各 level 的 DiT 可以更小）
- **LoRA-style adapter**：一个共享 DiT + 每 level 一个轻量 LoRA 适配器

对于 受限显存推理，**shared denoiser + 简单插值 upsample** 是最经济的方案（只需加载一个 DiT，显存占用与单 scale 模型相同）。

### 3.4 与 Single-Scale 视频 DiT 的对比

| 维度 | Single-Scale DiT（Wan/CogVideoX） | Pyramid Flow Matching |
|------|----------------------------------|----------------------|
| Latent 分辨率 | 固定 | 由粗到细（渐增） |
| 早期步数 | 高分辨率（浪费） | 低分辨率（高效） |
| 晚期步数 | 高分辨率（必要） | 高分辨率（必要） |
| Total attention ops | N_steps × O(n²_full) | 加权平均，约节省 60-80% |
| 实现复杂度 | 简单 | 需要 level 切换和 upsample |
| 质量 | 理论上最好（全分辨率） | 接近（最后几级仍然是全分辨率） |

---

## 4. 推理数据流

```
prompt → text encoder → text embeddings
   │
   ▼
Level 1（噪声 → 结构）：z ~ N(0,I), shape=(C, T/8, H/32, W/32)
   for t in [1.0, 0.9, ..., 0.7]（10 步）:
       v = DiT(patchify(z), t, text)    # 低分辨率 → attention 极轻
       z = z + dt * v
   │
   ▼  upsample 2× 时间 + 2× 空间
Level 2（结构 → 轮廓）：z shape=(C, T/4, H/16, W/16)
   for t in [0.69, ..., 0.4]（10 步）:
       v = DiT(patchify(z), t, text)
       z = z + dt * v
   │
   ▼  upsample
... 重复直到目标分辨率
   │
   ▼
VAE decoder: → video pixels
```

**注意**：CFG 可以像普通 DiT 一样在每步做（cond+uncond），也可以只在最后几个 level 做（因为早期 level 主要是结构，text guidance 在粗 level 的效果有限）。这进一步节省了早期 level 的 VRAM 和时间（从双 forward 变单 forward）。

---

## 5. 关键 Tensor Shape

### 5.1 Pyramid Level 对照（以 49f×720×480 为例）

| Level | Latent Shape (C=4) | Spacetime Tokens (p=(1,2,2)) | Full Attn 矩阵 |
|-------|-------------------|------------------------------|---------------|
| L1 (1/4 scale) | `(4, 4, 22, 15)` | 4×11×8 = 352 | ~248 KB |
| L2 (1/2 scale) | `(4, 7, 45, 30)` | 7×23×15 = 2,415 | ~11.7 MB |
| L3 (full scale) | `(4, 13, 90, 60)` | 13×45×30 = 17,550 | ~616 MB |

### 5.2 总计算量对比

| 方案 | 方案描述 | Total Attention Ops | 节省 |
|------|---------|-------------------|------|
| Full scale, 50 steps | 所有步在 L3 | 50 × 616MB ≈ 30.8 GB | 0% |
| Pyramid, 50 steps | 15+15+20 steps in L1/L2/L3 | 15×0.25MB + 15×11.7MB + 20×616MB ≈ 12.5 GB | **60%** |
| Pyramid, 30 steps | 10+10+10 steps | 10×0.25 + 10×11.7 + 10×616 ≈ 6.28 GB | **80%** |

---

## 6. 系统推理影响

### 6.1 显存瓶颈

- **早期 level**：attention 占用极低（< 1 MB），DiT 权重是唯一显存占用
- **晚期 level**：与 single-scale DiT 相同（attention 是瓶颈）
- **Level 切换**：upsample 需要临时 buffer（2× latent size），但时间很短

### 6.2 可用的 CUDA GPU 可行性

> 🟢 **适合**（方法论层面）。Pyramid 思想本身就是为了让大 latent 的 attention 负担分散到不同 scale。具体可行性取决于两个因素：① 是否有开源 pyramid 视频模型（当前生态中 pyramid flow 的成熟实现较少）；② shared denoiser 方案的参数量。

对于 受限显存场景：即使没有现成的 pyramid 视频模型，在自写 `diffusion_engine` 中实现 multi-scale denoising loop 也是非常有价值的优化方向（T16/T17 可考虑）。核心思路：先在小 latent 上粗去噪（attention 快），再在 target latent 上精去噪（attention 贵但步数少）。

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `scheduler.py`
- Multi-scale scheduler：当前 scheduler 假设 latent shape 固定，pyramid 要求在不同分辨率下有不同 timestep 序列。需要 `level_scheduler` 概念。

### 7.2 `pipeline.py`
- Multi-scale denoising loop：`pipeline` 不再是一个简单的 for 循环，而是嵌套的 level loops（每个 level 有自己的步数和 latent shape）。
- Upsample 步骤需集成到 pipeline 中。

### 7.3 `memory_manager.py`
- 动态 latent buffer resize：在 level 切换时重新分配 latent buffer。低分辨率阶段可能可以释放一些 buffer 给 text embeddings 或 model weights。

### 7.4 `dit.py`
- 需要支持不同分辨率的 latent 输入（patchify 逻辑不变，但输入 shape 可变）。

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- arXiv：搜索 "Pyramid Flow Matching efficient video generation"
- GitHub：搜索相关开源实现
- 社区讨论：pyramid flow 的工程落地经验

**读**：
- Pyramid flow 的数学原理（多尺度 flow matching 的 teacher-student 关系）
- Architecture section（各 level 的 denoiser 共享策略）
- Inference section（各 level 的步数分配和 upsample 方式）

**输出**：
- 本文档：`learning/papers/09_pyramid_flow_matching.md`（8 字段完整 + pyramid 推理路径 + 受限显存价值分析）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
