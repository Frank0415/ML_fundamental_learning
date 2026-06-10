# 10 — Consistency / Distillation / Few-Step Sampling 总览

> **方法族**：Consistency Model, LCM, Turbo, Lightning, Progressive Distillation
> **覆盖范围**：所有扩散模型的 few-step 推理加速方法
> **分类**：fast sampling（适用文生图 + 文生视频）
> **阅读日期**：2026-06-07

---

## 1. 为什么对现代 diffusion 推理重要

在 12GB VRAM 预算下，**每减少一步 denoiser forward，就是 1-3 秒延迟和数百 MB 显存峰值时间的节省**。全步长扩散推理（28-50 步）在 12GB 上只勉强可用，但 few-step 推理（1-8 步）将 diffusion 从"耐心等待"变成"即时生成"。本章覆盖了从 Consistency Model 到 FLUX schnell 的各类 few-step 方法，它们是理解"蒸馏如何改变扩散推理的可行性"的关键。

---

## 2. 方法类型

这不是一个具体的模型，而是一类**方法/技术**的总览。核心问题：**如何用极少的去噪步骤（1-8 步）生成高质量结果？**

## 3. 各类方法总览

### 3.1 Consistency Model（一致性模型，Song et al., 2023）

**核心思想**：将扩散模型的完整 ODE 轨迹映射为一步映射——任何时间点的噪声 latent 直接映射到数据 latent。

```
传统扩散：x_T → (N 步 denoising) → x_0
一致性模型：x_T → (1 步映射) → x_0
```

**数学**：训练一个 consistency function `f(x_t, t)` 满足：
- 边界条件：`f(x_0, 0) = x_0`（在数据点上，函数是恒等映射）
- 自一致性：`f(x_t, t) = f(x_{t'}, t')`（沿 ODE 轨迹的所有点映射到同一个 x_0）

**推理**：仅需 1 步（从 x_T 直接得到 f(x_T, T) ≈ x_0），或 2 步（x_T → x_{T/2} → f(x_{T/2}, T/2)）以提高质量。

**代表模型**：Consistency Models（原始），Latent Consistency Models（LCM）。

### 3.2 Latent Consistency Model（LCM, Luo et al., 2023）

LCM 将一致性模型的思想应用到 **latent diffusion** 框架中（即 Stable Diffusion 的 latent 空间）。关键创新：
- 在 latent 空间而非像素空间做 consistency mapping
- 使用 **LoRA** 而非从头训练——意味着可以在不改变原 SD 权重的情况下添加 fast sampling 能力
- 推理 1-4 步即可得到合理结果

**代表性 HCF 模型**：
- `latent-consistency/lcm-lora-sdv1-5`：SD1.5 + LCM LoRA，1-4 步推理
- `latent-consistency/lcm-lora-sdxl`：SDXL + LCM LoRA

### 3.3 Turbo 系列（SD Turbo, SD3 Turbo, SDXL Turbo）

Turbo 变体通过 **对抗训练 + 蒸馏** 的混合策略将原模型压缩为 1-4 步推理：

| Turbo 变体 | 原模型 | 蒸馏步数 | 12GB 可行性 |
|-----------|--------|---------|-----------|
| SD Turbo | SD 2.1 | 1 步 | 🟢 非常舒适 |
| SDXL Turbo | SDXL | 1-4 步 | 🟢 舒适 |
| SD3.5 Large Turbo | SD3.5 Large | 4 步 | 🟡 极限可跑（8B 模型 + T5） |

**Turbo 的蒸馏策略**：
1. 用原模型（teacher）生成大量 paired data：(noise, denoised_image at 50 steps)
2. 训练 student 模型（Turbo）从 noise 一步映射到 denoised image
3. 加入对抗 loss（GAN loss）来补偿蒸馏造成的细节损失
4. 结果：Turbo 模型比纯蒸馏（无对抗 loss）的模型细节更好

### 3.4 FLUX schnell（快速变体）

FLUX schnell 是 **guided distillation** 的结果——不是简单的 step reduction，而是将 CFG guidance 在蒸馏过程中内化：

- **原始 FLUX dev**：50 步，cfg=3.5-7.0，cond+uncond 双 forward
- **FLUX schnell**：4 步，cfg≈0（guidance 已内化到模型权重中），单 forward

这意味着 schnell 的每步计算量比 dev 的一半还少（因为 dev 每步需要 cond+uncond 两次 forward）。总计算量：`4 步 × 1 forward = 4` vs `50 步 × 2 forward = 100` → **25× 加速**。

**schnell 的启示**：蒸馏不仅可以减少步数，还可以**消除 CFG 的双 forward 开销**。对于 12GB 场景，这至关重要（因为 CFG 的双 forward 是每步的显存峰值）。

### 3.5 Progressive Distillation（渐进蒸馏）

Progressive Distillation 不直接将 50 步压缩到 1 步，而是逐步压缩：

```
Teacher (50 steps) → Student (25 steps)
Student (25 steps) → Student (12 steps)
Student (12 steps) → Student (6 steps)
Student (6 steps) → Student (4 steps)
...
```

每一步蒸馏只压缩 2×，确保质量损失可控。这与一步蒸馏（50→4）不同，后者可能因为"压缩比太大"而丢细节。

**SD3.5 Large Turbo** 就是 progressive distillation 的产物：SD3 Large (50 steps) → 25 → 12 → 6 → 4 steps。

### 3.6 Lightning / AnimateLCM / VideoLCM

这些是视频模型的 fast sampling 变体：
- **AnimateLCM**：LCM 方法在 AnimateDiff（视频模型）上的应用，4 步视频生成
- **VideoLCM**：LCM 在通用视频 DiT 上的应用
- **Lightning**：Stable Video Diffusion 的蒸馏版，1-4 步

---

## 4. 各类方法的数学统一

尽管实现细节不同，这些方法的基本思路相似：

```
给定 teacher 模型 f_teacher（需要 N 步），训练 student 模型 f_student（只需 M 步，M ≪ N）
最小化：‖f_student^M(x_T) − f_teacher^N(x_T)‖²

其中 f^K(x) 表示从噪声 x 开始，K 步 denoising 后的输出
```

不同方法的核心差异在于：
- **蒸馏目标**：是匹配 teacher 的 1-step 输出（Consistency Model），还是匹配 teacher 的 full trajectory
- **是否用对抗 loss**：Turbo 用，LCM 一般不用
- **训练方式**：LCM 用 LoRA（轻量），Turbo 用 full fine-tune（重但效果好）

---

## 5. 为什么可以少步采样

### 5.1 ODE 轨迹的"直线性"

在全步长扩散中，ODE（常微分方程）轨迹是弯曲的——每个 timestep 需要跨过一段曲线。蒸馏的过程本质上是将弯曲的 ODE 轨迹"拉直"：

```
原始 ODE 轨迹（弯曲）：
x_T ──→ x_{T-1} ──→ x_{T-2} ──→ ... ──→ x_2 ──→ x_1 ──→ x_0

蒸馏后（拉直）：
x_T ──────────────────────────────→ x_0   (1 步)
或
x_T ──────→ x_mid ──────→ x_0        (2 步)
```

**为什么轨迹可以被拉直**：rectified flow 的理论保证了 ODE 轨迹可以通过 reflow 过程（用现有流生成的配对数据重新训练）逐步变直。蒸馏在这个意义上与 reflow 等价。

### 5.2 一致性函数的不变性质

Consistency Model 的关键数学性质是 **self-consistency**（自一致性）：沿 ODE 轨迹的所有点映射到同一个 x_0。这意味着即使一步从 x_T 跳到 x_0 是"大跳跃"，一致性函数的训练目标确保了跳跃的有效性——因为所有中间点也在训练中被"拉到"同一个 x_0。

---

## 6. 系统推理影响

### 6.1 Few-Step 推理的显存和延迟影响

| 步数 | Denoiser Forward 次数 | Peak VRAM | Wall Time（估, 2B DiT） | 12GB 适用性 |
|------|---------------------|-----------|----------------------|-----------|
| 50 步（全步长） | 50 × 2 = 100（cond+uncond） | ~10 GB | 3-5 min | 🟡 边界 |
| 28 步（SD3 默认） | 28 × 2 = 56 | ~10 GB | 1-2 min | 🟡 边界 |
| **8 步** | 8 × 2 = 16 | ~10 GB | 15-30 sec | 🟢 舒适 |
| **4 步** | 4 × 1 = 4（CFG 内化） | ~10 GB | 5-10 sec | 🟢 非常舒适 |
| **1 步** | 1 × 1 = 1 | ~10 GB | 1-2 sec | 🟢 极速 |

**关键观察**：每步 peak VRAM 相同（~10 GB），但总步数少意味着：
1. Wall time 大幅缩短（50 步 5min → 4 步 10s = 30×）
2. Offload 策略的上下文切换次数减少（如果使用 CPU offload）
3. GPU 的总功耗和热量产生减少

### 6.2 CFG 内化的重要性

FLUX schnell 和 SD3 Turbo 的 CFG 内化（蒸馏后 cfg≈0）意味着每步只需一次 forward（不需要 cond+uncond 双 forward）。这带来两个好处：
- 每步时间减半
- 每步显存峰值减半（因为不需要同时存储 cond 和 uncond 的中间激活）

### 6.3 12GB RTX 5070 Ti 可行性

> 🟢 **Few-step sampling 是 12GB 场景的最大利好**。它不降低每步 peak VRAM，但大幅缩短 wall time，同时使本来需要 28+ 步的模型可以在超时前完成。结合 offload，1-4 步推理确保了 12GB 上能完成文生图。

**推荐 few-step 模型列表**（12GB 优先）：

| 模型 | 步数 | VRAM | Wall Time | 推荐度 |
|------|------|------|----------|--------|
| SD3.5 Medium | 28 | ~5 GB | ~1 min | ⭐⭐⭐ |
| SD3.5 Large Turbo | 4 | ~10 GB | ~8 sec | ⭐⭐⭐⭐ |
| FLUX.1-schnell | 4 | ~6 GB | ~5 sec | ⭐⭐⭐⭐⭐ |
| Sana-Sprint-0.6B | 2 | ~7 GB | < 1 sec | ⭐⭐⭐⭐⭐ |
| LTX-Video 2B | 4-8 | ~6 GB | < 1 min | ⭐⭐⭐⭐⭐ |

---

## 7. 对我的 diffusion_engine 的启发

### 7.1 `scheduler.py`
- 需要支持 **non-uniform timestep**（distilled scheduler）。当前 `RectifiedFlowScheduler` 使用均匀 t∈[0,1]，但 distilled 模型的 timestep 分布不均匀（初始几步和最后几步的间隔可能很大）。
- 接口应支持 `timestep_list: List[float]` 而非仅 `num_steps: int`。
- T16 的 scheduler benchmark 应直接比较 full-step（28 步）和 few-step（4 步）的 latency/quality tradeoff。

### 7.2 `pipeline.py`
- Few-step 推理路径：pipeline 应支持 `num_steps=4` 配置，并在内部调用正确的 scheduler。
- CFG 内化路径：pipeline 应能识别 cfg≈0 的情况并跳过 unconditional forward（节省 50% 时间）。

### 7.3 `memory_manager.py`
- Few-step 推理的 allocation 策略：步数少意味着预分配 buffer 的价值更大（buffer 复用次数少，预分配开销占总时间比更大）。
- CFG 内化时不需要 ping-pong buffer（因为不需要 cond+uncond 双 buffer）。

### 7.4 `text_conditioning.py`
- CFG 内化后不再需要 unconditional embedding，text conditioning 模块可以简化。

---

## 8. 查什么 / 读什么 / 输出什么

**查**：
- Consistency Models：`https://arxiv.org/abs/2303.01469`
- LCM：`https://arxiv.org/abs/2310.04378`
- SD Turbo：`https://huggingface.co/stabilityai/sd-turbo`
- FLUX schnell：`https://huggingface.co/black-forest-labs/FLUX.1-schnell`
- Progressive Distillation：`https://arxiv.org/abs/2202.00512`

**读**：
- 各方法的核心思想（每篇读 abstract + method overview）
- 各方法的实际推理步数和 VRAM 报告
- FLUX schnell vs dev 的 diffusers pipeline 源码（理解 CFG 内化的实现）

**输出**：
- 本文档：`learning/papers/10_consistency_distillation_fast_sampling.md`（8 字段完整 + 各方法对比 + 12GB few-step 推荐路线）

---

*阅读日期：2026-06-07 | 状态：已完成 | 对应任务：T8 (Wave 2)*
