# CFG 与 Negative Prompt — 现代扩散模型的条件引导

> **对应任务**：T12
> **产出日期**：2026-06-07
> **前置阅读**：`learning/notes/03_diffusion推理数据流.md`（第 8 节 CFG）、`learning/notes/04_scheduler设计.md`

---

## 1. CFG 是什么

CFG（Classifier-Free Guidance）是现代扩散模型（SD3、FLUX、DALL·E 3 等）中最核心的条件控制机制。它的本质思想是：

> 用"有条件模型的输出"与"无条件模型的输出"之差作为引导方向，放大条件信号，抑制噪声。

**公式**（在 vector field 空间）：

```
v_cfg = v_uncond + s × (v_cond - v_uncond)
```

其中：
- `v_cond`：模型在给定 prompt 条件下的 vector field 预测
- `v_uncond`：模型在空 prompt（或 negative prompt）下的 vector field 预测
- `s`（cfg_scale）：引导强度，典型值 4–10
- `v_cfg`：CFG 调整后的最终 vector field

**为什么在 vector field 层面做 CFG，不在 latent 层面？**

| 层面 | 做法 | 为什么错误 |
|------|------|-----------|
| Vector field（正确） | v_cfg = v_uncond + s × (v_cond - v_uncond) | 调整的是"模型预测的方向"，保持 ODE 积分一致性 |
| Latent（错误） | x_cfg = x_uncond + s × (x_cond - x_uncond) | 在 latent 空间做线性插值，破坏 ODE 轨迹，生成结果变糊或崩溃 |

**核心原因**：rectified flow ODE 的速度场 v_θ(x_t, t) 定义了概率流的方向。CFG 应该调整这个方向，而不是直接修改当前状态。在 latent 上做 CFG 相当于在错误的时间点做插值——当前 latent 是 t 时刻的中间态，不是最终状态。

---

## 2. s 的不同取值 — 边界行为

### s = 1.0（无引导）

```
v_cfg = v_uncond + 1.0 × (v_cond - v_uncond) = v_cond
```

等同于直接使用条件模型输出，不加引导。此时文本条件完全由模型自身处理，无额外放大。

**适用场景**：
- 模型本身的条件跟随能力足够强（如 FLUX schnell 的 few-step 推理）
- 追求"忠实于 prompt"而非"强烈风格化"
- CFG 会导致 artifacts 的极端参数下

### s = 0.0（仅无条件）

```
v_cfg = v_uncond + 0.0 × (v_cond - v_uncond) = v_uncond
```

完全忽略条件信号，等价于无条件生成。

**理论意义**：展示模型在无条件下的"先验"分布。实践中很少使用（除非做 ablation 实验）。

### s < 1.0

```
v_cfg = (1 - s) × v_uncond + s × v_cond
```

介于条件和无条件之间，引导弱于直接条件。

### s > 1.0（强引导）

```
v_cfg = v_uncond + s × (v_cond - v_uncond)
```

条件信号被 **s 倍放大**。这是最常见的使用方式，s 通常取 4–10。

**s 过大的副作用**：
- 图像过度饱和、对比度异常
- "过拟合" prompt 的某些词，忽略整体语义一致性
- 出现高频 artifacts（如棋盘格、鬼影）
- 本质是 ODE 积分中引入了非物理的方向放大

**经验值**：
| 模型 | 推荐 s | 说明 |
|------|--------|------|
| SD3 Medium | 4.5–7.0 | 标准文生图 |
| FLUX.1-schnell | 0.0–2.0 | few-step 模型，CFG 非必需 |
| Sana | 4.0–5.0 | 高效模型，对 s 较敏感 |
| CogVideoX | 6.0 | 视频推理 |

---

## 3. Negative Prompt（负向提示）

### 原理

Negative prompt 是"无条件嵌入"的增强版——不是用空字符串的 embedding 做 v_uncond，而是用**明确的负面描述**（如 "blurry, low quality, distorted"）的 embedding 做 v_uncond。

```
v_cfg = v_negative + s × (v_positive - v_negative)
```

**效果**：将生成结果"推离"负面描述的方向。这是对无条件 CFG 的细粒度扩展。

### 与空字符串 unconditional 的对比

| 类型 | v_uncond 来源 | 效果 |
|------|-------------|------|
| 标准 CFG | 空字符串 "" | 通用引导：远离"平均图像" |
| Negative prompt CFG | 具体负面描述文本 | 目标引导：远离特定风格/内容 |

### 实现注意

在 prompt embedding cache 中，cond 和 uncond（或 negative）是两个独立的 cache entry。这意味着：
- 首次调用 `encode("a cat")` + `encode("")` → 2 次 miss
- 第二次相同调用 → 2 次 hit
- 切换 negative_prompt → 只有 negative 部分是 miss

---

## 4. Batched vs Sequential CFG — 实现维度

### Batched CFG

```python
# 拼接 cond 和 uncond → 一次 forward
latents_cat = torch.cat([latents, latents], dim=0)    # (2B, C, H, W)
text_cat = torch.cat([uncond_emb, cond_emb], dim=0)    # (2B, L, D)
v_cat = denoiser(latents_cat, t, text_cat)             # 一次 forward
v_uncond, v_cond = v_cat.chunk(2, dim=0)
v_cfg = v_uncond + s * (v_cond - v_uncond)
```

**优点**：
- 一次 forward pass（GPU 并行度高）
- 约 1.5× faster than sequential（batch 2× 的 overhead 小于 2×）

**缺点**：
- 显存占用翻倍（batch size = 2B）
- 不适合大分辨率（1024×1024 时 2× batch 可能爆显存）

**适用**：
- 显存充裕时（如 12GB 跑 512×512 以下）
- latency 敏感场景

### Sequential CFG

```python
# 两次 forward
v_uncond = denoiser(latents, t, uncond_emb)  # forward 1
v_cond = denoiser(latents, t, cond_emb)      # forward 2
v_cfg = v_uncond + s * (v_cond - v_uncond)
```

**优点**：
- 显存仅需单 batch
- 适合大分辨率或 VRAM 紧张的场景

**缺点**：
- 两次 forward pass → ~2× 时间
- GPU 利用率低（单 batch 可能喂不饱 GPU）

**适用**：
- 显存紧张时（如 12GB 跑 1024×1024）
- 开发/调试阶段

### 数值等价性

在确定性 ODE（无随机噪声注入）下，batched 和 sequential 产生**完全相同的结果**（浮点误差 < 1e-4）。因为两者使用相同的模型权重、相同的 latent 输入、相同的 timestep，唯一的区别是 batch grouping——在纯推理模式下，batch 不改变数值结果。

> **验证**：`pipeline_smoke.py` 的 `test_sequential_vs_batched_cfg` 测试此等价性。

---

## 5. 双 forward 的显存代价

### 显存构成（单步 forward）

| 组件 | 大小（fp16, 1024×1024） | 说明 |
|------|------------------------|------|
| Latent | 4 × 128 × 128 × 2B = 128KB | 极小 |
| Text embedding | 77 × 4096 × 2B ≈ 0.6MB | CLIP-L 的 hidden_state |
| Model weights | 2–8 GB | 取决于模型大小 |
| Attention activations | 数百 MB–数 GB | N² attention，N = patch 数 |
| AdaLN 中间激活 | 数十 MB | Linear + SiLU 中间态 |

Batched CFG 的额外开销：
- latent batch: 128KB → 256KB（可忽略）
- attention activations: 翻倍（这是大头）
- model weights: 不变（共享）

**结论**：batched CFG 的显存瓶颈在 attention 激活翻倍，而非 latent 本身。对于 2B 参数模型在 1024² 下，sequential 可能是更安全的选择。

---

## 6. 与 LLM 推理中"引导"概念的对比

| 维度 | LLM（如 GPT） | Diffusion（如 SD3） |
|------|-------------|-------------------|
| 引导方式 | system prompt / instruction tuning | CFG（classifier-free guidance） |
| 引导位置 | token logits 上（logit bias） | vector field 上（ODE 方向调整） |
| 负数概念 | 不常见（通常不做"负向 token 推离"） | negative prompt 是标准 feature |
| 实现代价 | 极低（仅修改 logits） | 需额外一次 model forward（sequential）或翻倍 batch（batched） |
| 调优参数 | temperature / top-p / repetition_penalty | cfg_scale / negative_prompt |

**关键差异**：LLM 的引导是在离散 token 空间的对数概率上操作，而扩散的 CFG 是在连续向量场（probabilistic flow）上操作。这导致扩散的 CFG 计算代价远高于 LLM 的 logit bias。

---

## 7. 实现中的注意事项

1. **CFG 在 scheduler.step 之前**：先在 vector field 层面完成 CFG，再用调整后的 v_cfg 驱动 ODE step。绝对不要在 latent 更新后做 CFG 插值。

2. **cfg_scale = 1.0 的优化**：此时无需 encode negative_prompt，也无需 uncond forward，直接使用 cond forward 输出。pipeline 应处理此 fast path。

3. **float16 精度注意**：CFG 公式 `v_uncond + s * (v_cond - v_uncond)` 在 fp16 下要注意数值范围。如果 s 很大（如 20+），`v_cond - v_uncond` 的小差异可能被放大到损失精度。

4. **与 prompt embedding cache 的交互**：cond 和 uncond embedding 都在 cache 中独立存储，确保 batch CFG 和 sequential CFG 都能命中。
