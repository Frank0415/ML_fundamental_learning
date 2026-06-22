# Latent Buffer 与显存预算 — 扩散推理的显存管理

> **对应任务**：T12
> **产出日期**：2026-06-07
> **前置阅读**：`learning/notes/07_text_encoder和prompt_embedding_cache.md`（第 5 节 cache 对比）

---

## 1. 为什么扩散推理需要 Latent Buffer Manager

### 直觉：每步 latent 全刷新 → 重复 malloc/free

扩散去噪循环的核心操作：

```python
for step in range(N):
    v = denoiser(latent, t, text_emb)
    latent = latent + dt * v  # 旧 latent 被覆盖
```

**naive 实现的问题**：

每次 `latent = latent + dt * v` 时，PyTorch 会：
1. 为 `dt * v` 分配新 tensor
2. 为 `latent + ...` 分配新 tensor
3. 释放旧 latent tensor

在 28 步循环中，这意味着约 28 × 3 = 84 次 GPU malloc/free。每次 cudaMalloc 的固定开销约 10–50 μs（取决于碎片化程度），累计约 1–4 ms。虽然不大，但在 latency-sensitive 场景中值得优化。

### 真正的收益：显存可预测性

更大的收益不是性能，而是**显存可预测性**。预分配 buffer 后，整个 denoising loop 的显存占用是确定的：

```
预分配 buffer 池: 5 × latent_size
+ model weights
+ text embedding
+ attention activations
= 总显存占用（完全可预测）
```

不必担心 loop 中间的"峰值显存"超过预算，因为 buffer 已在构造时分配好。

---

## 2. Latent Buffer Manager 设计

### 5 个 buffer 的语义

| Buffer 名 | 用途 | 初始化 | 更新时机 |
|-----------|------|--------|---------|
| `x_t` | 当前 latent | zeros → noise（首次） | 每步后 swap 得到 x_next |
| `x_next` | 下一步 latent | zeros | 每步 denoiser+scheduler 后写入 |
| `noise` | 初始噪声 | randn(seed) → 存储 | reset() 时刷新 |
| `cfg_cond` | CFG 条件结果 | zeros | 每步 CFG 后写入 v_cond |
| `cfg_uncond` | CFG 无条件结果 | zeros | 每步 CFG 后写入 v_uncond |

**x_t 和 x_next 的 ping-pong 机制**：

```
初始状态:
  x_t  = noise
  x_next = zeros

Step 0:
  计算 x_next = x_t + dt * v
  swap("x_t", "x_next")  → x_t 成为更新后的 latent
                          → x_next 成为旧 x_t（可被覆盖）

Step 1:
  计算 x_next = x_t + dt * v  # 写入"旧 x_t"的位置
  swap("x_t", "x_next")
  ...
```

**不复制数据**：swap 交换的是 Python 引用（tensor 指针），不是 tensor 数据。O(1) 操作。

### 构造参数

```python
class LatentBufferManager:
    def __init__(self,
        image_shape=(1, 4, 64, 64),     # (B, C, H, W)
        video_shape=None,               # (B, C, T, H, W) 优先
        device="cpu",
        dtype=torch.float32,
        seed=0,
    ):
```

**image vs video shape**：
- 图像 latent: `(B=1, C=4, H, W)` — 4D
- 视频 latent: `(B=1, C=16, T, H, W)` — 5D（CogVideoX 有 16 个 latent channels）
- 视频场景下 buffer 占用量显著增加（T 维度可达 49–81 帧）

---

## 3. Ping-pong Buffer 的数学解释

### 为什么是 2 个 buffer 而不是 3 个？

标准 ping-pong buffer 只需要 2 个：

```
         写入               读取
Step k: [x_next]  ← f(x_t)    x_t  ← [读 buffer A]
                                ↓
        swap → [x_next] 写指针 ↔ [x_t] 读指针
                                ↓
Step k+1: [x_next] ← f(x_t)    x_t ← [读 buffer B（原 A 的内容）]
```

**不需要第 3 个 buffer** 因为：
1. 当前 latent (x_t) 在作为输入传递给 denoiser 时不需要修改
2. denoiser 输出（v）与 latent 不同 shape（都是 (B,C,H,W) 但语义不同）
3. scheduler step（x_next = x_t + dt * v）可以直接写入 x_next buffer

**为什么不是 in-place 更新？**
理论上可以 `latent += dt * v`，但：
- 破坏了"旧 x_t 可用于 debug / checkpointing"的机会
- 如果 denoiser 内部需要引用原始 latent（如某些 normalization），in-place 可能导致错误
- ping-pong 的额外显存成本仅 1× latent_size，可忽略

---

## 4. 中等显存配置分账 — 真实推理预算

### 假设：SD3 Medium @ 1024×1024, fp16, 28 steps

| 组件 | 大小（估算） | 说明 |
|------|------------|------|
| **Model Weights** | | |
| MMDiT (2B params) | 4.0 GB | fp16: 2B × 2bytes |
| CLIP-L (430M) | 0.86 GB | text_encoder |
| CLIP-G (1.3B) | 2.6 GB | text_encoder_2 |
| VAE Decoder (80M) | 0.16 GB | 解码器权重 |
| **小计** | **~7.6 GB** | |
| | | |
| **Latent Buffer** | | |
| 5 × (1,4,128,128) fp16 | 5 × 128KB = 0.64 MB | 可忽略 |
| CFG 额外 batch | +1 × 128KB = 0.13 MB | batched CFG 翻倍 batch |
| **小计** | **< 1 MB** | latent buffer 不是瓶颈 |
| | | |
| **Text Embedding** | | |
| CLIP-L seq (77, 768) fp16 | ~118 KB | |
| CLIP-G pooled (1280) fp16 | ~2.5 KB | |
| **小计** | **< 1 MB** | |
| | | |
| **Attention Activations** | | |
| MMDiT self-attention (24 layers) | ~1–2 GB | N² attention, N=4096 patches per 128² latent |
| Cross-attention (24 layers) | ~0.5–1 GB | Q·K^T 中间激活 |
| **小计** | **~1.5–3 GB** | 这是真正的显存瓶颈！ |
| | | |
| **其他** | | |
| PyTorch CUDA context | ~0.5 GB | cuBLAS/cuDNN 内部 buffer |
| 系统保留 | ~0.5 GB | NVIDIA driver 占用 |
| **小计** | **~1 GB** | |
| | | |
| **总计** | **~10–12 GB** | 非常接近 在中档显存卡的上限！ |

### 关键发现

1. **Latent buffer 占用 < 1 MB** — 不是显存瓶颈。Ping-pong buffer 机制主要解决 malloc 碎片化，而非显存节省。

2. **Model weights 占大头（~7.6 GB）** — 这也是为什么 CPU offload（把 text encoder 或 VAE offload 到 CPU）是 受限显存推理的必要策略。

3. **Attention activations 是第二大开销（~1.5–3 GB）** — N² attention 在高分辨率下爆炸。1024² latent 有 4096 patches，self-attention = (4096²) × head_dim × num_heads × num_layers。

4. **T5-XXL（22GB）根本放不下** — SD3 Medium 的 optional T5 必须完全跳过，或使用 CPU offload。

### 受限显存策略

```
总预算: 中等显存配置 × 0.85 = 10.2 GB（留 15% 给驱动和碎片）

优先级 1: 确保 model weights 不超
  → fp16 存储所有模型
  → MMDiT 用 fp16（4GB），CLIP-L+G 用 fp16（3.5GB）
  → 小计 7.5GB，剩余 2.7GB

优先级 2: 控制 attention activations
  → 降分辨率：1024² → 768²（patches 从 4096 降为 2304）
  → 或使用 attention slicing / flash attention

优先级 3: 减少 batch size
  → sequential CFG（单 batch）而非 batched CFG
  → VAE tiling（把 decode 的大激活切块）

优先级 4: CPU offload
  → text encoder 用完即 offload 到 CPU（省 3.5GB）
  → VAE decoder 用完即 offload
```

---

## 5. MemoryStats 接口的设计理由

### 为什么不用 torch.cuda.memory_summary()

`torch.cuda.memory_summary()` 输出非常详细但**机器可读性差**（是一个格式化的多行字符串）。我们需要一个程序化接口：

```python
class MemoryStats:
    peak_allocated: int       # CUDA: max_memory_allocated()
    peak_reserved: int        # CUDA: max_memory_reserved()
    current_allocated: int    # CUDA: memory_allocated()
    allocation_count: int     # CUDA: memory_stats()["allocation.all.current"]
```

### 后端兼容性

| 后端 | peak_allocated | peak_reserved | allocation_count |
|------|---------------|---------------|-----------------|
| CUDA | ✅ torch.cuda.max_memory_allocated() | ✅ torch.cuda.max_memory_reserved() | ✅ memory_stats() |
| MPS | ❌ 不支持 | ❌ 不支持 | ❌ 不支持 |
| CPU | ✅ tracemalloc.get_traced_memory()[1] | ❌ | ❌ |

**MPS 的困境**：Apple MPS 后端不提供显存统计 API。在 Mac 上开发时，MemoryStats 返回 0 或 fallback 到 tracemalloc（跟踪 Python 分配而非 GPU 分配）。

这导致了 M5 开发 + 可用的 CUDA GPU 远程推理的双轨策略的必要性。

---

## 6. 扩散独有 vs LLM 通用 — 内存管理的差异总结

### 扩散独有

| 特性 | 说明 |
|------|------|
| **Latent 全刷新** | 每步 latent 完全被新值覆盖，无"历史"需要保存 |
| **Text embedding 不变** | 所有 denoising step 用同一份 text embedding → cache 是高杠杆优化 |
| **Attention 无因果 mask** | N² 激活无法被 KV cache 规避 |
| **Ping-pong 划算** | 2 个 buffer 交替使用即可覆盖整个 loop，不增加额外显存 |

### LLM 通用

| 特性 | 说明 |
|------|------|
| **KV cache 是必须** | 过去 token 的 K/V 在 autoregressive decoding 中不可丢弃 |
| **Prefix cache** | 跨请求共享 system prompt / prefix 的 KV |
| **Paged attention** | 将 KV cache 分页管理以消除碎片 |
| **Continuous batching** | 动态聚合不同请求的 forward 以提高 GPU 利用率 |

### 为什么不能把 LLM 的 KV cache 模式搬到扩散

试图在扩散中保留"过去 step 的 latent"（类比 KV cache）会：
1. **浪费显存**：旧 latent 对下一步的 ODE 积分无意义
2. **数值错误**：ODE 积分需要当前 latent 和 vector field，不是历史 latent 的聚合
3. **概念混淆**：扩散的"step"不是 LLM 的"token"——step 之间不是追加关系，而是替换关系

**正确思路**：扩散的内存优化应聚焦于模型权重的 offload 和 attention 激活的压缩，而非 latent 历史的缓存。Ping-pong buffer 只是工程实现上的整洁性优化，不是显存节省的核心。
