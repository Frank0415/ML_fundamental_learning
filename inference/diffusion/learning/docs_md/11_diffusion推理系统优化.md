# 11 - Diffusion 推理系统优化

Diffusion 的主要开销来自重复 denoising、attention 中间矩阵和 VAE decode，优化位置与 LLM 不同。本章用 T16、T17 的实验数据检查六项具体做法。

## 1. Prompt Embedding Cache（提示词嵌入缓存）

扩散模型在推理期间的 text encoding 只执行一次：输入的 prompt 经 T5/CLIP 编码为固定长度的 embedding，在后续所有 denoising step 中复用。如果用户连续生成多张图像，且 prompt 不变，那 text encoding 就是完全重复的计算。

### 实验数据（T16）

| 指标 | 无缓存 | 有缓存 | 节省 |
|------|-------|-------|------|
| 总延迟 | 0.7544 s | 0.3711 s | **50.8%** |
| 平均每次延迟 | 7.544 ms | 3.711 ms | - |
| allocation 次数 | 300 | 100 | 200 次 |
| Cache hit ratio | - | **52%** | - |

**Cache key 设计**：model_id + text_encoder hash + prompt hash + negative_prompt hash + max_sequence_length + dtype + device/offload 策略。这确保了跨模型隔离，一个 FLUX-schnell 的 cache 不会被 SD3-Medium 误命中。实验中验证了跨模型隔离：10 条共享 prompt 在两个模型间全部正确 cache miss。

**限制**：embedding cache 能省掉 text encoding，但这部分通常只占总推理时间的 1-5%，主要耗时仍在 denoiser loop。实际收益还包括减少重复显存分配和 CPU→GPU 传输。

## 2. Latent Buffer Manager（潜空间缓冲管理器）

扩散推理的去噪循环中，每步 denoiser forward 的输出（噪声预测/矢量场）都要写回当前 latent。如果每步都用 `torch.empty()` 创建新 tensor，会产生 GPU 显存碎片。Latent buffer manager 通过预分配、in-place reset 和 ping-pong swap 避免重复分配。

### 实验数据（T16）

| 模式 | 分配次数 | 峰值已分配 | 每步平均延迟 |
|------|---------|-----------|-------------|
| In-place reset | 5 | 327,680 B | 0.231 ms |
| Out-of-place reset | 61 | 327,680 B | 0.387 ms |

**主要结论**：in-place reset 节省了 **91.8%** 的 allocation 次数（5 vs 61），同时每步延迟也低 40%（0.231 vs 0.387 ms）。峰值已分配内存不变（两者都预分配，区别在于是否复用），但 allocation 次数的减少直接转化为更少的 CUDA 同步和更稳定的推理延迟。

## 3. Scheduler Step Profiling（调度器步数分析）

scheduler 的选择直接影响推理质量与速度的平衡。本 benchmark 对比了 Euler（等步长）和 RectifiedFlow（log-linear 步长）在 4/8/16/28/50 步下的表现。

### 实验数据（T16）

| 步数 | Euler (ms) | RF (ms) | 差异 | 适用场景 |
|------|-----------|---------|------|---------|
| 4 | 0.34 | 0.34 | 0% | 蒸馏模型（schnell/turbo/sprint） |
| 8 | 0.65 | 0.67 | -3.08% | Lightning / few-step（Hyper-SD） |
| 16 | 1.37 | 1.37 | 0% | 中等步数：性价比最优 |
| 28 | 2.35 | 2.36 | -0.43% | SD3 / FLUX-dev 默认 |
| 50 | 4.19 | 4.22 | -0.72% | 高步数：边际收益递减 |

**主要发现**：延迟与步数近似线性（\(R^2 \approx 1.000\)）。\(50\) 步 \(\approx 4\) 步 \(\times 12.5\)。Euler 和 RectifiedFlow 每步都执行一次 denoiser forward + ODE update，计算量接近，主要差异是步长间距。在这组实验中，减少步数比切换 scheduler 类型更能降低延迟。

## 4. CFG Batching（分类器无关引导批处理）

CFG（Classifier-Free Guidance）在每步需要跑 **两次** denoiser forward：一次 conditional（带 prompt embedding），一次 unconditional（空 prompt embedding）。结果按 \(v_{\mathrm{cond}} + \mathrm{cfg\_scale}\left(v_{\mathrm{cond}} - v_{\mathrm{uncond}}\right)\) 组合。

两种策略：**Sequential CFG** 分别执行 cond 和 uncond forward；**Batched CFG** 将两组 latent 拼成双倍 batch，用一次 forward 同时处理。

### 实验数据（T17）

| CFG Scale | Sequential (ms/步) | Batched (ms/步) | 加速比 | Seq 显存 | Batch 显存 |
|-----------|-------------------|-----------------|--------|----------|------------|
| 1.0 | 9.00 | 8.82 | 1.02× | 1.5 MB | 2.0 MB |
| 3.0 | 8.97 | 8.82 | 1.02× | 1.5 MB | 2.0 MB |
| 7.5 | 8.97 | 8.89 | 1.01× | 1.5 MB | 2.0 MB |
| 15.0 | 9.00 | 8.92 | 1.01× | 1.5 MB | 2.0 MB |

数值差异 **0.00e+00**（理论上完全一致，微小差异仅来自浮点 accumulate order）。中等显存配置 策略：剩余 > 6GB → Batched CFG；3-6GB → 视具体情况；< 3GB → Sequential CFG。

## 5. Attention Memory（注意力显存估算）

这是扩散推理中 **最大的瓶颈**。DiT/MMDiT 的 full attention 复杂度为 \(O(N^2)\)，其中 \(N\) 是 latent token 数。1024×1024 图像 → latent 128×128 → patch_size=2 → 4096 tokens → attention matrix 32 MB (fp16)，可接受。但 \(2048^2\) → 16384 tokens → 512 MB。

### 实验数据（T17）

| 场景 | Tokens | Attn Matrix (fp16 MB) | Total Standard (MB) | Mem-Eff 估算 (MB) | 节约比 |
|------|--------|----------------------|--------------------|--------------------|---------|
| DiT 512² (latent 64×64, p=2) | 1024 | 2.0 | 12.0 | 10.0 | 1.2× |
| SD3 1024² (128×128, p=2) | 4096 | 32.0 | 72.0 | 40.1 | 1.79× |
| SD3 2048² (256×256, p=2) | 16384 | 512.0 | 672.0 | 160.5 | 4.19× |
| CogVideoX 49f 480p | 11520 | 253.125 | 365.6 | 112.9 | 3.24× |

\(O(N^2)\) 复杂度验证：\(N \to 2N\) 时，\(N^2 \to 4N^2\)。\(512^2\) 图像用 standard attention 安全，\(1024^2\) 推荐 memory-efficient attention，\(2048^2\) 和长视频 **必须** memory-efficient attention。

## 6. VAE Tiling（VAE 分块解码）

VAE tiling 在应用层把 latent 或像素切成多个 tile，逐块 decode，再用 overlap blending 处理接缝。它属于显式 chunk decode，与 torch.compile、flash-attn 解决的不是同一层问题。

### 实验数据（T17）

| 分辨率 | Mode | Tiles | 延迟 (ms) | 峰值显存 (MB) | vs Full 速度 | vs Full 显存 |
|--------|------|-------|----------|-------------|-------------|--------------|
| 1024² | Full | 1 | 29.45 | 18.0 | - | - |
| 1024² | Tiled 16×16 | 64 | 66.93 | 24.5 | 2.27× | 1.36× |
| 1024² | Tiled 64×64 | 4 | 36.61 | 28.4 | 1.24× | 1.58× |
| 2048² | Full | 1 | 117.53 | 72.0 | - | - |
| 2048² | Tiled 32×32 | 64 | 198.94 | 97.4 | 1.69× | 1.35× |

典型开销：overlap=4 latent pixels → 额外 10-20% 计算量。受限显存配置下 VAE tiling **几乎总是必要的**。推荐 tile 大小：1024² → 64×64 latent tile；2048² → 32×32 latent tile。

## 7. 与 LLM KV Cache 的根本差异

> **优化对象**
>
> Diffusion 每一步都会刷新 latent，因此不能照搬 LLM 的跨 token KV cache。这里需要控制的是 attention activation、去噪步数和每步的双 forward 开销。

| 对比维度 | LLM（自回归解码） | Diffusion（迭代去噪） |
|---------|------------------|---------------------|
| **状态积累** | 每步追加 1 个 token，历史 token 的 key/value 可复用（KV cache 线性增长 \(O(N)\)） | 每步 latent 全部刷新，无法复用上一步的计算结果（这是根本差异） |
| **注意力复杂度** | GQA + causal mask，每个新 token 只需 attend 历史 token（\(O(N)\) per step） | Full attention，所有 token 两两 attend（\(O(N^2)\) per step） |
| **缓存内容** | KV cache：存储历史 token 的 key/value projection | Prompt embedding cache：存储 text encoder 输出（每 session 只算 1 次） |
| **分页技术** | PagedAttention：将 KV cache 分页管理，类似 OS 虚拟内存 | Latent buffer manager：预分配 + ping-pong swap + in-place reset |
| **批处理** | Continuous batching：动态合并请求的相同 phase | CFG batching：cond+uncond 拼接为双倍 batch，一次 forward |
| **显存瓶颈** | KV cache（历史 context 越长，显存越大） | Attention matrix（分辨率越高，\(O(N^2)\) 爆炸） |

LLM 的 KV cache 依赖一个前提：旧 token 保持不变，新 token 只需读取历史。扩散每个 denoising step 都会更新整块 latent，因此上一轮的 K/V 不能直接作为下一轮缓存。两者一个追加状态，一个重写状态。

那哪里相似？在 prompt/text encoder 层面。扩散和 LLM 推理的第一步都是 text encode，这个步骤的 cache 策略是通用的：相同的 prompt → 相同的 embedding → 避免重复编码。这一点两者都会做。但进入主循环后，优化路径就完全不同了。

> **结论**
>
> 本章测了六项优化：prompt embedding cache（52% hit ratio）、latent buffer manager（91.8% allocation 节省）、scheduler profiling（线性 scaling）、CFG batching（sequential 与 batched 的取舍）、attention memory（\(O(N^2)\)）和 VAE tiling。LLM 逐步追加 token 历史，扩散则在每一步刷新整块 latent，所以 denoiser 内部不能照搬 LLM KV cache。prompt encoding 仍可共享缓存，两类模型也都要处理 attention 的 \(O(N^2)\) 开销。

## 和我的 diffusion_engine 的关系

本章的所有优化都可以在 `diffusion_engine/` 中找到对应的实现模块。prompt embedding cache 接口在 `core/text_conditioning.py`，latent buffer manager 在 `core/memory_manager.py`，CFG batching 逻辑在 `core/pipeline.py` 的 denoising loop 中，VAE tiling 在 `core/vae_stub.py`。实验脚本在 `experiments/diffusion_inference_optimization/` 下，所有数据来自 2026-06-07 的 benchmark 运行。scheduler 的实现细节见 04 页，attention 的 DiT/MMDiT 实现见 05/06 页。
