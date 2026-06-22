# Diffusion 推理系统优化实验

> **负责任务**：T16（prompt cache / latent buffer / scheduler benchmark）+ T17（CFG batching / attention memory / VAE tiling）
> **执行环境**：Mac M5（开发测试）+ 可用的 CUDA GPU（真实 benchmark）
> **最后更新**：2026-06-07

---

## 1. 目标：Diffusion 推理系统优化与 LLM 的根本差异

### 1.1 LLM 推理优化（本项目的参照系）

LLM 的推理优化集中在**自回归生成**这一单一范式上：

| 优化技术 | 解决什么问题 | 为什么有效 |
|---------|-------------|-----------|
| **KV cache** | 避免每步重算过去 token 的 key/value | token 序列一旦生成就不会变，可无限追加 |
| **Paged attention** | KV cache 碎片化 | 将连续 cache 分块，按需分配/释放 |
| **Continuous batching** | GPU 利用率低 | 多个请求共享同一 forward pass |
| **Speculative decoding** | 每 token 延迟高 | 用小模型"猜"多个 token，大模型批量验证 |
| **Prefix cache** | 长 prompt 多轮复用 | 共享 prefix 的 key/value 跨请求复用 |

**核心前提**：过去 token 的 hidden state 是"不变的"——这是 KV cache 存在的第一性原理。

### 1.2 Diffusion 推理优化（本目录的研究对象）

扩散模型推理**完全不是**自回归范式，不存在"过去 token 不变"这一前提：

| 优化方向 | 解决什么问题 | 为什么有效 | LLM 有对应吗 |
|---------|-------------|-----------|------------|
| **Prompt embedding cache** | 同一 prompt 重复走 text encoder | text encoder 输出在 denoising loop 内不变 | 部分对应 prefix cache，但粒度完全不同 |
| **Latent buffer 预分配** | 每步 malloc/free latent 张量 | 预分配 + ping-pong 避免碎片化 | 无——LLM hidden state 增量追加 |
| **CFG batching** | cond/uncond 两次 forward | 拼接 batch 做一次 forward | 类似 continuous batching 但含义不同 |
| **Scheduler step 选择** | ODE 积分步数与质量/速度平衡 | 少步=faster，多步=higher quality | 无——LLM 无 ODE 积分 |
| **VAE tiling** | 高分辨率 decode 显存爆炸 | 分 tile decode 再拼接 | 无——LLM 输出 token 无需 decode 大张量 |
| **Attention memory 控制** | N² attention 在 4K 图有 65536² 矩阵 | linear attention / flash-attn / chunking | partial——LLM 也有 attention 优化但 scale 不同 |

> **★★★ 核心认知：Diffusion 的主优化不是 LLM 的 KV cache ★★★**
>
> - LLM KV cache 存储的是 attention 层的 key/value 历史，用于自回归生成中跨 token 步复用。
> - Diffusion denoising 每步 latent 全刷新——上一步的 K/V 没有复用价值。
> - Diffusion 真正的优化焦点是：
>   1. **Prompt embedding cache**（text encoder 输出缓存，非 attention KV cache）
>   2. **CFG batched forward**（一次 forward 处理 cond+uncond，省掉一次前向）
>   3. **Latent buffer 预分配**（避免 per-step malloc，非 paged attention）
>   4. **Scheduler step 选择**（ODE 积分步数与质量的平衡）
>   5. **VAE tiling**（高分辨率 decode 显存控制）
>   6. **Video chunking**（长视频帧数显存控制）

### 1.3 在 显存预算下的优化优先级

| 优先级 | 优化项 | 预计收益 | 中等显存配置 影响 |
|-------|--------|---------|----------|
| P0 | Prompt embedding cache | ~1–3 GB（避免重复加载 text encoder） | 决定性 |
| P0 | CFG batching | ~1.3× 加速（省一次 forward） | 显存代价 ~1.8×（双倍 batch） |
| P1 | Scheduler 选择 | 4→50 步，速度差 10× | 无额外显存 |
| P1 | Latent buffer 预分配 | 碎片化消除，~1–2ms per run | 可忽略（~1 MB） |
| P2 | VAE tiling | 1024²→2048² 可行 | 解码峰值降低 50%+ |
| P2 | Attention memory | O(n²)→O(n) linear attn | 4K 图必需 |

---

## 2. 本目录实验清单

### 2.1 T16 — 三个基础优化实验（本任务）

| # | 脚本 | 核心目标 | 关键指标 |
|---|------|---------|---------|
| 1 | `prompt_embedding_cache.py` | 缓存 text encoder 输出，避免重复编码 | cache hit ratio, saved latency, saved allocation count |
| 2 | `latent_buffer_manager.py` | 管理 latent 张量预分配与复用 | allocation_count, peak_allocated, peak_reserved, latency_per_step |
| 3 | `scheduler_step_benchmark.py` | 比较不同 ODE 步数的速度/质量 | latency_per_step, total latency, scheduler 类型差异 |

### 2.2 T17 — 三个高级对照实验

| # | 脚本 | 核心目标 | 关键指标 |
|---|------|---------|---------|
| 4 | `cfg_batching_experiment.py` | sequential vs batched CFG | latency, peak VRAM, numerical difference |
| 5 | `attention_memory_benchmark.py` | 不同配置下 attention 显存估算 | activation_size, peak memory, tokens |
| 6 | `vae_tiling_experiment.py` | tiling vs full decode | VRAM, throughput, visual diff |

---

## 3. 执行顺序

1. **T16（本任务）**：先实现前 3 个脚本——`prompt_embedding_cache.py`、`latent_buffer_manager.py`、`scheduler_step_benchmark.py`
2. **T17**：再实现后 3 个脚本——`cfg_batching_experiment.py`、`attention_memory_benchmark.py`、`vae_tiling_experiment.py`
3. **T18**：汇总所有实验结果到最终报告

所有脚本均为独立可运行文件，无跨脚本导入依赖，可按任意顺序单独运行。

---

## 4. 如何运行

### 4.1 环境要求

- Python ≥ 3.13
- **仅需 numpy**（`pip install numpy` 或 `uv pip install numpy`）
- **不依赖 torch、diffusers、transformers**——所有实验均为纯 numpy + 模拟

### 4.2 单个脚本运行

```bash
# 1. Prompt embedding cache 实验
python experiments/diffusion_inference_optimization/prompt_embedding_cache.py \
  --demo --num_prompts 100 --repeat_ratio 0.3 \
  --output_dir experiments/diffusion_inference_optimization/results

# 2. Latent buffer manager 实验
python experiments/diffusion_inference_optimization/latent_buffer_manager.py \
  --demo --num_steps 28 --image_shape 1 4 64 64 \
  --output_dir experiments/diffusion_inference_optimization/results

# 3. Scheduler step benchmark 实验
python experiments/diffusion_inference_optimization/scheduler_step_benchmark.py \
  --demo --step_list 4 8 16 28 50 \
  --output_dir experiments/diffusion_inference_optimization/results
```

### 4.3 查看帮助

每个脚本都带有完整的 `--help`：

```bash
python experiments/diffusion_inference_optimization/prompt_embedding_cache.py --help
python experiments/diffusion_inference_optimization/latent_buffer_manager.py --help
python experiments/diffusion_inference_optimization/scheduler_step_benchmark.py --help
```

---

## 5. 结果目录

所有实验结果输出到 `experiments/diffusion_inference_optimization/results/`：

```
results/
├── prompt_cache_<timestamp>.json        # prompt cache hit/miss/latency 数据
├── latent_buffer_<timestamp>.json       # latent buffer 预分配 vs 动态分配统计
├── scheduler_benchmark_<timestamp>.json # scheduler 对比数据（JSON）
├── scheduler_benchmark_<timestamp>.md   # scheduler 对比表格（Markdown 可读）
└── summary_<timestamp>.md              # 实验汇总结论（T17 完成后）
```

时间戳格式：`YYYYMMDD_HHMMSS`。

---

## 6. 各实验详细设计

### 6.1 Prompt Embedding Cache（`prompt_embedding_cache.py`）

**Cache key 设计**（≥7 字段）：

```
key = PromptEmbeddingCacheKey(
    model_id,            # 模型 ID（如 "SD3-Medium" / "FLUX-schnell"）
    tokenizer_hash,      # tokenizer config 的 SHA256 前 16 位
    text_encoder_hash,   # text encoder config 的 SHA256 前 16 位
    prompt,              # 正向 prompt 文本
    negative_prompt,     # 负向 prompt 文本
    max_sequence_length, # 最大 token 序列长度
    dtype,               # 数据类型（float32/float16/bfloat16）
    device,              # 设备（cpu/cuda/mps）
    offload_strategy,    # offload 策略（cpu/cuda/none）
)
```

**为什么 ≥7 字段是必需的**：

- 不同 `model_id` 必须 miss：不能用 FLUX 的 cache 喂 SD3
- 不同 `tokenizer_hash` / `text_encoder_hash` 必须 miss：tokenizer 或 text encoder 换版本后 embedding 不可互换
- 不同 `max_sequence_length` 必须 miss：截断长度不同，embedding shape 不同
- 不同 `dtype` / `device` / `offload_strategy` 必须 miss：tensor 数据和位置都不同
- `negative_prompt` 必须参与 key：空字符串 "" 和 "ugly" 产生不同的 uncond embedding

**Demo 设计**：

- 用 `ToyTextConditioner`（T12）的接口 mock
- 模拟 100 次 prompt 调用，30% 重复（模拟产品图片/固定 prompt 场景），70% 唯一
- 记录：cache hit ratio、latency with cache vs without、saved allocation count
- 对比不同 model_id 之间的 cache 隔离行为

### 6.2 Latent Buffer Manager（`latent_buffer_manager.py`）

**Buffer shape 支持**：

```
image:  (B, C, H, W)      如 (1, 4, 64, 64) = 16,384 元素
video:  (B, C, T, H, W)   如 (1, 4, 16, 32, 32) = 65,536 元素
tokens: (B, N, D)          如 (1, 256, 64) = 16,384 元素  (patch 后)
```

**Buffer 管理**：

- 构造时预分配固定数量 buffer（`x_t`、`x_next`、`noise`、临时 buffer）
- `get(name)`：零拷贝返回 numpy view
- `swap(name1, name2)`：ping-pong 交换，避免 per-step copy
- `reset(name, generator)`：in-place 用同一 buffer 重新 init 噪声
- `out_of_place_reset(name, generator)`：分配新 buffer（对照组）
- `stats()`：allocation_count、current_allocated、peak_allocated、fragmentation_estimate

**Demo 比较**：

- **A. In-place reset**：每次 step 在预分配 buffer 上直接覆盖——零额外分配
- **B. Out-of-place reset**：每次 step 分配新 buffer → 旧 buffer 被 GC——大量 malloc/free
- 模拟 28 步推理，记录 allocation_count、peak_allocated bytes、每步 latency

**显存预算下的真实占比**：

- 1024² latent (fp16, 16ch)：2,097,152 元素 × 2B = 4 MB × 4 buffers = **16 MB**
- 真正瓶颈在 attention activations：4096 token × 4096 × 16 heads × 4B = **~1 GB per layer**
- Latent buffer 预分配解决的是**碎片化**问题，不是显存总量问题

### 6.3 Scheduler Step Benchmark（`scheduler_step_benchmark.py`）

**对比的 step 数及其含义**：

| Step 数 | 对应场景 | 质量估计 | 适用模型 |
|---------|---------|---------|---------|
| **4** | distilled-only / turbo | 低但快（~0.5s） | FLUX-schnell, SD3-Turbo, Sana-Sprint |
| **8** | lightning / few-step | 中低（~1s） | SD3-Lightning, Hyper-SD |
| **16** | 中间档 | 中等（~2s） | 少步质量基线 |
| **28** | SD3 default | 中高（~3.5s） | SD3-Medium, SD3.5-Medium |
| **50** | 高步数（可能过拟合） | 最高（~6.5s） | SD3-Large, FLUX-dev 高质量模式 |

**对比的 scheduler 类型**：

- `EulerScheduler`（sigma 空间，log-linear 间隔）——传统 DDIM/score-based 路线
- `RectifiedFlowScheduler`（t∈[0,1] 空间，线性间隔）——SD3/FLUX 路线
- 相同 step 数下两者约有 ±10% 的 latency 差异（步长间距不同导致每步计算量微差）

**模拟方式**：

- 不调用真实 denoiser（避免 torch 依赖）
- 用简单 numpy 计算模拟 denoiser forward（矩阵乘法 + 激活函数）
- 每步记录耗时

---

## 7. 关键设计原则

### 7.1 零 torch 依赖

所有实验脚本**不 import torch**，使用纯 `numpy` + Python 标准库。理由：

- 在 Mac M5（Metal，无 CUDA）上开发时 torch 对 MPS 的支持不稳定且行为不可预测
- 本批实验的核心是**指标设计和对比逻辑**，不是真实 GPU 性能
- T14 的 reference image inference 会引入 torch，T16/T17 的数值模拟先行验证设计正确性

### 7.2 Cache key 不是 prompt 字符串

```python
# ❌ 错误设计（diffusion_engine 已经避免的问题）
cache_key = prompt  # 只含 prompt 字符串 → 不同模型/设备/长度会碰撞

# ✅ 正确设计（T12/T16 强制要求）
cache_key = PromptEmbeddingCacheKey(
    model_id="SD3-Medium",
    tokenizer_hash="a1b2c3d4",      # SHA256 前 8 字节
    text_encoder_hash="e5f6g7h8",
    prompt="a cat",
    negative_prompt="",
    max_sequence_length=77,
    dtype="float16",
    device="cuda",
    offload_strategy="cpu",
)
```

### 7.3 所有注释中文，代码标识符英文

遵循项目全局约定：代码中变量名、函数名、类名使用英文，注释和 docstring 使用中文。

---

## 8. 失败/Blocker 模板

```markdown
# Diffusion Inference Optimization — Blocker

**日期**：YYYY-MM-DD
**实验**：[prompt_cache / latent_buffer / scheduler]
**设备**：[cpu / mps / cuda]
**Python 版本**：3.x.x

## 错误类型
[数值错误 / 环境不支持 / 结果不符合预期]

## 复现步骤
python experiments/diffusion_inference_optimization/<script>.py --demo ...

## 排查路径
1. [已尝试的诊断步骤]
2. [排除的可能原因]

## 后续建议
[修复方案或替代路线]
```

---

## 9. 与 `diffusion_engine/` 的关系

| 本实验脚本 | 使用的 `diffusion_engine/core/` 接口 |
|-----------|--------------------------------------|
| `prompt_embedding_cache.py` | ToyTextConditioner（T12）——mock text encoder |
| `latent_buffer_manager.py` | LatentBufferManager 接口参考（T12）——纯 numpy 重实现 |
| `scheduler_step_benchmark.py` | EulerScheduler、RectifiedFlowScheduler（T10）——直接 import |

> **注意**：`diffusion_engine/core/` 的 `memory_manager.py` 和 `text_conditioning.py` 依赖 torch，本目录的实验脚本实现了**纯 numpy 版本的相同接口**，以保持零 torch 依赖。两者接口兼容，但实现和依赖链独立。

---

## 10. 参考

- **计划详情**：`.omo/plans/modern-diffusion-inference-roadmap.md` T16–T17 章节
- **引擎模块**：
  - `diffusion_engine/core/scheduler.py` — EulerScheduler + RectifiedFlowScheduler（T10，纯 numpy）
  - `diffusion_engine/core/text_conditioning.py` — TextConditioner Protocol + ToyTextConditioner（T12，torch）
  - `diffusion_engine/core/memory_manager.py` — LatentBufferManager + MemoryStats（T12，torch）
- **学习笔记**：
  - `learning/notes/06_cfg和negative_prompt.md`
  - `learning/notes/07_text_encoder和prompt_embedding_cache.md`
  - `learning/notes/08_latent_buffer和显存预算.md`
- **论文卡片**（与 scheduler benchmark 相关）：
  - `learning/papers/10_consistency_distillation_and_fast_sampling.md` — T16 直接输入
  - `learning/papers/01_scaling_rectified_flow_transformers_sd3.md` — 28 步 RF 基线
