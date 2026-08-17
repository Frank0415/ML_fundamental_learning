# 旧引擎清点与复用审计报告

> **审计日期**：2026-06-07
> **审计对象**：`minivLLM/`（LLM 推理引擎，目标模型 Qwen3-0.6B）
> **审计目的**：为 `diffusion/` 项目评估旧引擎模块的复用价值，给出 A/B/C 结论
> **审计方法**：逐文件读取源码 + ast_grep 模式搜索 + 模块功能分类

---

## 1. diffusion/ 当前文件夹结构

```
diffusion/
├── .omo/                          # 计划与任务追踪
│   ├── plans/
│   │   └── modern-diffusion-inference-roadmap.md
│   ├── notepads/
│   │   └── modern-diffusion-inference-roadmap/
│   ├── evidence/
│   │   ├── task-1-env.txt
│   │   └── task-1-hf-check.txt
│   ├── drafts/
│   └── boulder.json
├── minivLLM/                      # ★ 旧引擎所在目录（详见第 2 节）
│   ├── .git/
│   ├── .gitignore
│   ├── .python-version            # 3.13
│   ├── pyproject.toml             # 依赖声明
│   ├── uv.lock
│   ├── README.md                  # 空文件
│   ├── LICENSE
│   ├── validate_model.py          # HF 对照脚本（310 行）
│   ├── scripts/
│   │   └── download_qwen.py       # 模型权重下载
│   ├── docs/
│   │   └── qwen3_attn.md          # Qwen3 attention 维度变换文档（230 行）
│   └── minivllm/                  # ★ 核心源码包
│       ├── config.py              # 推理配置 dataclass（41 行）
│       ├── core/
│       │   └── kv_cache.py        # 连续 KV 缓存（86 行）
│       ├── layers/
│       │   └── numpy/
│       │       ├── activation.py  # SiluAndMul（11 行）
│       │       ├── attention.py   # GQA+causal attention（71 行）
│       │       ├── embedding.py   # Embedding/VocabHead/LMHead（32 行）
│       │       ├── linear.py      # Linear 封装（20 行）
│       │       ├── norm.py        # RMSNorm（17 行）
│       │       └── rope.py        # RotaryEmbedding（57 行）
│       ├── model/
│       │   └── qwen3.py           # Qwen3 模型主实现（185 行）
│       └── utils/
│           └── context.py         # Context dataclass（27 行）
└── reports/                       # 本报告所在目录
    └── engine_inventory.md
```

**文件统计**：
- 总计 47 个文件（含 .git 内部文件）
- 实际源码文件 15 个（py + toml + md）
- Python 源码总行数：约 900 行（不含 docs/ 和 .git）
- 最长的单文件：`validate_model.py`（310 行）
- 最短的源码模块：`activation.py`（11 行）

---

## 2. 旧引擎所在目录

**实际目录名：`minivLLM/`**（非 `engine/`）。

该目录是一个**独立 Git 仓库**（有自己的 `.git/`），与 `diffusion/` 上层目录（计划中称为 `diffusion/` 的 workspace）存在嵌套关系。`minivLLM/` 不是 `diffusion/` 的子模块，而是被整体拷贝或克隆到 workspace 中的独立项目。

关键元数据：
- Python 版本要求：`>=3.13`（`.python-version` 文件确认为 `3.13`）
- 项目名：`minivllm`（`pyproject.toml` → `[project] name`）
- 定位：小型 LLM 推理引擎，专为 Qwen3-0.6B 设计

---

## 3. 语言与技术栈

| 维度 | 详情 | 证据 |
|------|------|------|
| **主语言** | Python 3.13+ | `.python-version` = `3.13`；`pyproject.toml`：`requires-python = ">=3.13"` |
| **深度学习框架** | PyTorch 2.11+ | `pyproject.toml`：`torch>=2.11.0` |
| **模型加载** | Transformers 5.8+ | `pyproject.toml`：`transformers>=5.8.0`；`config.py` 使用 `AutoConfig.from_pretrained()` |
| **权重下载** | huggingface_hub 1.14+ | `pyproject.toml`：`huggingface-hub>=1.14.0` |
| **代理支持** | socksio 1.0+ | `pyproject.toml`：`socksio>=1.0.0`（用于 SOCKS5 代理下载） |
| **包管理** | uv | 存在 `uv.lock` |
| **C++ / CUDA** | **无** | 全文搜索无 `.cu` / `.cpp` / `.cuh` 文件；无 `CUDAExtension`、`load_inline`、`torch.cuda.amp` 自定义 kernel |
| **Triton** | **无** | 全文搜索无 `import triton` 或 `@triton.jit` |
| **Diffusion 模块** | **无** | 全文搜索无 `diffusion` / `denoise` / `timestep` / `DDPM` / `DDIM` / `DiT` / `MMDiT` |
| **Paged Attention** | **无真实实现** | `context.py` 定义了 `block_tables`、`slot_mapping` 字段但无任何使用这些字段的 attention kernel |

---

## 4. 模块逐项盘点

以下对计划中要求的 14 个关键模块进行逐项检查。判定标准：
- **Yes**：模块已实现且功能完整
- **Partial**：模块定义了接口/占位但未完成或未连接
- **No**：模块完全不存在

### 4.1 Tensor Abstraction（张量抽象层）

**结论：No**

`minivLLM/` 直接使用 `torch.Tensor`，无任何自定义张量包装类。全库无 `TensorWrapper`、`TensorMeta`、`DTensor` 等抽象。Linear/RMSNorm/Attention 等模块的 `forward()` 直接接收和返回 `torch.Tensor`。

对 diffusion 的影响：如果新引擎需要支持多精度/量化/分布式张量抽象，minivLLM 无法提供任何基础。

### 4.2 Model Loader（模型加载器）

**结论：Partial**

- `config.py` 的 `Config.__post_init__()` 调用 `AutoConfig.from_pretrained(self.model)` 加载 HF 配置 → 依赖 Transformers 库
- `validate_model.py` 的 `run_compare()` 调用 `AutoModelForCausalLM.from_pretrained()` 加载完整 HF 模型 → 依赖 Transformers 库
- **无自定义权重加载器**：`validate_model.py` 的 `_load_hf_weights_into_mini()` 是将 HF 模型的 state_dict 手动映射到 mini 模型的命名空间，但这不是一个通用的 loader

对 diffusion 的影响：minivLLM 没有提供独立的权重 I/O 模块。扩散模型（DiT/MMDiT）的权重加载需要从头实现。

### 4.3 Transformer Block（Transformer 块）

**结论：Yes**

文件：`minivllm/model/qwen3.py` 第 95-127 行

```python
class Qwen3DecoderLayer(nn.Module):
    # Pre-Norm 结构：
    #   input → input_layernorm → attn → post_layernorm → mlp
    # 带 residual stream 的显式传递
```

实现特点：
- **Pre-Norm decoder** 架构，适合 LLM 自回归推理
- residual 作为显式参数传入传出（而非 hidden_states += 的隐式写法）
- 包含 attention（Qwen3Attn）和 FFN（Qwen3FFN）两个子模块
- 使用 RMSNorm 作为归一化层
- 依赖 GQA + RoPE + causal mask 的 attention

对 diffusion 的影响：**可参考其 Pre-Norm 结构和 residual 管理方式**，但扩散 transformer（DiT/MMDiT）需要 AdaLN-Zero 或 cross-attention conditioning，当前 block 的逻辑需要大幅改造。

### 4.4 Attention（注意力机制）

**结论：Yes（但仅限 GQA + causal，不可直接用于 DiT）**

文件：`minivllm/layers/numpy/attention.py`（71 行）+ `minivllm/model/qwen3.py` 的 `Qwen3Attn`（62 行）

关键实现：
- **Grouped Query Attention (GQA)**：`repeat_kv()` 将 KV heads 扩展到 Q heads 数
- **Causal Mask**：`causal_mask()` 生成上三角 `-inf` mask，用于自回归解码
- **QK-Norm**：无 bias 时对 Q/K 分别做 RMSNorm（`qwen3.py` 第 57-59 行）
- **手动 softmax attention**：`attention.py` 第 64-69 行，纯 PyTorch 实现，无 FlashAttention

**为什么不能直接套到 DiT/MMDiT：**

| 特性 | minivLLM Attention | DiT/MMDiT Attention |
|------|-------------------|---------------------|
| Mask 类型 | Causal（上三角 mask） | 无 causal mask（全注意力） |
| KV 头数 | GQA（num_kv_heads < num_heads） | 通常无 GQA（或 Joint Attention） |
| 位置编码 | RoPE（旋转位置编码） | 无位置编码（或 timestep 嵌入） |
| 序列操作 | 预填充（prefill）/逐 token 解码 | 全序列并行去噪 |
| 分块/调度 | 依赖 Context 的 cu_seqlens 进行可变长批处理 | 扩散 latent 是固定大小的 batch |

`attention.py` 中的 `Attn` 类（第 33-70 行）可作为**纯 PyTorch attention 的封装参考**（Q×K 点积除以 sqrt(dk)、softmax、matmul V），但这是 PyTorch 基础操作，不构成独特复用价值。

### 4.5 MLP / FFN（前馈网络）

**结论：Partial（SiluAndMul 已定义但未使用，FFN act_fn = None）**

文件：
- `minivllm/layers/numpy/activation.py`：定义了 `SiluAndMul`（11 行），实现 `silu(x) * y` 的 gated activation
- `minivllm/model/qwen3.py` 的 `Qwen3FFN`（77-93 行）：

```python
class Qwen3FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        self.gate_up = Linear(hidden_size, intermediate_size*2)
        self.gate_down = Linear(intermediate_size, hidden_size)
        self.act_fn = None  # TODO: SilU and MUL gate   ← ★ 未连接！

    def forward(self, x):
        gate_up = self.gate_up(x)
        x = self.act_fn(gate_up)  # ← 调用 None，会报 TypeError
        x = self.gate_down(x)
        return x
```

**事实**：`SiluAndMul` 类已在 `activation.py` 正确定义，但 `Qwen3FFN` 的 `self.act_fn` 被硬编码为 `None`，导致前向传播会在 `self.act_fn(gate_up)` 处抛出 `TypeError: 'NoneType' object is not callable`。这是一个**已知未修复的 Bug**，说明 `Qwen3FFN` 从未被正确端到端测试过（`validate_model.py` 的 HF 对比测试走的是不同路径，它通过 `_load_hf_weights_into_mini` 加载权重后做 hidden states 对比，但 forward 中 FFN 的 act_fn 调用在对比路径中可能被跳过或以其他方式绕过了）。

**对 diffusion 的影响**：
- `SiluAndMul` 类本身**可直接复用**：扩散模型（如 DiT）的 FFN 也使用 SwiGLU / Gated SiLU 激活，实现与 `SiluAndMul` 一致
- `Qwen3FFN` 不可直接复用，需要修复并适配扩散的 FFN 结构
- `gate_up` + `gate_down` 的 Linear 组合模式可参考

### 4.6 Norm（归一化层）

**结论：Yes**

文件：`minivllm/layers/numpy/norm.py`（17 行）

```python
class RMSNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is not None:
            total = hidden_states + residual
            new_hidden = torch.rms_norm(total, ...)
            return new_hidden, total    # ← 返回新 hidden 和新 residual
        else:
            return torch.rms_norm(hidden_states, ...)
```

实现特点：
- 调用 `torch.rms_norm` 而非手动实现（PyTorch 2.x 内置算子）
- **支持 residual 流**：接受可选的 `residual` 参数，将 Norm 和 residual add 合并为一步
- 仅 17 行，非常紧凑

对 diffusion 的影响：
- **RMSNorm 的 residual 融合模式可参考**，但扩散模型更需要 **AdaLN（Adaptive Layer Normalization）** 或 **AdaLN-Zero**，它们根据 timestep embedding 动态生成 scale/shift 参数
- 扩散主干可能使用 GroupNorm 而非 RMSNorm
- 综上：**仅参考，不可直接复用**

### 4.7 RoPE（旋转位置编码）

**结论：Yes**

文件：`minivllm/layers/numpy/rope.py`（57 行）

`RotaryEmbedding` 类：
- 预计算 cos/sin 缓存（`cos_sin_cache`），存储在 `register_buffer` 中
- 使用 `@torch.compile` 加速前向
- `apply_rotary_emb()` 实现标准 RoPE 旋转：`x1*cos - x2*sin`，`x2*cos + x1*sin`
- `get_rope()` 使用 `@lru_cache(1)` 单例化

对 diffusion 的影响：
- **不可直接复用**：扩散模型（DiT/MMDiT）不使用 RoPE。扩散 transformer 处理的是 latent 图像 patches，位置信息通过 2D sinusoidal position embedding 或可学习 position embedding 注入，而非 RoPE
- 可保留作为**参考实现**，未来若需要将 RoPE 用于非图像扩散场景（如视频扩散或文本条件编码），可参考此实现

### 4.8 Timestep Embedding（时间步嵌入）

**结论：No**

全文搜索无 `timestep`、`time_embed`、`sinusoidal_embedding`、`TimestepEmbedder`、`t_embed`。扩散模型的核心组件（将噪声时间步 t 映射为特征向量）在 minivLLM 中完全不存在。

这是旧引擎与扩散引擎最根本的不兼容点之一。扩散引擎的 timestep embedding 需要从零实现。

### 4.9 Scheduler / Batch Scheduler（调度器 / 批调度器）

**结论：No**

- `config.py` 有 `max_num_seqs=256` 和 `max_num_batched_tokens=8192` 两个配置字段，但没有任何调度逻辑使用它们
- `context.py` 的 `Context` 有 `is_prefill` 字段用于区分预填充/解码阶段，但无调度器代码
- 无 continuous batching、无 prefix caching、无请求队列管理

对 diffusion 的影响：扩散引擎的"调度"概念完全不同（噪声调度器 noise scheduler，如 DDPM/DDIM/Flow Matching 的 alpha/beta schedule），与 LLM 的请求调度无任何交集。

### 4.10 Sampling Loop（采样循环）

**结论：Partial**

文件：`validate_model.py` 第 185-232 行（`_run_generation_comparison()`）

```python
for _ in range(8):
    hidden = mini_model(input_ids, positions)
    last_logits = mini_model.compute_logits(hidden[-1:])
    next_token = last_logits.argmax(dim=-1).item()  # greedy
    generated.append(next_token)
    input_ids = torch.tensor(generated, device=device)
    positions = torch.arange(len(generated), device=device)
```

实现特点：
- 手写 greedy decoding（`argmax`），**无 temperature/top-k/top-p 采样器**
- 每次迭代重新构建完整的 `input_ids` 和 `positions`（无增量 KV cache 填充）
- 仅用于 `validate_model.py` 的对比测试，不是独立的 sampler 模块

对 diffusion 的影响：
- 扩散采样循环（denoising loop）与 LLM 自回归采样循环完全不同：扩散是从纯噪声开始，逐步去噪 T→0 步，每步更新所有 latent，而非逐 token 追加
- 无复用价值

### 4.11 Memory Manager（显存管理器）

**结论：No**

- `config.py` 有 `gpu_memory_utilization: float = 0.9` 字段，但无任何代码使用它来分配/管理 GPU 显存
- 无 block allocator、无内存池、无 prefix cache 的显存预算计算
- `KVCache` 类使用固定大小的预分配 buffer（`torch.empty(shape)`），不做动态分配

对 diffusion 的影响：扩散引擎也需要显存管理，但管理对象不同（activation checkpointing、latent buffer、模型权重 offloading），实现方式也不同。minivLLM 无相关代码可复用。

### 4.12 KV Cache

**结论：Yes**

文件：`minivllm/core/kv_cache.py`（86 行）

`KVCache` 类：
- **连续预分配**：在 `__init__` 中分配 `(num_layers, max_seq_len, num_kv_heads, head_dim)` 的完整 buffer
- `write(layer_idx, positions, k, v)`：按绝对位置写入 K/V
- `read(layer_idx, end_pos)`：读取 [0, end_pos) 范围的 K/V
- `reset()`：清零缓存
- 无 paged attention 的分块逻辑（无 block table、无 slot mapping 的使用）

**为什么不能直接套到扩散引擎：**

扩散模型（DiT/MMDiT）的去噪过程每步产生**全新的 latent**，不存在"过去的 token 不变"这一前提：
- LLM：token 序列一次生成后永不改变 → KV cache 可跨 step 复用
- Diffusion：每步去噪后 latent 全部更新 → 上一步的 K/V 对下一步无意义

因此，扩散引擎**不需要 KV cache**。`KVCache` 类对扩散项目无复用价值。如未来扩散项目中引入类 AR 组件（如 MaskGIT 或逐 patch 自回归解码），可重新评估。

### 4.13 Paged Attention

**结论：No（仅有占位字段，无真实实现）**

文件：`minivllm/utils/context.py`

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None   # 可变长序列的累积长度
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None    # ← paged attention 相关
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None    # ← paged attention 相关
```

`block_tables` 和 `slot_mapping` 是 vLLM 风格 paged attention 的标准字段，但：
- 全文搜索 `block_tables`、`slot_mapping` 仅在 `context.py` 的定义中出现
- 无任何 attention kernel 使用这些字段做分块 KV 访问
- 无 block manager、无物理/逻辑块映射逻辑
- `KVCache` 类使用连续预分配，不涉及分块

**结论：minivLLM 的 Context 类抄袭了 vLLM 的接口设计，但没有实现背后的 paged attention 逻辑。这些字段是纯占位。**

---

## 5. 与 Diffusion 的复用价值逐项判断

以下对每个关键组件给出复用结论，回答三个问题：(a) 组件在旧引擎中的状态，(b) diffusion 是否需要类似组件，(c) 复用方式。

### 5.1 Linear 封装

- **状态**：`minivllm/layers/numpy/linear.py`（20 行），`nn.Linear` 的薄封装
- **扩散需要吗**：是，所有 transformer 层都需要 Linear
- **复用判断**：**可参考，但直接复用价值低**。PyTorch 的 `nn.Linear` 已经足够好用，20 行的 wrapper 增加的 bias 管理逻辑不值得为此引入依赖。

### 5.2 RMSNorm

- **状态**：`minivllm/layers/numpy/norm.py`（17 行），含 residual 融合
- **扩散需要吗**：部分需要。扩散 transformer 使用 AdaLN/AdaLN-Zero/GroupNorm
- **复用判断**：**仅参考**。RMSNorm 的 residual 融合写法可参考，但扩散需要的是基于 timestep embedding 注入 scale/shift 的 AdaLN，实现完全不同。

### 5.3 RoPE

- **状态**：`minivllm/layers/numpy/rope.py`（57 行），标准实现
- **扩散需要吗**：不需要。DiT/MMDiT 使用 2D sinusoidal position embedding
- **复用判断**：**不可复用**。保留为参考，未来若扩展到视频扩散可能需要。

### 5.4 Attention（GQA + Causal）

- **状态**：`minivllm/layers/numpy/attention.py`（71 行） + `minivllm/model/qwen3.py` 的 `Qwen3Attn`（62 行）
- **扩散需要吗**：需要 attention，但是 **full attention（无 causal mask）** 或 **joint attention（DiT 中的 text+image cross-attention）**
- **复用判断**：**仅可作为 PyTorch attention 封装参考，不可直接套到 DiT/MMDiT**。
  - LLM attention：GQA + causal mask + RoPE，服务于自回归 token 生成
  - DiT attention：full attention（所有 patches 互相 attend）+ AdaLN 调制，服务于一整张图的并行去噪
  - 除 Q·K^T / sqrt(dk) → softmax → matmul V 这一核心公式外，两者架构差异巨大
  - 建议：参考 `Attn` 类的 forward 流程写法，但新引擎的 attention 需要从头实现

### 5.5 MLP / FFN

- **状态**：`Qwen3FFN` 在 `minivllm/model/qwen3.py`（77-93 行），act_fn=None（Bug）
- **扩散需要吗**：是，需要 Gated FFN（SwiGLU 或类似 activation）
- **复用判断**：`SiluAndMul`（`activation.py`）**可直接复用**。这是本审计中唯一可直接复用的代码：`silu(x) * y` 的实现与扩散 FFN 的 gated activation 完全一致。但需要修复 `Qwen3FFN` 将其正确连接。

### 5.6 Transformer Block（Qwen3DecoderLayer）

- **状态**：`minivllm/model/qwen3.py`（95-127 行），Pre-Norm decoder
- **扩散需要吗**：是，但需要 AdaLN-Zero 或等效的 conditioning 机制
- **复用判断**：**可参考 Pre-Norm 和 residual 传递结构，但要做大幅改造**：
  - 需要注入 timestep embedding 到 norm 层（AdaLN）
  - 需要移除 causal mask
  - 需要替换 RoPE 为 2D position embedding
  - 需要添加 cross-attention（MMDiT 的 text-image 联合注意力）

### 5.7 KV Cache

- **状态**：`minivllm/core/kv_cache.py`（86 行），连续预分配
- **扩散需要吗**：**不需要**。这是最明确的不可复用项。
- **复用判断**：**不可复用**。原因已在 4.12 节详述：diffusion 每步 denoising 后 latent 全部更新，无跨 step KV 复用场景。此文件可完全忽略。

### 5.8 Paged Attention

- **状态**：`context.py` 有占位字段，无实现
- **扩散需要吗**：不需要
- **复用判断**：**不可复用**。无实现，且扩散无需 KV cache 管理。

### 5.9 Timestep Embedding / Scheduler / Sampling Loop

- **状态**：全部不存在
- **扩散需要吗**：是，这三个是扩散引擎的核心
- **复用判断**：**不可复用**。需要从零实现。

### 5.10 Tokenizer / Embedding / LMHead

- **状态**：`embedding.py` 定义了 `VocabHead`（input embedding）和 `LMHead`（output projection）
- **扩散需要吗**：可能需要 text encoder（CLIP/T5），但不是 LLM token embedding
- **复用判断**：**不可复用**。扩散模型的 text embedding 来自预训练的 text encoder（如 CLIP text model），不是从零训练的 token embedding + LM head。

### 5.11 Config / Context

- **状态**：`config.py`（41 行）和 `context.py`（27 行），面向 LLM 推理
- **扩散需要吗**：部分。需要 diffusion 专用配置
- **复用判断**：**仅参考**。Config 的 dataclass 写法可参考，但字段完全不同（扩散需要 `num_timesteps`、`noise_schedule`、`patch_size` 等）。Context 的 `cu_seqlens` / `is_prefill` 等字段与扩散无关。

---

## 6. 结论：A / B / C 决策

### 候选方案

| 方案 | 描述 | 含义 |
|------|------|------|
| **A** | 大幅复用现有引擎 | 超过 50% 的模块可直接使用或仅需小改 |
| **B** | 部分复用，新旧混合 | 核心 transformer 组件可复用，diffusion 专用模块新写 |
| **C** | 完全不适合，重写新引擎 | 旧引擎模块不可直接复用，新建 `diffusion_engine/` |

### 最终结论：**C（完全不适合，新建 `diffusion_engine/`）**，次选 **B**

#### 支持 C 的核心论据：

1. **minivLLM 是 LLM 推理引擎，不是通用 transformer 库**。其架构设计和所有模块都是围绕自回归语言模型推理构建的：causal mask、RoPE、prefill/decode 两阶段、KV cache、LM head、token embedding。这些与扩散模型的 denoising loop 无交集。

2. **唯一可直接复用的组件是 `SiluAndMul`**（11 行代码），不值得为它引入整个 minivLLM 依赖。

3. **核心组件的架构差异不可调和**：
   - Attention：GQA + causal → DiT 需要 full attention + AdaLN modulation
   - Norm：RMSNorm → DiT 需要 AdaLN-Zero / GroupNorm
   - Position：RoPE → DiT 需要 2D sinusoidal
   - Sampling：自回归 greedy → DiT 需要迭代去噪 loop
   - Memory：KV cache → DiT 不需要（K/V 每步刷新）

4. **旧引擎本身不完整**：
   - `Qwen3FFN.act_fn = None`（Bug，从未跑通完整的 FFN forward）
   - 无 paged attention 实现（仅有占位字段）
   - 无采样器（仅 greedy argmax）
   - 总代码量约 900 行，体量小

5. **diffusion 引擎需要的核心能力，minivLLM 全部缺失**：
   - Timestep embedding
   - Noise scheduler（DDPM/DDIM/Flow Matching）
   - AdaLN / AdaLN-Zero
   - 2D patch embedding
   - Cross-attention（text-to-image）
   - Denoising sampling loop
   - VAE decoder integration

#### 为什么 B 也可以是合理选择：

如果未来计划中"复用"的定义放宽为"代码风格和 PyTorch 范式参考"，则 B 是合理选择：

- `Linear` 封装风格（20 行薄 wrapper 模式）可在 `diffusion_engine/layers/linear.py` 中延续
- `RMSNorm` 的 residual 融合写法可迁移到 AdaLN 设计
- `RotaryEmbedding` 的 cos/sin 缓存 + `@torch.compile` 模式可参考（虽然内容不同）
- `Attn` 类的手写 softmax attention 流程可作为 `diffusion_engine/layers/attention.py` 的起点模板
- `Qwen3DecoderLayer` 的 Pre-Norm + 显式 residual 传递可作为 `DiTBlock` 的结构原型
- `KVCache` 的 `register_buffer` + `write/read/reset` API 风格可参考（如果扩散未来引入类 KV 缓存）

但即便如此，**新引擎的核心模块（attention、norm、transformer block）都需要从头实现**，旧代码仅提供风格参考而非逻辑复用。

### 建议

**以 C 为主方案，启动 `diffusion_engine/` 的新建**，同时在以下方面以 B 为补充：
- 将 `minivLLM/minivllm/layers/numpy/activation.py` 的 `SiluAndMul` 直接复制或引用到新引擎的 `diffusion_engine/layers/activation.py`
- 读旧代码作为 PyTorch 编码规范和模块组织方式的参考
- 对于 RMSNorm → AdaLN 的迁移，可参考 RMSNorm 的 residual 融合模式

**此报告可被后续 T3（复用决策说明）和 T4（README）引用。**

---

## 附录 A：逐文件判断摘要

| 文件 | 行数 | 扩散相关度 | 判断 |
|------|------|-----------|------|
| `minivllm/model/qwen3.py` | 185 | 低（LLM decoder） | 仅参考结构 |
| `minivllm/layers/numpy/attention.py` | 71 | 中（core attn 公式通用） | 仅参考（GQA+causal 不适用） |
| `minivllm/layers/numpy/norm.py` | 17 | 中（Norm 通用） | 仅参考（需改 AdaLN） |
| `minivllm/layers/numpy/rope.py` | 57 | 低（diffusion 不用 RoPE） | 不可复用 |
| `minivllm/layers/numpy/linear.py` | 20 | 高（Linear 通用） | 可参考风格 |
| `minivllm/layers/numpy/activation.py` | 11 | 高（SwiGLU 通用） | **可直接复用** |
| `minivllm/layers/numpy/embedding.py` | 32 | 低（token embedding） | 不可复用 |
| `minivllm/core/kv_cache.py` | 86 | 极低（diffusion 无需 KV cache） | 不可复用 |
| `minivllm/utils/context.py` | 27 | 低（LLM 推理上下文） | 不可复用 |
| `minivllm/config.py` | 41 | 低（LLM 配置） | 仅参考 dataclass 写法 |
| `validate_model.py` | 310 | 低（LLM 验证） | 不可复用（但可参考 HF 对照测试模式） |
| `scripts/download_qwen.py` | 78 | 低（模型下载） | 不可复用 |
| `docs/qwen3_attn.md` | 230 | 低（文档） | 参考文档 |

## 附录 B：搜索确认的"不存在项"清单

以下模式通过全文搜索（grep）确认在 minivLLM 中不存在：

- `diffusion` / `denoise` / `denoising` / `DDPM` / `DDIM` / `DiT` / `MMDiT`
- `timestep` / `time_embed` / `sinusoidal_embedding`
- `adaln` / `ada_ln` / `adain` / `GroupNorm`
- `triton` / `@triton.jit` / `.cu` / `.cuh` / `.cpp`（C++/CUDA）
- `flash_attn` / `FlashAttention` / `paged_attention`（实际实现）
- `vae` / `autoencoder` / `encoder` / `decoder`（扩散编解码器）
- `classifier_free` / `cfg_scale` / `guidance`
- `patch_embed` / `patchify` / `unpatchify`
