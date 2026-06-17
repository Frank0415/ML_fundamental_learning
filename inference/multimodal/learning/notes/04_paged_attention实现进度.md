# 04 — Paged Attention 实现进度

> **审核日期**: 2026-06-07
> **审核目标**: minivLLM 中与 paged attention 相关的所有代码
> **审核方法**: 静态分析 + 关键词扫描（block_table, paged, slot_mapping, context_lens）
> **计划引用**: 引擎审计计划的 A/B/C/D 四部分检查项

---

## A 部分 — 确定 Paged Attention 的通用实现模式

### A.1 paged attention 是什么？

Paged attention 借鉴了操作系统虚拟内存的页表（page table）思想。它将 KV cache 切分为固定大小的"页"（blocks），每个请求维护一个 block table 来记录它的逻辑 token 位置映射到了哪些物理 blocks。这解决了三个问题：

1. **内存碎片**：序列长度不同时，contiguous buffer 会产生外部碎片。Block 级别分配消除了这个问题。
2. **动态增长**：序列增长时只需分配新 block，不需要移动已有数据。
3. **内存共享**：多个请求可以共享相同的 prefix 的 KV cache blocks（prefix caching）。

### A.2 通用实现的五个核心组件

任何 paged attention 实现都需要以下组件：

| 组件 | vLLM 术语 | minivLLM 对应 |
|------|----------|---------------|
| Block 分配器 | `BlockAllocator` / `CpuGpuBlockAllocator` | 不存在 |
| Block Table | `block_tables` (shape: `[batch_size, max_blocks_per_seq]`) | `Context.block_tables` 字段存在但未使用 |
| Slot Mapping | `slot_mapping` (shape: `[total_tokens]`) | `Context.slot_mapping` 字段存在但未使用 |
| Paged Attention Kernel | `paged_attention_v1/v2` (CUDA/Triton) | 不存在 |
| 调度器集成 | `Scheduler` 调用 `allocate`/`free` | 不存在 |

### A.3 参考 benchmark：vLLM 的 PagedAttention v1/v2

- **v1**: Q[16] × KV[16×head_dim] → 逐个 block 做 matmul，简单但效率一般
- **v2**: 利用 shared memory + warp tile，提高吞吐
- **vLLM v0.6+**: FlashInfer 集成，进一步优化

minivLLM 作为学习引擎，实现一个简单的 v1 风格（逐个 block attention）即可满足教学目的。

### A.4 现状

- 通用模式的理解已建立
- vLLM 的参考架构可作为实现指南
- 无实际的分配器或 kernel 代码

**是否达标**：✅ 通用模式已理解，理论铺垫完成

---

## B 部分 — 审计 minivLLM 的现有 KV/cache/context 脚手架

### B.1 `KVCache` 类 (`core/kv_cache.py`)

- **类型**：contiguous buffer，**不是 paged attention**
- **形状**：`(num_layers, max_seq_len, num_kv_heads, head_dim)`
- **分配方式**：一次性预分配，固定大小
- **Block Tables**：无
- **Block 管理器**：无

**与 paged attention 的差距**：

| 特性 | 当前 KVCache | 需要的 Paged KVCache |
|------|-------------|---------------------|
| 分配单位 | token 级别（位置索引） | block 级别（block_id 索引） |
| 内存布局 | `[layer, pos, kv_head, head_dim]` | `[num_blocks, layer, block_size, kv_head, head_dim]` 或类似 |
| 多序列 | 不支持 | 通过 block table 映射支持 |
| 序列取消/释放 | 需清空位置 | 释放 block，可重用 |
| Prefix 共享 | 不支持 | 支持（多个 block table 可指向相同物理 blocks） |

### B.2 `Context` dataclass (`utils/context.py`)

已定义的 paged-attention 相关字段：

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None   # 变长序列的 Q 维度累积和
    cu_seqlens_k: torch.Tensor | None = None   # 变长序列的 K 维度累积和
    slot_mapping: torch.Tensor | None = None   # token → KV slot 映射
    context_lens: torch.Tensor | None = None   # 每个序列的当前长度
    block_tables: torch.Tensor | None = None   # 序列级 block 映射表
```

**这些字段的作用**（在 vLLM 中）：
- `slot_mapping`：将每个 token 映射到 KV cache 中的物理 slot（可以是 block_id + offset）
- `block_tables`：形状 `(batch_size, max_blocks_per_seq)`，记录每个序列的 block 使用顺序
- `cu_seqlens_q/k`：变长序列的 cumulative sequence lengths，用于 packed attention
- `context_lens`：每个序列的当前 token 数

**现状**：
- 所有字段从未被设置（`set_context` 从未被调用）
- `LMHead.forward` 读取 `context.is_prefill` 但始终为 False
- block_tables 字段存在但无任何代码使用它

### B.3 `Config` 中的 paged attention 字段

```python
kvcache_block_size: int = 16        # block 大小
num_kvcache_blocks: int = -1        # 总 block 数（-1 = 自动计算）
```

这些字段暗示作者计划实现 paged attention，但当前无任何代码使用它们。

### B.4 关键词扫描结果

在全仓库搜索以下关键词：

| 关键词 | 出现文件 | 出现形式 |
|--------|---------|----------|
| `block_table` | `utils/context.py` | 字段定义，未使用 |
| `slot_mapping` | `utils/context.py` | 字段定义，未使用 |
| `paged` | — | **零出现** |
| `page_table` | — | **零出现** |
| `block_size` | `config.py` | 字段定义，未使用 |
| `num_kvcache_blocks` | `config.py` | 字段定义，未使用 |
| `kvcache_block_size` | `config.py` | 字段定义，未使用 |

**现状**：✅ 脚手架字段已定义，但完全未接入实现

---

## C 部分 — 评估从 contiguous 到 paged 的改造工作量

### C.1 需要新建/修改的模块

| 模块 | 动作 | 说明 |
|------|------|------|
| `core/kv_cache.py` | 重写 | 从 contiguous buffer 改为 block-based 分配 |
| `core/block_manager.py` | 新建 | 实现 block 分配/释放/CoW |
| `core/scheduler.py` | 新建 | 管理请求队列 + block 分配 |
| `utils/context.py` | 不改代码，调用 set_context | 接入调度器，正确设置 block_tables 等 |
| `layers/numpy/attention.py` | 新建 `PagedAttn` 或扩展 `Attn` | 支持按 block table 读取 KV |
| `model/qwen3.py` | 扩展 forward | 接受 KV cache + block tables 参数 |
| `config.py` | 无需改 | 已有相关字段 |

### C.2 最小可行实现的工作量估计（学习版）

一个学习用途的 paged attention 实现可以简化为：

1. **Block Manager** (~100 行)：基于 Python list 的 block 分配器，支持 allocate/free/copy
2. **Block Table** (~50 行)：Tensor `[batch_size, max_blocks_per_seq]` 的管理
3. **Paged Attention Matmul** (~50 行)：逐个 block 做 `Q × K_block^T`，不依赖 CUDA kernel
4. **调度器** (~150 行)：简单的 FIFO scheduler，管理 prefill/decode 切换

总计约 350 行 Python/PyTorch。虽然性能远低于 vLLM 的 CUDA 实现，但可以完整展示 paged attention 的工作原理。

### C.3 前提条件

在开始 paged attention 实现之前，需要先修复以下阻塞项：

1. B1：`Qwen3Attn → Attn` 参数不兼容（TypeError）
2. B2：`Qwen3FFN.act_fn = None`（TypeError）
3. 将 KV cache 接入 forward（让模型使用 `KVCache.write`/`KVCache.read`）

这三个前提条件都不涉及 paged attention，但它们是 paged attention 实现的必要前驱。

**现状**：✅ 改造工作量已评估，前提条件已识别

---

## D 部分 — 对比 vLLM 在相同模块的接口差异

### D.1 核心接口对比

| 接口 | vLLM | minivLLM |
|------|------|----------|
| KV cache 类 | `KVCache` (C++/CUDA + Python wrapper) | `KVCache`（纯 Python） |
| Block 分配 | `BlockTable` + `CpuGpuBlockAllocator` | 不存在 |
| Attention kernel | `paged_attention_v1/v2`（CUDA）或 FlashInfer | `Attn.forward`（纯 PyTorch matmul） |
| 上下文传递 | `AttentionMetadata` + `attn_metadata` | `Context` dataclass（未使用） |
| 调度器 | `Scheduler`（含 preemption 和 swap） | 不存在 |

### D.2 关键差异

1. **vLLM 的 AttentionMetadata 是信息丰富的结构体**：包含 `slot_mapping`、`block_tables`、`context_lens`、`query_start_loc`、`seq_start_loc` 等，由调度器在每个 step 填充。minivLLM 的 `Context` 仅有字段定义，无填充逻辑。

2. **vLLM 的 paged_attention kernel 是性能关键**：使用 CUDA 实现，通过 `block_tables` 间接寻址。minivLLM 没有 CUDA kernel，学习版可以用纯 Python 循环替代。

3. **vLLM 的 block manager 有两级**：`CpuGpuBlockAllocator` 管理 GPU 内存，`BlockTable` 管理序列的 block 使用。minivLLM 不需要 GPU/CPU 两级分配（学习用途简化）。

4. **vLLM 有 CoW (Copy-on-Write)**：支持 prefix caching 和 beam search。minivLLM 暂不需要。

### D.3 结论

minivLLM 的 `Context` 和 `Config` 中的 paged attention 字段与 vLLM 的接口设计一致，说明作者在规划时参考了 vLLM。当前这些字段仅为占位符，无实现。后续实现时应继续遵循 vLLM 的接口约定。

**现状**：✅ 接口差异已对比，vLLM 参考方向已确认

---

## 总体总结

| 部分 | 内容 | 现状 | 是否达标 |
|------|------|------|----------|
| A | 通用实现模式 | 理论已理解，参考架构已确认 ✅ | ✅ |
| B | minivLLM 现有脚手架 | 字段存在（block_tables 等），但完全未接入实现 | ⚠️ 脚手架达标，实现未达标 |
| C | 改造工作量 | 已评估，约 350 行可用；前提条件（B1+B2+KV 接线）已识别 | ✅ |
| D | vLLM 接口对比 | 已对比，Context/Config 与 vLLM 一致 | ✅ |

**当前 paged attention 状态：部分实现。** Wave 2 / Task 6 已在 `experiments/paged_attention_fix_or_impl/` 下实现 correctness-first 最小 paged KV 路径：block 分配器、block table、request state、paged KV tensor、prefill/decode/free 生命周期，以及 attention 前 gather 为 contiguous K/V 的对齐 benchmark。`minivLLM/` 内部 engine 未改，scheduler、prefix sharing、CUDA/Triton paged attention kernel 仍未实现。

| 组件 | 状态 |
|------|------|
| BlockManager | 已实现 |
| BlockTable | 已实现 |
| RequestState | 已实现 |
| PagedKVCache | 已实现 |
| prefill/decode/free 生命周期 | 已实现 |
| gather 后复用现有 attention | 已实现 |
| scheduler / continuous batching | 未实现 |
| prefix sharing | 未实现 |
| CUDA/Triton paged kernel | 未实现 |

**下一优先事项**：在保持 logits 对齐的前提下，评估是否把外部 paged KV adapter 接入更高层 scheduler；性能优化留到后续 Wave。
