# vLLM PagedAttention 源码参考

> 本文档结构化列出 vLLM PagedAttention 的关键类、API 和实现启发，用于理解 minivLLM 已实现的 correctness-first PagedKV 路径（Task 6）的设计来源。

## 1. vLLM 核心组件概览

vLLM 的 PagedAttention 实现分布在几个核心模块中，本文档主要参考 vLLM v0.6.x 的代码结构。vLLM 使用 CUDA kernel 实现高性能 block 级别 attention，但本节聚焦于**设计层面**的概念映射，不涉及 CUDA kernel 细节。

### 1.1 架构层次

```
vLLM Serving Layer
    ├── Scheduler          # 请求调度，prefill/decode 决策
    ├── BlockSpaceManager  # 物理 block 分配/释放/管理
    │   ├── BlockTable      # 每个序列的逻辑→物理映射表
    │   └── CacheEngine     # 实际 GPU tensor 管理
    ├── ModelRunner        # 模型前向推理入口
    │   └── AttentionMetadata  # 拼接 gather 后的 K/V 和 slot_mapping
    └── Memory Profiler    # 决定 total_blocks 数量
```

minivLLM 的 Task 6 实现采用了类似的层次结构，但做了 correctness-first 的简化：`BlockManager` 管理物理 block 池（Python list），`BlockTable` 维护每个请求的逻辑映射，`PagedKVCache` 持有 (num_blocks, num_layers, block_size, num_kv_heads, head_dim) 的 Tensor。

## 2. 关键类与 API

### 2.1 Block Manager 与 Block Table

**vLLM 的设计理念**：BlockTable 是每个序列持有的映射表，BlockManager 是全局的物理 block 池管理器。物理 block 的大小固定（通常 16），分配/释放都是 block 级别。

**vLLM 中的关键类和 API**：

- **`BlockTable`**：逻辑到物理的映射。本质上是一个可变长度的 list，`block_table[i]` 返回第 i 个逻辑 block 对应的物理 block ID。
- **`BlockSpaceManager.allocate(seq_group)`**：为一个请求分配 block。
- **`BlockSpaceManager.free(seq)`**：释放请求占用的所有 block。
- **`BlockSpaceManager.append_slot(seq)`**：decode 阶段追加一个 token 的 slot。
- **`BlockSpaceManager.can_allocate(seq_group)`**：检查是否有足够的 free block 来分配。

**minivLLM 的对应实现（Task 6）**：

- `BlockManager` 维护 `free_blocks: List[int]` 和 `allocated_blocks: Dict[int, str]`（物理 block ID → 所属 request ID）。
- `prefill_allocate(request_id, num_tokens)` → 分配 `ceil(num_tokens/block_size)` 个 block，返回 `(block_table, num_blocks_allocated)`。
- `decode_append(request_id, block_table)` → 若最后一个 block 未满则 free slot+1；满了则分配新 block。
- `free_request(request_id)` → 归还所有 block 到 free 池。

### 2.2 KV Cache 物理布局

vLLM 中 KV cache 的物理布局是 `[num_blocks, num_layers, block_size, num_kv_heads, head_dim]` 或 `[num_blocks, 2, num_layers, block_size, num_kv_heads, head_dim]`（K 和 V 分开）。写 K/V 时按 `[block_id, layer_idx, offset_inside_block]` 的索引写入。

**关键接口**：

- **`CacheEngine.gpu_cache`**：GPU 上的物理 block tensor，形状如上。
- **`slot_mapping`**：一个 `int64` tensor，映射每个 token 在物理 block 中的写入位置。`-1` 表示 padding。`slot_mapping = block_table[block_idx] * block_size + offset`。

minivLLM 的 `PagedKVCache` 类似：形状 `(num_blocks, num_layers, block_size, num_kv_heads, head_dim)` 分别持有 `k_cache` 和 `v_cache`。写操作通过 `block_id` 和 `offset` 索引。

### 2.3 KV Gather 与 Attention

vLLM 在 attention 计算前需要将分散在多个 block 的 K/V 拼接为 contiguous tensor。vLLM 使用 CUDA custom kernel (`paged_attention_v1/v2`) 直接在 block 索引上做 attention，避免显式 gather。

minivLLM 使用 correctness-first 的纯 PyTorch 方法：`gather_kv_for_attention()` 先按 `block_table` 索引取出所有 block 的 K/V，用 `torch.cat` 拼成 `[seq_len, num_kv_heads, head_dim]` 的 contiguous tensor，然后走标准 MHA 计算。代价是每次 decode 都要 gather 一次，但正确性优先。

### 2.4 调度器集成

vLLM 的 Scheduler 围绕 PagedAttention 设计：

- **Prefill batch**：调度器拿到一批新请求，先调用 `can_allocate(num_tokens)` 检查有足够 free block，然后 `allocate`，再走 prefill（一次计算所有 prompt token K/V 并写入 block）。
- **Decode batch**：调度器对已有请求调用 `append_slot`，并限制 batch 中的 decode 请求数（以控制显存碎片和延迟）。
- **抢占（Preemption）**：如果 free block 不足，调度器可选择 swap（GPU→CPU 迁移 KV）或 recompute（释放 block 并重新 prefill）。

minivLLM 当前无 scheduler（Task 6 仅做 correctness-first），这些调度器集成是后续 Wave 的扩展方向。

## 3. 碎片率与显存效率

vLLM 的 PagedAttention 将 KV cache 内部碎片率降至 <4%，这是其核心优势。碎片只发生在每个请求最后一个未写满的 block。计算方式：

```
fragmentation_ratio = (total_allocated_blocks * block_size - total_used_tokens) /
                       (total_allocated_blocks * block_size)
```

受限显存场景下，若 block_size=16、每个请求最后浪费平均 8 个 slot（浪费率 50% → 但只在一个 block），且总共 128 个请求，浪费 = 128 × 0.5 × 16 × kv_per_token ≈ 128 × 0.5 × 16 × 72KB ≈ 74MB。占 中等显存配置 约 0.6%，可忽略。

## 4. 对本项目的启发

vLLM 的 PagedAttention 设计直接影响了 minivLLM Task 6 的实现。主要启发：

1. **Block table 间接寻址是核心**：从 contiguous 到 paged 的本质变化是引入了 block table 这一层间接映射。理解这一层的 overhead 和正确性保证，是多模态 KV cache 管理的前提（Task 11）。
2. **correctness-first 的正确性保证**：vLLM 也是先 correctness（v1 kernel）再优化（v2 kernel, FlashInfer integration）。我们的 correctness-first 路径正是沿用这个思路。
3. **多模态需要扩展 block 分配逻辑**：vLLM 原生的 block 分配不考虑 visual token 的特殊性。在多模态场景中，visual token 的 prompt（prefill）是一次性写入的，不需要后续 decode 追加 block。但 visual token 和 text token 的混合布局增加了 block table 的管理复杂度——这个扩展正是 Task 11 的核心内容。
