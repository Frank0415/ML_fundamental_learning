# 11 — vLLM PagedAttention: 高效 LLM 推理的虚拟内存管理

## 一句话总结

vLLM 提出 PagedAttention，将操作系统虚拟内存的页表（page table）思想引入 LLM 推理的 KV cache 管理，用固定大小的物理 block + 逻辑 block table 的间接寻址替代连续内存分配，近乎消灭 KV cache 的内存碎片（碎片率降至 <4%），并天然支持 prefix caching。

## 关键 Idea

### 1. 问题：KV Cache 内存碎片

传统 LLM 推理引擎为每个请求预分配 `max_seq_len` 的连续 KV cache 空间。问题在于：
- 大多数请求的实际生成长度远小于 `max_seq_len`，预留空间大量浪费（内部碎片）。
- 不同请求生命周期不同，先完成的请求释放其连续空间后，剩余空间可能因大小不匹配而无法给新请求使用（外部碎片）。
- 碎片导致显存利用率仅 20%~40%，大量显存被浪费。

### 2. PagedAttention 的核心设计

PagedAttention 将 KV cache 切为固定大小的 **block**（如每 block 存 16 个 token 的 K/V）。每个 block 是一个物理显存单元，包含所有 attention 层的 K 和 V（即 `[num_blocks, num_layers, block_size, num_kv_heads, head_dim]` 的布局）。请求不直接持有物理 block，而是持有一个 **block table**（逻辑索引 → 物理 block ID 的映射表）。

关键操作：

- **prefill_allocate(num_tokens)** → 分配 `ceil(num_tokens/block_size)` 个物理 block，写入 block table。
- **decode_append(new_token)** → 如果当前最后一个 block 已满，分配新 block；否则写入当前 block 的下一个空闲 slot。
- **free_request()** → 将该请求使用的所有物理 block 标记为 free，可被后续请求复用。
- **gather_kv_for_attention()** → 根据 block table 将各 block 的 K/V 拼成 contiguous tensor，然后做标准的 MHA 计算。

### 3. Block Manager 与碎片率

Block Manager 维护一个物理 block 池（free list）。分配时从 free 池取 block，释放时归还。因为所有 block 大小相同，不存在外部碎片。**内部碎片**仅在每个请求的最后一个 block 未写满时发生，碎片率通常 <4%（远低于 contiguous 的 60%~80% 浪费）。

`fragmentation_ratio = wasted_slots / total_allocated_slots`，其中 `wasted_slots = sum(block_size - last_block_used for each request)`。

### 4. Prefix Caching（前缀缓存）

多个请求通常共享相同的 prompt 前缀（如 system prompt）。PagedAttention 允许不同请求的 block table 指向相同的物理 block 来共享 prefix：

- 请求 A 的 prefix 写入 block [3, 7]。
- 请求 B 有相同 prefix，其 block table 也指向 [3, 7]，不分配新 block。
- 请求 B 的 decode 阶段分配新 block [9]，与共享的 [3, 7] 不冲突。

这通过 **hash-based prefix cache** 实现：每个 block 的内容（所有层的 K/V 拼接）计算 hash，存入 hash 表。新请求到达时，逐 block 检查 hash 是否命中，命中即复用。

### 5. Scheduling 集成

PagedAttention 与调度器紧密集成：

- 调度器在 prefill 之前调用 `can_allocate(num_tokens)` 检查是否有足够 free block。
- 若 free block 不足，调度器可以选择：等待（queue）或抢占（preempt）低优先级请求。
- 抢占时，被抢占请求的 block 被 swap 到 CPU 内存（GPU→CPU copy），后续恢复时 swap 回 GPU。

## 与本项目的关联

minivLLM 已经实现了 correctness-first 的 PagedKV 路径（Task 6）：`BlockManager`、`BlockTable`、`PagedKVCache`、`prefill_allocate/decode_append/free_request` 和 `gather_kv_for_attention` 均已工作。vLLM PagedAttention 论文是这整套实现的理论来源。接下来多模态 KV cache 管理（Task 11）需要在 paged KV 基础上加入 visual token 的块分配策略，论文中的 prefix caching 思路对 visual token 的共享也有借鉴价值。
