# SGLang RadixAttention 源码参考

> 本文档结构化列出 SGLang RadixAttention 的关键组件、API 和实现启发，用于理解高级 KV cache 前缀共享策略，并作为 minivLLM 多模态 KV cache 管理的参考基线。

## 1. SGLang 核心架构概览

SGLang 是一个高效的 LLM 推理 runtime，其两大核心创新是：（1）RadixAttention——基于 Radix Tree 的前缀缓存；（2）Structured Language——结构化生成语言。本节聚焦 RadixAttention。

### 1.1 SGLang 推理流程

```
请求到达
  │
  ▼
TokenizerManager ──→ tokenize prompt（可并行）
  │
  ▼
Scheduler ──→ 遍历 Radix Tree，查找 prefix 匹配
  │            ├─ 命中部分 → 复用已有 KV block
  │            └─ 未命中部分 → 标记为需要 prefill
  │
  ▼
ModelRunner ──→ prefill（仅未命中部分）+
  │              decode（逐 token 生成）
  │
  ▼
Radix Cache ──→ 将新计算的 KV block 插入 Radix Tree
  │              释放引用计数为 0 的节点
```

## 2. Radix Tree 数据结构

### 2.1 核心设计

Radix Tree 是一种压缩前缀树（Patricia Trie），每条边压缩存储一个 token 子序列。SGLang 的 Radix Tree 节点包含：

- **`key`**：表示从父节点到当前节点的 token 子序列（压缩存储）。
- **`children`**：子节点字典，key 是子节点的第一个 token。
- **`value`**：KV cache block 列表（物理 block ID 列表）。
- **`ref_count`**：引用计数，表示当前有多少活跃请求使用该节点。
- **`parent`**：父节点引用。
- **`last_access_time`**：最近一次访问时间（用于 LRU 驱逐）。

### 2.2 基本操作

**插入（insert）：**
- 新请求的 token 序列从 Root 开始匹配。匹配到无法继续时，在该分叉点创建新节点（或拆分已有边）。
- 新节点的 KV block 列表指向物理 block 池中新分配（或复用的）block。
- 被匹配到的已有节点引用计数 +1。

**查找（match_prefix）：**
- 给定一个新 prompt 的 token 序列，从 Root 出发按 token 逐位匹配。
- 返回匹配到的节点列表 + 未匹配的剩余 token 数。KV cache 复用匹配到的节点对应的 block。

**驱逐（eviction）：**
- 当显存不足时，基于某种策略驱逐节点。通常使用 LRU：选择 `last_access_time` 最早的未引用节点驱逐。
- 驱逐时释放节点对应的物理 block，并从树中删除该节点。如果节点的父节点只有一个子节点，可能触发边的合并。

### 2.3 SGLang 源码中的关键类

SGLang 源码（Python 层）中的核心类映射（基于 SGLang v0.3.x 结构）：

| 类 | 文件路径（大致） | 功能 |
|---|---|---|
| `RadixCache` | `sglang/srt/managers/schedule_batch.py` 或 `mem_cache/radix_cache.py` | Radix Tree 的主类，持有 root 节点和相关统计 |
| `TreeNode` | 同上 | 树节点类，包含 key/children/value/ref_count 等字段 |
| `Scheduler` | `sglang/srt/managers/scheduler.py` | 调度器，在 schedule 前调用 `RadixCache.match_prefix` |
| `ModelRunner` | `sglang/srt/model_executor/model_runner.py` | 根据匹配结果决定 prefill 范围 |

## 3. 显存管理

SGLang 的 KV cache 同样采用 block-based 分配，与 vLLM PagedAttention 的物理布局类似：`[num_blocks, num_layers, block_size, num_kv_heads, head_dim]`。

**与 vLLM 的关键差异**：

1. **Block 生命周期由 Radix Tree 节点管理**：不是每个请求独立持有 block，而是 Radix Tree 节点持有共享 block。节点引用计数 = 使用该节点的活跃请求数。
2. **释放时机**：当请求完成，它引用的所有树节点引用计数 -1。引用计数变为 0 的节点可以被驱逐（释放 block）。
3. **LRU 驱逐**：当空闲 block 不足时，按 LRU 驱逐引用计数为 0 的节点。驱逐一个节点时，其所有子节点也需级联驱逐（或提升到父节点）。

## 4. Prefix 匹配与调度

### 4.1 自动前缀匹配

SGLang 的调度流程中，每个请求到达后的第一步就是 `RadixCache.match_prefix(token_ids)`。匹配结果决定：

- **完全命中**：所有 prompt token 都在 Radix Tree 中。不需要 prefill，直接进入 decode。这是最理想的情况。
- **部分命中**：前 N 个 token 的 KV 可复用（通过匹配到的节点），剩余 token 需要 prefill。这是最常见情况。
- **完全未命中**：Root 节点都不匹配。全部 token 需要 prefill。

### 4.2 Batch 调度优化

SGLang 的调度器利用 Radix Tree 做 batch 优化：

- 将具有相同未匹配 prefix 的请求组成一个 batch，共享 prefill 计算。
- 例如两个请求分别为 "What is Python?" 和 "What is C++?"，共享 prefix "What is " 的 KV（通过 Radix Tree 匹配），两个请求合并 batch 只 prefill "Python?" 和 "C++?" 各一次。

## 5. 多模态扩展的挑战

SGLang 的 Radix Tree 原生为文本 token 序列设计。在多模态 VLM 服务中，visual token 的加入带来了几个核心挑战：

1. **Visual token 的 Hash 不可靠**：文本 token 的 Radix Tree 匹配基于 token ID 的精确匹配。但 visual token 是连续向量值（浮点），不同图像产生的 visual token 完全不同，无法通过 Radix Tree 匹配。
2. **文本-Visual 边界**：prompt 序列通常为 `[text_prefix] + [image_token] + [text_suffix]`，其中 `[text_prefix]` 可共享，但 `[image_token]` 不可共享。Radix Tree 需要在到达 `[image_token]` 时停止匹配。
3. **Visual token 的单独索引**：一种可能的扩展是对 visual token 使用独立的哈希索引（基于图像内容的 embedding hash），与 Radix Tree 的文本 prefix 耦合。

## 6. 对本项目的启发

SGLang 的 RadixAttention 对本项目的启发主要体现在：

1. **Prefix caching 的粒度**：vLLM 是 block 级 hash，SGLang 是 token 级 tree。block 级实现简单、效率高；token 级命中率更高、更灵活。minivLLM 当前采用 block 级（vLLM 风格），后续如果多轮对话场景增多，可考虑引入 token 级索引。
2. **Visual token 的共享难题**：多模态 prefix caching 的核心难点是 visual token 的唯一性问题。SGLang 的树结构在这个场景下反而成为优势——在文本→visual token 的边界可以自然地分叉，文本 prefix 依然共享。
3. **引用计数 = 零成本共享**：Radix Tree 的节点引用计数机制使得 KV cache 的共享不需要额外的显存开销。同一个 tree 节点下的 block 被多个请求同时引用，逻辑清晰。minivLLM 目前的 block table 是一对一的，未来支持 batch scheduling 时需引入类似的共享机制。
