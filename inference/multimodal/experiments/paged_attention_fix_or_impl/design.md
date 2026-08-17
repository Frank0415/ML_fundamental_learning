# Paged Attention：固化最小设计

版本：Wave 1 / v0.1
与 minivLLM 配置绑定：`kvcache_block_size=16`，`num_kvcache_blocks=-1`（自动），`gpu_memory_utilization=0.9`

---

## 1. 问题定义

minivLLM 当前存在 KV 缓存的两种路径风险：

1. **未实现**：`paged_attention` 是占位，实际 fallback 到 contiguous 实现，显存碎片化未计入。
2. **阻塞**：如果 contiguous KV 预分配过大会直接 OOM，过小则 max_model_len 受限。

本设计给出最小接口集，使得 minivLLM 能在 中低显存 GPU 上将 KV 缓存切分为 block_size=16 的固定块，按需分配、即时释放。不引入 scheduler 或 continuous batching，那是 Wave 2+ 的范畴。

---

## 2. 核心数据结构

### 2.1 BlockTable

每个请求维护一张 BlockTable，记录该请求占有哪些物理 block。

```
BlockTable:
  - physical_block_ids: list[int]   # 按 token 顺序排列的物理块 ID
  - num_tokens: int                 # 当前已存储的 token 数
  - num_blocks: int                 # 已分配块数 = ceil(num_tokens / block_size)
```

**不变量**：

- `num_tokens ≤ num_blocks × block_size`
- `physical_block_ids` 的元素唯一（一个物理块只被一个请求持有）。
- `free_request()` 调用时必须把 `physical_block_ids` 归还给 `BlockManager.free_blocks` 池。

### 2.2 BlockManager

全局单例，管理所有物理 KV 块的生命周期。

```
BlockManager:
  - total_blocks: int               # GPU 上可分配的 KV 块总数
  - free_blocks: set[int]           # 未分配的物理块 ID
  - allocated_blocks: set[int]      # 已分配的物理块 ID
  - block_size: int                 # 从 config.kvcache_block_size 读取，固定 16
  - reserved_system_blocks: int     # 系统保留块数（如 0-3 给框架开销）
```

**关键方法**：

- `allocate_block() -> int`：从 `free_blocks` 弹出一个块 ID，加入 `allocated_blocks`。若 `free_blocks` 为空，抛出 `OutOfKVBlocks`。
- `free_blocks_of_request(block_ids: list[int])`：将 `block_ids` 中的所有块从 `allocated_blocks` 移回 `free_blocks`。
- `available_blocks() -> int`：返回 `len(free_blocks)`。
- `fragmentation_ratio() -> float`：见 §3 指标。

### 2.3 RequestState

每个推理请求的运行态，包含 BlockTable 引用。

```
RequestState:
  - request_id: str
  - block_table: BlockTable
  - seq_len: int                    # prefill 完成后的序列长度
  - is_prefill_done: bool
  - max_tokens: int                 # 最大生成 token 数（含 prompt）
```

**生命周期**：

1. `prefill_allocate()` 创建 `RequestState` 并分配 `ceil(prompt_len / block_size)` 个块。
2. 每次 decode，`decode_append()` 检查是否需要追加新块。
3. 请求结束（EOS / max_tokens 达到），`free_request()` 回收所有块。

### 2.4 PagedKVCache

张量容器，持有 `(total_blocks, num_layers, 2, block_size, num_heads, head_dim)` 形状的 GPU 张量，两份子张量分别对应 K 和 V。

```
PagedKVCache:
  - kv_tensor: Tensor               # shape: (total_blocks, num_layers, 2, block_size, num_heads, head_dim)
  - write_kv(block_id, layer, k_or_v, token_offset, data)
  - read_kv(block_id, layer, k_or_v, token_offset) -> Tensor
  - gather_kv_for_attention(physical_block_ids, layer, start_token, end_token) -> (k, v)
```

**`gather_kv_for_attention()` 职责**：

根据 `physical_block_ids` 中 token 的排列，将分布在多个物理块中的 K/V 子段拼接成一个连续张量，供 attention 算子消费。这是 paged attention 与 contiguous attention 唯一的关键切换点，在这一步之前/之后，其余计算（Q 投影、softmax、输出投影）完全相同。

---

## 3. 统计指标

以下指标由 `BlockManager` 在每次分配/释放后更新，可通过 `BlockManager.stats()` 一键导出。

| 指标名 | 类型 | 含义 | 推导 |
|--------|------|------|------|
| `allocated_blocks` | int | 当前所有请求占用的块数 | `len(allocated_blocks)` |
| `free_blocks` | int | 未被占用的块数 | `len(free_blocks)` |
| `total_blocks` | int | GPU KV 块总数 | `allocated_blocks + free_blocks + reserved_system_blocks` |
| `used_tokens` | int | 所有请求实际存储的 token 数 | `sum(req.block_table.num_tokens for req in active_requests)` |
| `wasted_slots` | int | 已分配但未使用的槽位 | `allocated_blocks * block_size - used_tokens` |
| `fragmentation_ratio` | float | 碎片率 | `wasted_slots / (allocated_blocks * block_size)`，值越接近 0 越好 |

**量化通过阈值**：`fragmentation_ratio ≤ 0.10` 视为健康。单请求场景下，`wasted_slots ≤ block_size`（即最多浪费一个块），所以 `fragmentation_ratio ≤ 1/num_blocks_per_req`。

---

## 4. 正确性门槛：contiguous vs paged logits 对齐

这是 paged attention 实现的"量化通过阈值"。

**测试场景**：

1. 用随机 token ID 序列（长度 N）做一次 prefill。
2. 用 contiguous KV 实现算一遍 attention，得到参考 logits（fp32）。
3. 用 paged KV 实现（`gather_kv_for_attention()` 返回拼接后的 K/V）算同一轮 attention。
4. 对比两者 logits。

**阈值**：

```python
torch.allclose(logits_paged, logits_contiguous, atol=1e-5, rtol=1e-5)
```

对于 decode 步骤，用 KV cache 逐步追加的方式验证。每步 `decode_append()` 后做一次对齐检查。

**不通过的情况**：

- 拼接逻辑有 offset 错误（block 内 token 顺序错位）。
- 多 block 跨块时 K/V 的 padding 位置被误读。
- `block_size` 不能整除 head_dim 导致 reshape 引入精度损失（需要用非整除 block_size 额外测试，当前 block_size=16 可避开此问题）。

---

## 5. 非目标（明确排除）

本设计为 **Wave 1 固化**，以下内容 **不在范畴内**：

1. **不做 scheduler**：不实现 continuous batching、不实现 request 优先级队列、不实现 prefill/decode 交错调度。只有一个 `run_request()` 顺序执行。
2. **不做 prefix sharing**：每个请求独立分配 block，不共享 system prompt 的 KV。
3. **不写定制 CUDA kernel**：所有操作使用 PyTorch 原生 op（`torch.gather`、`torch.cat`、`torch.index_select` 等）。定制 kernel 留到 Wave 3 性能优化。
4. **不做训练 / LoRA / 量化感知调优**：仅推理，仅 bf16/fp16。
5. **不做分布式**：单卡，`tensor_parallel_size=1`。
6. **不做 speculative decoding**：不考虑 draft model、tree attention 等。
7. **第一版不做 prefix caching / cascade attention**：这些是 Wave 2+ 的优化项。

以上非目标明确排除后，Wave 1 的唯一交付物是一个可验证的 paged KV 核心，`BlockManager` + `PagedKVCache` + 对齐测试。

---

## 6. 接口签名草图

| 接口名 | 输入 | 输出 | 备注 |
|--------|------|------|------|
| `BlockManager.__init__(total_blocks, block_size, reserved_system_blocks)` | `total_blocks: int, block_size: int, reserved_system_blocks: int` | 无 | 初始化 free/allocated 集合 |
| `BlockManager.allocate_block()` | 无 | `block_id: int` | 无空闲块时抛 `OutOfKVBlocks` |
| `BlockManager.free_blocks_of_request(block_ids)` | `block_ids: list[int]` | 无 | 将指定块归还 free 池 |
| `BlockManager.available_blocks()` | 无 | `int` | 空闲块数量 |
| `BlockManager.fragmentation_ratio()` | 无 | `float` | [(allocated * block_size - used_tokens) / (allocated * block_size)] |
| `BlockManager.stats()` | 无 | `dict[str, int \| float]` | 导出所有指标 |
| `BlockTable.__init__(block_size)` | `block_size: int` | 无 | 空表 |
| `BlockTable.append_block(block_id)` | `block_id: int` | 无 | 追加物理块 |
| `BlockTable.token_to_block_offset(token_idx)` | `token_idx: int` | `(block_idx, offset_in_block)` | token 位置→块内偏移 |
| `BlockTable.num_tokens` (property) | 无 | `int` | 当前 token 数 |
| `RequestState.__init__(request_id, max_tokens)` | `request_id: str, max_tokens: int` | 无 | 初始化，block_table 为空 |
| `prefill_allocate(block_mgr, request, prompt_len)` | `block_mgr: BlockManager, request: RequestState, prompt_len: int` | 无 | 分配 ceil(prompt_len/block_size) 个块 |
| `decode_append(block_mgr, request)` | `block_mgr: BlockManager, request: RequestState` | `new_block_id: int \| None` | 若当前块已满则分配新块 |
| `free_request(block_mgr, request)` | `block_mgr: BlockManager, request: RequestState` | 无 | 回收所有块并清除 request |
| `PagedKVCache.__init__(total_blocks, num_layers, num_heads, head_dim, block_size, dtype)` | `total_blocks: int, num_layers: int, num_heads: int, head_dim: int, block_size: int, dtype: torch.dtype` | 无 | 分配 (total_blocks, num_layers, 2, block_size, num_heads, head_dim) 张量 |
| `PagedKVCache.write_kv(block_id, layer, k_or_v, token_offset, data)` | `block_id: int, layer: int, k_or_v: int (0=K,1=V), token_offset: int, data: Tensor` | 无 | 写入 KV |
| `PagedKVCache.read_kv(block_id, layer, k_or_v, token_offset)` | `block_id: int, layer: int, k_or_v: int, token_offset: int` | `Tensor` | 读取 KV |
| `PagedKVCache.gather_kv_for_attention(physical_block_ids, layer, start_token, end_token)` | `physical_block_ids: list[int], layer: int, start_token: int, end_token: int` | `(k: Tensor, v: Tensor)` | 拼接多块 K/V 为连续张量 |
