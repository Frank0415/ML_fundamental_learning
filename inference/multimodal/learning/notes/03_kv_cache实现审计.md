# 03 — KV Cache 实现审计

> **审核日期**: 2026-06-07
> **审核目标**: `minivLLM/minivllm/core/kv_cache.py` + 全仓库交叉引用
> **审核方法**: 静态分析 + grep 搜索 `KVCache`、`kv_cache` 引用

---

## 1. KV Cache 是什么，为什么需要它？

在自回归解码中，每生成一个新 token，模型需要计算该 token 对所有历史 token 的 attention。如果每次都重新计算完整序列的 K 和 V（O(n²) 的 attention），随着序列增长，延迟会迅速膨胀。

KV cache 的思路很简单：**缓存之前所有 token 的 K 和 V，每次只需计算新 token 的 Q 与所有缓存的 K/V 做 attention**。这样每个 decode step 的 attention 复杂度从 O(n²) 降到 O(n)（仍需要线性扫描所有 K/V，但不重新投影）。

一个合格的 KV cache 引擎需要：
1. **写入**：prefill 结束后将所有 token 的 K/V 写入 cache
2. **追加**：每个 decode step 将新 token 的 K/V 追加到 cache
3. **读取**：decode 时读取完整的 K/V
4. **内存管理**：分配、回收、重排 cache 空间

---

## 2. `KVCache` 类的代码审计

### 2.1 初始化 (`kv_cache.py` 行 5-24)

```python
class KVCache(nn.Module):
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim,
                 *, device=None, dtype=None):
        super().__init__()
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        shape = (num_layers, max_seq_len, num_kv_heads, head_dim)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer("k_cache", torch.empty(shape, **factory_kwargs), persistent=False)
        self.register_buffer("v_cache", torch.empty(shape, **factory_kwargs), persistent=False)
```

缓存形状：`(num_layers, max_seq_len, num_kv_heads, head_dim)`

**这就是 contiguous buffer**。它为每个 layer 预分配了一块连续的 `(max_seq_len, num_kv_heads, head_dim)` 空间。序列中的每个 token 占据一个固定位置，token_0 写到 `[layer, 0, :, :]`，token_1 写到 `[layer, 1, :, :]`，依此类推。

对于单序列推理，contiguous buffer 完全够用。但它在以下场景下有严重问题：

- **多序列 batch**：每个序列长度不同，contiguous 分配导致大量内存浪费
- **序列动态增长**：某序列从 100 增长到 200，中间可能需要"移动"数据
- **序列完成/取消**：释放的内存空间不连续，形成碎片

这正是 paged attention 要解决的问题。

### 2.2 `write` 方法 (`kv_cache.py` 行 31-55)

```python
def write(self, layer_idx, positions, k, v):
    self._check_layer_idx(layer_idx)
    positions_tensor = self._normalize_positions(positions, k.device)
    self._check_kv(k, v, positions_tensor.numel())
    if positions_tensor.min() < 0 or positions_tensor.max() >= self.max_seq_len:
        raise IndexError("positions out of KV cache range")
    cache_positions = positions_tensor.to(device=self.k_cache.device, dtype=torch.long)
    self.k_cache[layer_idx, cache_positions] = k.to(...)
    self.v_cache[layer_idx, cache_positions] = v.to(...)
```

功能：将 `k` 和 `v`（形状 `(num_tokens, num_kv_heads, head_dim)`）写入指定 layer 的指定位置。

位置参数 `positions` 可以是单个整数（单 token decode）或 1-D tensor（多 token prefill）。

逻辑审计：
- 边界检查正确（`positions.min() >= 0` 且 `positions.max() < max_seq_len`）
- 支持批量写入，通过 `cache_positions` 索引
- 数据类型和设备的自动转换（to device, to dtype）

**结论**：`write` 方法实现正确。

### 2.3 `read` 方法 (`kv_cache.py` 行 57-62)

```python
def read(self, layer_idx, end_pos):
    self._check_layer_idx(layer_idx)
    if end_pos < 0 or end_pos > self.max_seq_len:
        raise IndexError("end_pos out of KV cache range")
    return self.k_cache[layer_idx, :end_pos], self.v_cache[layer_idx, :end_pos]
```

功能：读取指定 layer 的 `[0, end_pos)` 范围内的所有 K/V。

**关键假设**：`read` 总是从位置 0 开始读取。它不支持随机位置读取（例如，某些序列的 cache 起始位置不在 0）。对于单序列 contiguous buffer，这没问题；对于多序列 paged，需要 block 级别的读取。

**结论**：`read` 方法对于 contiguous buffer 是正确的。

### 2.4 辅助方法

- `reset()` 将全部 cache 清零（用于测试/调试）
- `layer_cache(layer_idx)` 返回一个 layer 的完整 k_cache 和 v_cache slice
- `_check_layer_idx`、`_check_kv`、`_normalize_positions`：内部校验方法，实现正确

---

## 3. 交叉引用审计：谁用了 KV Cache？

在整个 `multimodal/minivLLM/` 仓库中搜索 `KVCache`、`kv_cache`：

```
minivllm/
├── core/kv_cache.py       ← 定义 KVCache
├── model/qwen3.py         ← 无引用
├── layers/numpy/*.py      ← 无引用（attention.py 不含 KVCache）
├── utils/context.py       ← 无引用
├── config.py              ← 无引用（虽有 kvcache 字段但无 KVCache import）
└── validate_model.py      ← 无引用
```

**结论：`KVCache` 在整个引擎中没有任何 forward 引用。** 它是完全孤立的 dead code。

这意味着：
1. `validate_model.py` 的生成循环每次调用 `model(input_ids, positions)` 都是独立的全序列 attention
2. 没有缓存机制来避免重复计算
3. 随着序列增长（例如生成了 100 个 token），每个新 token 都会对前面 100+ 个 token 重新计算 attention，导致 O(n²) 的 decode 开销

---

## 4. 当前的 decode 在 `validate_model.py` 中是如何工作的？

```python
# validate_model.py 行 214-223
for _ in range(8):
    hidden = mini_model(input_ids, positions)       # 每次执行全序列 attention
    last_logits = mini_model.compute_logits(hidden[-1:])  # 只取最后一个位置的 logits
    next_token = last_logits.argmax(dim=-1).item()
    generated.append(next_token)
    input_ids = torch.tensor(generated, device=device)
    positions = torch.arange(len(generated), device=device)
```

这是"暴力 decode"：每次把整个序列重新送入模型，让 attention 重算一切。它在功能上是正确的（attention + causal mask 保证每个位置只受前面位置影响），但在效率上是灾难性的。

如果使用 KV cache，理想的 decode 循环应该是：

```python
# 伪代码
k_cache = KVCache(num_layers, max_seq_len, ...)
for _ in range(8):
    # prefill 阶段（仅第一次）
    hidden, k, v = model.forward_with_cache(input_ids, positions, k_cache)
    k_cache.write(layer_idx, positions, k, v)
    next_token = sample(logits[-1])
    # decode 阶段
    hidden, k_new, v_new = model.forward_with_cache(next_token_id, new_position, k_cache)
    k_cache.write(layer_idx, new_position, k_new, v_new)
    ...
```

这需要 `model.forward` 具备"读取 KV cache 并只计算增量 attention"的能力。当前的 `Qwen3Attn.forward` 和 `Attn.forward` 都不具备这个能力。

---

## 5. 与 Context 的关系

`Context` dataclass（`utils/context.py`）定义了 `slot_mapping` 和 `block_tables` 字段，这是 paged attention KV cache 所需的元数据。但 `KVCache` 本身不使用这些字段，`Context` 也不引用 `KVCache`。

两个模块之间没有连接。

---

## 6. 总结

| 项目 | 状态 |
|------|------|
| `KVCache` 类定义 | 存在，实现正确（contiguous buffer） |
| `KVCache.write` | 实现正确 |
| `KVCache.read` | 实现正确（仅支持从 0 读取） |
| `KVCache` 的 forward 引用 | **为零**（未被任何模型代码 import/调用） |
| KV cache 接入 attention | **不存在**（`Attn.forward` 不读 cache） |
| KV cache 接入 decode | **不存在**（`validate_model.py` 逐次全序列重算） |
| 与 paged attention 的关系 | `KVCache` 为 contiguous buffer，不是 paged |

**引擎当前的 KV cache 状态**：有一个正确实现但完全未使用的 contiguous cache。将它接入 forward 是实现增量解码的第一步。在此之前，所有推理都是 O(n²) 的 full-sequence attention。
