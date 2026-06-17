# 02 — Attention 实现审计

> **审核日期**: 2026-06-07
> **审核目标**: `minivLLM/minivllm/layers/numpy/attention.py` + `minivLLM/minivllm/model/qwen3.py` 中的 `Qwen3Attn`
> **审核方法**: 静态分析，不 import 引擎，不修改文件

---

## 1. 当前 attention 栈的两层结构

minivLLM 的 attention 系统由两个不同层级的代码组成：

1. **通用算子层**：`attention.py` 中的 `Attn` 类，实现标准 MHA（多头注意力）+ GQA（分组查询注意力）+ causal mask
2. **模型特定层**：`qwen3.py` 中的 `Qwen3Attn` 类，封装了 QKV 投影、RoPE 注入、调用 `Attn`、输出投影

思路没错，但两层之间的接口存在致命的不匹配。

---

## 2. `Attn` 类的逐行审计

### 2.1 初始化 (`attention.py` 行 33-46)

```python
class Attn(nn.Module):
    def __init__(self, num_heads, head_dim, num_kv_heads, is_causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.n_reps = self.num_heads // self.num_kv_heads
        assert self.n_reps != 0
        self.head_dim = head_dim
        self.is_causal = is_causal
```

构造函数接收四个参数：`num_heads`（query head 数）、`head_dim`（每个 head 的维度）、`num_kv_heads`（KV head 数，用于 GQA）、`is_causal`（是否使用 causal mask）。

逻辑审查：
- GQA 的重复因子 `n_reps` 计算正确：`num_heads // num_kv_heads`
- 断言 `n_reps != 0` 是正确的放卫（防止 `num_heads < num_kv_heads` 的无效配置）
- 保存 `is_causal` 作为实例变量，供 forward 使用

**结论**：初始化逻辑正确。

### 2.2 repeat_kv 函数 (`attention.py` 行 5-14)

```python
def repeat_kv(k: torch.Tensor, v: torch.Tensor, n_reps: int):
    if n_reps == 1:
        return k, v
    k = k[:, :, :, None, :].expand(...)
    v = v[:, :, :, None, :].expand(...)
    return k.reshape(...), v.reshape(...)
```

将 KV head 从 `(B, S, kv_heads, head_dim)` 扩展到 `(B, S, num_heads, head_dim)`，以便与 Q 做 batch matmul。

实现方式：通过 `unsqueeze + expand + reshape`，不分配额外内存（`expand` 返回 view）。这与 PyTorch 的 `_repeat_kv` 函数等价。

**结论**：实现正确。但需要注意 `expand` 不检查边界，如果被调用时 shape 不对，会静默产生错误结果而非报错。

### 2.3 forward 方法 (`attention.py` 行 48-70)

```python
def forward(self, q, k, v):
    k, v = repeat_kv(k, v, self.n_reps)
    q = q.transpose(1, 2)  # [B, S, H, D] → [B, H, S, D]
    k = k.transpose(1, 2)  # [B, S, H, D] → [B, H, S, D]
    v = v.transpose(1, 2)

    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)

    if self.is_causal:
        scores = scores + causal_mask(q.size(2), k.size(2), device=scores.device)
    attn_out = torch.matmul(softmax(scores, dim=-1), v)
    return attn_out.transpose(1, 2).contiguous()
```

执行的步骤：
1. 将 KV head 扩展到与 Q head 数匹配（GQA）
2. 转置 Q/K/V 到 `[B, H, S, D]` 格式（batch matmul 需要 head 在第二维）
3. 计算 `Q × K^T / sqrt(dk)`
4. 如果 `is_causal=True`，加上 `causal_mask`（上三角 `-inf`）
5. Softmax + V 加权
6. 转回 `[B, S, H, D]` 并 contiguous

逻辑审查：
- 维度操作正确：`transpose(1,2)` → `matmul` → `transpose(1,2)` 是标准 MHA 流程
- 除以 `sqrt(dk)` 是标准缩放
- `causal_mask` 只在上三角填 `-inf`，对角线上不 mask（允许当前 token attend 到自己）

**重要限制**：`causal_mask(q.size(2), k.size(2))` 总假设查询的起始位置是 0。在 KV cache 的 decode 场景中，query 的绝对位置是 `pos` 而非 0，因此 `causal_mask` 不应该从 `(0,0)` 开始计算。这个局限性在 `causal_mask` 函数的注释中已有说明（行 23-26），但目前尚未修复。

**结论**：`Attn.forward` 对于 **无 KV cache 的 full-sequence attention** 是正确的。对于 **有 KV cache 的 decode** 需要扩展 `causal_mask`。

---

## 3. `Qwen3Attn` 的逐行审计

### 3.1 初始化 (`qwen3.py` 行 12-59)

```python
class Qwen3Attn(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, ...):
        ...
        self.qkv_proj = Linear(hidden_size, self.head_dim * (self.num_heads + self.kv_heads * 2), ...)
        self.o_proj = Linear(self.head_dim * self.num_heads, hidden_size)
        self.rotary_emb = get_rope(...)
        self.attn = Attn(
            num_heads=self.num_heads,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            S=self.head_dim,        # ← 不存在于 Attn.__init__
            is_decode=True,          # ← 不存在于 Attn.__init__
        )
```

QKV 投影矩阵的 out_features 计算：`head_dim * (num_heads + kv_heads * 2)`，对应 Q（num_heads 个 head）+ K（kv_heads 个 head）+ V（kv_heads 个 head）= `q_size + kv_size + kv_size`。这个计算是正确的。

输出投影 `o_proj` 将 num_heads 个 head 拼接后投影回 hidden_size。正确。

**RoPE 的构造**：使用 `get_rope(head_size=self.head_dim, rotary_dim=self.head_dim, ...)`。这里 `rotary_dim == head_dim` 意味着每个 head 的全维度都参与旋转，正确的 RoPE 默认行为。

**Attn 构造的错误**：`Qwen3Attn` 向 `Attn()` 传入了两个不存在的参数：
- `S`：预期含义可能是 `head_dim`（在 FlashAttention 或某些实现中 `S` 表示序列维度），但 `Attn.__init__` 的签名中没有这个参数
- `is_decode`：预期含义可能是"是否处于 decode 阶段"，但 `Attn.__init__` 只有 `is_causal` 参数

**结论**：`Qwen3Attn()` 构造时立即抛出 `TypeError: __init__() got an unexpected keyword argument 'S'`。这是阻塞级 Bug（B1）。

### 3.2 forward (`qwen3.py` 行 61-75)

```python
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim).unsqueeze(0)
    k = k.view(-1, self.kv_heads, self.head_dim).unsqueeze(0)
    v = v.view(-1, self.kv_heads, self.head_dim).unsqueeze(0)
    if not self.qkv_bias:
        q = self.q_norm(q)
        k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)
    o = self.attn(q, k, v)
    output = self.o_proj(o.flatten(2)).squeeze(0)
    return output
```

这个 forward 在 `Attn` 构造成功的前提下是正确的：
- `qkv_proj → split → view/unsqueeze`：将线性投影输出拆解为 Q/K/V 并 reshape 为 `[B, S, H, D]`
- `q_norm/k_norm`：Qwen3 特有的 per-head QK 归一化（仅当无 bias 时）
- `rotary_emb(positions, q, k)`：对 Q 和 K 施加 RoPE
- `self.attn(q, k, v)`：执行标准 MHA（带 GQA + causal mask）
- `o.flatten(2)`：将 `[B, S, H, D]` 展平为 `[B, S, H*D]`，然后 o_proj 投影
- `squeeze(0)`：假设 batch=1

**一个微妙的问题**：`flatten(2)` 展平的是 dim 2 和 dim 3（head 和 head_dim），正确地将所有 head 拼接。但 `squeeze(0)` 硬编码了 batch=1，如果未来支持 batch>1 会出错。

**结论**：forward 逻辑正确（如果 `Attn` 可成功构造）。

---

## 4. GQA（分组查询注意力）的实现细节

minivLLM 的 GQA 实现是通过 `repeat_kv` 完成的。以 Qwen3-0.6B 为例：`num_heads=16, kv_heads=8, n_reps=2`。

K/V 的形状是 `[B, S, 8, D]` → `repeat_kv` 扩展为 `[B, S, 16, D]` → 与 Q `[B, S, 16, D]` 做 matmul。

这种实现是"naive GQA"，不优化 KV head 的共享模式。高性能引擎（如 vLLM）会在 CUDA kernel 中直接利用 `kv_heads < num_heads` 来减少内存访问。minivLLM 作为学习引擎，naive 实现是合理的。

---

## 5. Causal Mask 的局限

当前 `causal_mask` 的实现（`attention.py` 行 19-28）：

```python
def causal_mask(seq_q: int, seq_k: int, device="cpu"):
    return torch.triu(torch.full((seq_q, seq_k), float("-inf"), device=device), diagonal=1)
```

这个 mask 的语义是：query 位置 `i` 只能 attend 到 key 位置 `j <= i`。但在 KV cache 场景中：

- **Prefill**：`seq_q = seq_k = prompt_len`，mask 从 `(0,0)` 开始，正确
- **Decode**：`seq_q = 1`（单个新 token），`seq_k = cache_len + 1`（所有历史 token + 新 token），需要知道 query 的绝对位置来正确 mask。当前 `causal_mask(1, cache_len + 1)` 只 mask 掉 `j > 0` 的位置，但如果 query 的绝对位置是 `pos`，实际应该 mask `j > pos`。

后续实现 KV cache 时需要扩展 `causal_mask` 接受 `query_start_pos` 参数。

---

## 6. 审计总结

| 组件 | 状态 | 问题 |
|------|------|------|
| `Attn.__init__` | 已实现且可运行 | — |
| `Attn.forward` | 已实现且可运行 | `causal_mask` 对 decode 场景有限制 |
| `repeat_kv` | 已实现且可运行 | — |
| `Qwen3Attn.__init__` | 已实现但不确定正确 | 向 `Attn` 传入不存在参数 → TypeError |
| `Qwen3Attn.forward` | 已实现且可运行 | 假设 batch=1（`squeeze(0)`）；逻辑正确 |
| GQA 支持 | 已实现 | naive repeat_kv，无优化 |
| Causal Mask | 已实现且可运行 | 全序列场景正确，decode 场景需扩展 |
| Paged Attention 接入 | 未实现 | `Attn.forward` 无 block table 引用 |
