# Text Encoder 与 Prompt Embedding Cache — 扩散推理中的"条件编码"优化

> **对应任务**：T12
> **产出日期**：2026-06-07
> **前置阅读**：`learning/notes/03_diffusion推理数据流.md`（第 8 节）、`learning/notes/06_cfg和negative_prompt.md`

---

## 1. Text Encoder 在扩散模型中的角色

### 与 LLM 的根本不同

在 LLM 推理中，text encoder 不是独立模块——模型本身就是一个 text-to-text transformer。但在扩散模型中，**text encoder 是独立的前置模块**，其输出在整个 denoising loop 中**不变**。

**LLM 推理**：
```
token → model (每步回归输出下一个 token) → model → model → ...
       ↑ 每次 forward 都重新计算 text 的表示
```

**扩散推理**：
```
prompt → text_encoder → text embedding（不变！）
                           ↓
                    [denoising loop: step 1..N]
                           ↓
                    denoiser(x_t, t, text_embedding)
                           ↓
                    VAE decoder → image
```

**关键洞察**：text encoder 只需要跑一次！同一 prompt 的 embedding 在所有 denoising steps 中完全复用。这就是 prompt embedding cache 的优化基础。

### 真实模型中的 Text Encoder

现代扩散模型（SD3、FLUX）通常使用多个 text encoder：

| 模型 | Text Encoder(s) | 输出维度 | 用途 |
|------|----------------|---------|------|
| SD3 Medium | CLIP-L + CLIP-G + T5-XXL (可选) | pooled: 768+1280+4096 | pooled → AdaLN modulation<br>seq → cross-attention |
| FLUX.1-dev | CLIP-L + T5-XXL | pooled: 768+4096 | 同上 |
| Sana | Gemma-2B (轻量 LLM) | 2304 | 仅 seq embedding（无 pooled） |
| CogVideoX | T5-XXL | 4096 | 视频帧间共享 text embedding |

**计算代价**：（以 SD3 Medium 为例）
- CLIP-L: ~430M 参数，约 1.5GB fp16
- CLIP-G: ~1.3B 参数，约 2.6GB fp16
- T5-XXL: ~11B 参数，约 22GB fp16（通常 offload 或不加载）

在 中等显存配置 上，加载 CLIP-L + CLIP-G 已经占用约 4.1GB，T5-XXL 必须 offload。

---

## 2. 为什么需要 Prompt Embedding Cache

### 缓存收益估算

**无缓存场景**（naive 每次 encode）：

```
# CFG 的每次 forward 都需要 cond 和 uncond embedding
for step in range(28):
    cond_emb = text_encoder.encode("a cat")     # ❌ 重复编码！
    uncond_emb = text_encoder.encode("")        # ❌ 重复编码！
    v_cond = denoiser(x_t, t, cond_emb)
    v_uncond = denoiser(x_t, t, uncond_emb)
    ...
```

**有缓存场景**：

```
# 首次 encode（cache miss）
cond_emb = text_encoder.encode("a cat")     # 第 1 次 → miss → 存入 cache
uncond_emb = text_encoder.encode("")        # 第 1 次 → miss → 存入 cache

# 后续所有 step（cache hit）
for step in range(28):
    cond_emb = text_encoder.encode("a cat")     # hit ✓（查表，~0 cost）
    uncond_emb = text_encoder.encode("")        # hit ✓（查表，~0 cost）
    v_cond = denoiser(x_t, t, cond_emb)
    v_uncond = denoiser(x_t, t, uncond_emb)
    ...
```

**收益分析**：

- **encode 次数**：从 28 × 2 = 56 次降为 2 次
- **时间节省**：CLIP-L encode 一次约 50–100ms → 省 (56-2) × 75ms ≈ 4s
- **显存节省**：无重复中间激活（text encoder 的 forward 中间态不必保留）

**更关键的收益**：sequential CFG 的两次 forward 各需要一次 encode（无 cache 时），但缓存后两个 forward 共用同一份 embedding。对于 batched CFG，一次 forward 就需要两份 embedding——缓存避免了一轮 text encoder forward。

### 缓存寿命

- **同一 prompt 的不同 seed**：embedding 完全复用（seed 只影响初始噪声，不影响 text encoder）
- **同一 prompt 的不同 cfg_scale**：embedding 完全复用
- **同一 prompt 的不同 num_steps**：embedding 完全复用
- **切换 prompt**：miss，需要重新 encode

---

## 3. Cache Key 设计

### 必需的 6 个字段

| 字段 | 类型 | 说明 | 为什么必须 |
|------|------|------|-----------|
| `prompt` | str | 正向提示文本 | 不同 prompt → 不同 embedding，必须区分 |
| `negative_prompt` | str | 负向提示文本 | 同上 |
| `max_seq_len` | int | 最大序列长度 | 不同长度 → 不同 padding，tensor shape 不同 |
| `hidden_size` | int | 输出维度 | 不同模型 → 不同 hidden_size |
| `dtype` | str | 数据类型（fp16/fp32） | 不同精度 → 不同 tensor |
| `device` | str | 设备（cpu/cuda:0） | 不同设备 → 不能直接复用（需 .to() 搬迁） |

**Key 生成方式**：

```python
raw = f"{prompt}|{negative_prompt}|{max_seq_len}|{hidden_size}|{dtype}|{device}"
key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

使用 SHA256 hash 可避免 key 过长（prompt 可能几百字），并保证确定性。

**需要注意的细节**：
- `prompt` 需要 trim/normalize 吗？建议不做归一化——让 hash 精确反映用户输入。用户换一个空格也应该是新 cache entry。
- `device` 是字符串如 `"cuda:0"`——不同 GPU 之间的 buffer 不能直接共享，必须通过 `tensor.to()` 搬迁。

---

## 4. 缓存实现策略

### ToyTextConditioner（T12 实现）

完整实现 prompt embedding cache。由于是 toy 模型（随机 embedding），cache 的实际收益无法测量，但**接口和 key 逻辑完全真实**。

```python
class ToyTextConditioner:
    def __init__(self, hidden_size=64, max_seq_len=16, seed=42, ...):
        self._cache: dict[str, PromptEmbeddings] = {}
        self._hits: int = 0
        self._misses: int = 0

    def encode(self, prompt, negative_prompt="", max_seq_len=None):
        key = self._cache_key(prompt, negative_prompt, ...)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        embeddings = self._generate_embedding(prompt, max_seq_len)
        self._cache[key] = embeddings
        return embeddings
```

**Cache 一致性保证**：
- 相同 prompt → 相同 seed → 相同 embedding（确定性生成器）
- 使用 `seed + hash(prompt)` 作为子种子，确保不同 prompt 有不同 embedding

### 真实实现的考虑（T13+ HFCachedTextConditioner）

1. **多 text encoder 的缓存**：如果同时使用 CLIP-L + CLIP-G，需要缓存两者的输出（或分别缓存）
2. **tokenizer 的确定性**：tokenizer 的 output 可能受 padding/truncation 策略影响，需在 cache key 中体现
3. **offload 场景**：如果 text encoder 在 encode 后被 offload 到 CPU，下次 cache miss 时需要重新 load → 时间成本更高 → cache 收益更大
4. **内存 vs 时间 tradeoff**：缓存过多 prompt（如 100+ 个不同 prompt）会占用大量内存，可以设置 LRU 淘汰

---

## 5. 补充：Diffusion 的 Cache 为什么不是 LLM KV Cache

这是一个常见的混淆点。两者都叫"cache"，但本质完全不同：

### LLM KV Cache

```
t=0: "The"     → [K_0, V_0] 存入 cache
t=1: " cat"    → attention(Q_1, [K_0, K_1], [V_0, V_1])  ← 复用历史 K/V
t=2: " sat"    → attention(Q_2, [K_0, K_1, K_2], [V_0, V_1, V_2])
...
```

- **存储内容**：每层 self-attention 的 key/value 投影
- **跨步复用**：token t+1 的 attention 需要过去所有 token 的 K/V
- **随序列增长**：KV cache size ∝ seq_len × num_layers × hidden_dim
- **不可丢弃**：丢弃任何一步的 KV 会导致信息丢失

### Diffusion Prompt Embedding Cache

```
step 1: denoiser(x_1, t_1, cond_emb)  ← 使用 cond_emb
step 2: denoiser(x_2, t_2, cond_emb)  ← 使用相同的 cond_emb！
...
step N: denoiser(x_N, t_N, cond_emb)  ← 使用相同的 cond_emb！
```

- **存储内容**：text encoder 的最终输出（pooled + seq embedding）
- **跨步复用**：所有 denoising step 使用相同的 cond/uncond embedding
- **固定大小**：不随 denoising step 增长
- **本质**：这是一个**输入缓存**，不是**中间状态缓存**

### 总结表格

| 维度 | LLM KV Cache | Diffusion Prompt Embedding Cache |
|------|-------------|--------------------------------|
| 缓存什么 | attention 的 K/V 历史矩阵 | text encoder 的输出 tensor |
| 为什么跨步复用 | 过去 token 的 K/V 在下一 token 计算时必需 | 同一个 prompt 在所有 denoising steps 中不变 |
| 大小增长 | ∝ 序列长度 | 固定 |
| 存储位置 | GPU 显存（高性能） | GPU 显存或 CPU 内存（可 offload） |
| 失效条件 | 序列结束 / 新对话 | 切换 prompt |
| 优化收益 | 避免 O(N²) 重算 → O(N) | 避免重复 text encoder forward（数十次 → 1 次） |

---

## 6. 与下游模块的接口约定

### PromptEmbeddings 结构

```python
@dataclass
class PromptEmbeddings:
    cond: Optional[torch.Tensor] = None   # (B, L, D) 条件序列 embedding
    uncond: Optional[torch.Tensor] = None # (B, L, D) 无条件序列 embedding
```

下游模块（pipeline、denoiser）的约定：
- `cond` 和 `uncond` 同时不为 None（CFG 需要）
- 若 `cfg_scale = 1.0`，`uncond` 可以为 None
- shape 约定: `(B, L, D)` — batch, 序列长度, 隐藏维度
- dtype 和 device 与 denoiser 一致

### 与 DiT/TinyDiT 的对接

TinyDiT 的 `forward(x, t, text_tokens)` 中：
- `text_tokens` 的 shape: `(B, max_text_len, text_dim)` = `(1, 16, 64)`
- 这就是 `PromptEmbeddings.cond` 或 `.uncond` 的 tensor

TextConditioner 输出的 hidden_size 必须与 TinyDiT 的 text_dim 一致。当前 toy 约定为 64。
