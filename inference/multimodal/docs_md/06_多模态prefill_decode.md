# 06 · 多模态 Prefill 与 Decode

多模态推理的 prefill 和 decode 阶段与纯文本推理有本质区别。核心差异在于 visual token 是一次性写入 KV cache 的（prefill 阶段），之后的 decode 阶段只追加文本 token，不再产生新的 visual token。理解这个差异对于 KV cache 的显存规划和块分配策略至关重要。

## 1. 纯文本 Prefill/Decode 的回顾

在纯文本推理中，prefill 和 decode 是两种不同的前向模式：

| | Prefill | Decode |
|---|---------|--------|
| 输入长度 | N 个 token（prompt 全文） | 1 个 token（上次生成的） |
| Attention 方式 | 双向 causal mask（N × N） | 单向（1 × cached_seq_len） |
| KV Cache | 写入全部 N 个 token 的 K/V | 追加 1 个 token 的 K/V |
| 计算量 | O(N²)（但只做一次） | O(N)（每步很小） |
| 输出 | 最后一个 token 的 logits | 新 token 的 logits |

## 2. 多模态 Prefill 的独特之处

多模态 prefill 与纯文本 prefill 的主要差异在于输入的来源：

```
纯文本 Prefill:
  input = [token₀, token₁, ..., token_N]  # 全部来自 tokenizer
  → embed_tokens(input) → Transformer → logits

多模态 Prefill:
  input = [text_token₀, ..., <vision_start>, 
           visual_embed₀, visual_embed₁, ..., visual_embed_M,
           <vision_end>, text_token_K, ..., text_token_N]
  → 拼接 text embeddings + visual embeddings → Transformer → logits
```

注意这里 visual token 的 embedding 不经过 tokenizer 的 embedding 表（embed_tokens），而是直接由视觉编码器 + VL Adapter 产出。在 minivLLM 中，这正是 Task 8 实现的 `inputs_embeds` 路径：跳过 embed_tokens，以拼接后的 embedding 直接送入 Transformer。

## 3. Decode 阶段的一致性

多模态的 decode 阶段与纯文本的 decode 几乎完全相同。这是因为：

- visual token 是一次性写入的，在 prefill 阶段就全部进入了 KV cache。
- decode 阶段的每一步只生成一个文本 token，走正常的 embed_tokens → Transformer → logits → sample 流程。
- decode 时不需要再次计算 visual token——KV cache 已经保存了它们的历史 K/V。

> **关键原理**：Visual token 在 decode 阶段作为"已缓存的过去"存在。新生成的 token 做 self-attention 时，会 attend 到所有 visual token 和所有之前生成的文本 token。这个行为由 causal mask 自动保证：visual token 的 KV 已经在 cache 中，decode token 以 `is_causal=True` 的注意力看到它们。

## 4. 显存开销分析

多模态 prefill 的显存峰值发生在 prefill 阶段。此时需要同时容纳：

| 显存占用 | 说明 | 典型大小（12GB） |
|----------|------|------------------|
| 模型权重 | Qwen3-VL-4B bf16 | ~8GB |
| Visual token K/V（prefill） | 视觉 token 的 K/V 写入 | ~50MB（1024 visual tokens） |
| Text token K/V（prefill） | 文本 prefix + suffix 的 K/V | ~10MB（~100 text tokens） |
| Activation（prefill） | 前向中间激活 | ~500MB ~ 1GB |
| 剩余 | decode 阶段的 KV 扩展空间 | ~2GB ~ 2.5GB |

Decode 阶段的显存压力比 prefill 小得多：只需要单 token 的 activation，KV cache 的增量也很小（每步只追加 1 个 token 的 K/V，约 72KB）。因此多模态推理的瓶颈在 prefill，优化的关键是控制 visual token 的数量（限制 `max_pixels`）。

## 5. minivLLM 当前状态

> minivLLM 已经实现并通过了 `inputs_embeds` 路径的 text_parity 测试（Task 8）。纯文本 input_ids 路径与 inputs_embeds 路径的对齐结果：max\|diff\|=0.00e+00。这确认了输入的 embedding 拼接逻辑是正确的。
>
> 当前缺少的是视觉编码器的真实权重加载（需要在 Task 9 中接入 HF visual tower），以及多模态 KV cache 管理（Task 11）。在缺少真实视觉编码器的情况下，可以使用教学用的 dummy visual features 来模拟多模态 prefill。

## 6. 页面导航

- [← 文档首页](00_index.md)
- [ViT 与图像 Patch 编码](03_vit和图像patch.md)
- [CLIP 与图文对齐](04_clip和图文对齐.md)
- [Qwen-VL 多模态输入](05_qwen_vl多模态输入.md)
- [已有引擎审计](01_已有引擎审计.md)
- [Paged Attention 基础](02_paged_attention基础.md)
- [多模态 KV Cache 管理 →](07_多模态kv_cache管理.md)
- [vLLM 多模态推理参考](08_vllm多模态推理参考.md)
- [SGLang 多模态推理参考](09_sglang多模态推理参考.md)

---

minivLLM 多模态推理实验工作区 · Wave 4 / Task 14 · 2026-06-07
