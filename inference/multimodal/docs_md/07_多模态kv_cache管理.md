# 07 · 多模态 KV Cache 管理

多模态请求的 cache key 必须包含视觉上下文。即使 text-only prefix 完全相同，只要图片不同，对应的 KV 就不能直接共享。本页给出原因，并据此说明 Task 11（mm kv cache management）的设计约束。

## 1. 纯文本 KV Cache 的简单性

纯文本推理中，KV cache 管理相对简单。每个 token 的 K/V 值仅由该 token 自身的 embedding 和直到该位置的序列上下文决定。如果两个请求共享相同的 prefix（如 system prompt），那么这两个请求的 attention 计算在 prefix 区域内完全一致，KV 值也完全相同，可以安全共享 prefix 的物理 block。

这个逻辑在 vLLM 的 PagedAttention 和 SGLang 的 RadixAttention 中都是成立的。Hash 匹配或 Radix Tree 匹配的本质是"token 序列相同 → KV 值相同 → 物理 block 可共享"。

## 2. 为什么多模态打破了这条规则？

在多模态推理中，文本 prefix 的 K/V 同时受前置文本 token 和 visual token 影响。Transformer self-attention 会读取当前位置之前的全部 token，因此：

> **不能只按文本复用**：两个请求即使有完全相同的文本 system prompt，如果 system prompt 前面的 visual token 不同（不同的图片），那么 system prompt 部分的 K/V 值也会不同。因为 K/V 计算中，attention 会混合 visual token 的信息。

具体来说：

- 请求 A: `[视觉 token 猫图] + [系统提示: "你是一个助手"]`
- 请求 B: `[视觉 token 狗图] + [系统提示: "你是一个助手"]`

请求 A 和 B 的系统提示文本完全相同，但因为 self-attention 会 attend 到前面的视觉 token（且猫图和狗图的视觉 token 完全不同），所以系统提示区域的 K/V 值在 A 和 B 中是不同的。这意味着不能简单地复用 A 的系统提示 KV cache 给 B 使用。

## 3. 多模态 Cache Key 的必然需求

为了解决上述问题，多模态 KV cache 需要引入**多模态感知的 cache key**。最直接的做法是将 image embedding 的 hash 作为 cache key 的一部分：

```
纯文本 cache key:
  cache_key = hash(text_token_ids[0:prefix_len])

多模态 cache key:
  cache_key = hash(image_embedding + text_token_ids[visual_end_pos:prefix_len])
```

只有当图像内容相同且文本 prefix 相同时，KV 才可以共享。如果图像不同，即使是相同的文本 prefix，也需要独立存储。

## 4. 正确的多模态序列布局

Self-attention 会混合前序状态，因此多模态 token 的排列顺序会直接决定哪些 KV 可以共享。Qwen-VL 的标准布局是：

```
[BOS] [系统文本...] [<vision_start>] [visual tokens...] [<vision_end>] [用户问题...] [<im_start>]assistant
```

为什么视觉 token 放在系统提示之后？因为系统提示通常是固定不变的最高频共享 prefix。如果把视觉 token 放在系统提示前面（如 `[visual] + [system] + [question]`），那么视觉 token 会污染系统提示的 K/V，使得不同图像下系统提示的 KV 无法共享。

一个优化策略是将视觉 token 尽量靠后放置。例如：

```
[系统提示] [<vision_start>] [visual tokens] [<vision_end>] [用户问题]
```

在这种布局下，视觉 token 之前的系统提示部分（纯文本、无视觉影响）的 KV 在所有请求中都是相同的。因为这部分 token 的 attention 范围只包含前面的文本 token 和自己，不包含任何视觉 token。这一段的 KV 仍然可以安全共享。

## 5. 分阶段 Prefix Cache 策略

基于以上分析，一个实用的多模态 KV cache 分阶段策略是：

| 阶段 | 内容 | 是否可共享？ | Cache Key |
|------|------|--------------|-----------|
| Stage 1 | 视觉 token 之前的纯文本 | 可共享 | hash(token_ids) |
| Stage 2 | 视觉 token 区域 | 不可共享（除非同图） | hash(image_embedding) |
| Stage 3 | 视觉 token 之后的文本 | 不可共享（被视觉 token 影响） | hash(image + text postfix) |

在 Stage 1 中，纯文本 prefix 可以安全共享。如果一个系统提示有 200 个 token，每天被 10000 个请求共享，这 200 个 token 的 prefill 只需要计算一次，之后所有请求复用。这样能省去其余请求对相同前缀的重复 prefill；实际吞吐收益取决于 batch 和显存余量。

## 6. 页面导航

- [← 文档首页](00_index.md)
- [ViT 与图像 Patch 编码](03_vit和图像patch.md)
- [CLIP 与图文对齐](04_clip和图文对齐.md)
- [Qwen-VL 多模态输入](05_qwen_vl多模态输入.md)
- [多模态 Prefill/Decode](06_多模态prefill_decode.md)
- [已有引擎审计](01_已有引擎审计.md)
- [Paged Attention 基础](02_paged_attention基础.md)
- [vLLM 多模态推理参考 →](08_vllm多模态推理参考.md)
- [SGLang 多模态推理参考](09_sglang多模态推理参考.md)

---

minivLLM 多模态推理实验工作区 · Wave 4 / Task 14 · 2026-06-07
