# 09 · SGLang 多模态推理参考

SGLang 用 RadixAttention 管理 prefix cache。它把 prompt 放进 Radix Tree，并按共享前缀复用 KV block；本页关注这套机制在 VLM 请求中的适用边界。

## 1. SGLang 的 RadixAttention 与 VLM

在纯文本推理中，SGLang 会把所有 prompt 插入 Radix Tree，后续请求自动匹配共享 prefix。VLM 请求仍然遇到和 vLLM hash-based caching 相同的问题：图片变化后，visual token 对应的 KV 也会变化。

SGLang 社区目前对多模态 prefix caching 的讨论集中在以下方向：

- **文本-视觉分离节点**：在 Radix Tree 中标记视觉 token 的边界，文本节点正常参与 prefix 匹配，视觉节点根据 image hash 单独匹配。
- **图像哈希作为节点标签**：每个视觉 token 节点附带一个 image_embedding 哈希值。只有哈希值匹配时，才复用该视觉节点及其后代的 KV cache。
- **截断匹配**：当一个请求的 sequence 到达视觉 token 边界时，自动截断 prefix 匹配，之后的部分视为全新 prefix 单独计算。

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/multimodal/docs/assets/architecture/sglang_radixattention_lru.png" alt="SGLang RadixAttention 与 LRU 淘汰示意图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：SGLang 论文 Figure 3。它展示了 RadixAttention 如何在多轮请求中复用、分叉和淘汰前缀节点，比抽象描述更容易看懂 prefix reuse 的动态过程。
  </figcaption>
</figure>

## 2. SGLang 多模态 Serving 的实践经验

当前的 VLM serving 仍有几处设计可以参考：

| 特性 | SGLang 的做法 | 对受限显存配置的价值 |
|------|---------------|----------------|
| Radix Tree 文本 prefix | 视觉 token 之前的纯文本部分依然可共享 | 节省系统提示的重复 prefill，约 20~50MB/请求 |
| Continuous Batching | 支持 prefill 和 decode 混合 batch | 提升 受限显存下的吞吐效率 |
| Token 级 Block 管理 | 与 vLLM 类似的 block 级分配 | 相同的碎片率优势 |
| Structured Output | JSON / Regex 约束解码 | 对文档解析 VLM 任务特别有用（如 OCR 结构化提取） |
| RadixAttention Skip | 对于已完全匹配的 prefix，跳过 attention 计算 | 显存和延迟双优化 |

## 3. SGLang vs vLLM 在受限显存 VLM 场景下的对比

| 维度 | vLLM | SGLang |
|------|------|--------|
| KV Cache 机制 | PagedAttention + hash-based APC | RadixAttention + Radix Tree |
| 多模态支持 | 已较为成熟（v0.5+） | 快速演进中，部分 VLM 已支持 |
| Prefix 共享粒度 | Block 级（16 token 对齐） | Token 级（精确到每个 token） |
| 显存分配 | 可配置（block size 可定制） | 可配置（使用相同的 block 管理） |
| 社区成熟度 | 更成熟，VLM 文档更完善 | 快速增长，部分 VLM 仍为实验性 |
| 适用场景 | 通用 VLM serving | 多轮对话 + 渐进式 prefix（如 agent 场景） |

## 4. SGLang 的架构对 minivLLM 的启发

SGLang 的以下几个设计思想对 minivLLM 的多模态扩展具有参考价值：

- **前缀树的自动性**：不需要显式配置哪些 prefix 共享。所有 prompt 自动参与共享。这个"零配置 prefix sharing"的思路很适合 minivLLM 这类教学型引擎，用户不需要关心 KV 共享的内部机制。
- **引用计数释放**：Radix Tree 节点的引用计数机制让 KV 缓存的生命周期管理变得清晰。当最后一个引用该 prefix 的请求完成，KV block 自动释放。vLLM 的 hash-based 方案没有这个内置的引用追踪。
- **Skip Attention**：完全命中的 prefix 可以跳过 attention 并直接复用结果。大 batch 时节省较多计算，小 batch 的收益较小。

## 5. 我们这次实验中的参考状态

> **SGLang 参考路径未执行**
>
> 在 Task 10 的 reference 矩阵实验中，我们仅测试了 HF transformers 的 `model.generate()` 路径，未安装和运行 SGLang runtime。原因有二：一是 4 个 VLM 模型在基础 HF 加载阶段就已全部失败（缺 `accelerate`），二是 SGLang 本身需要 CUDA 环境（`torch.cuda.is_available()`），macOS MPS 下无法运行。
>
> SGLang 的 RadixAttention 在当前 minivLLM 实现中作为设计参考存在，而非运行依赖。minivLLM 的 PagedKV（Task 6）采用 vLLM 风格的 correctness-first 实现，但 SGLang 的 Radix Tree 概念在 learning 笔记中有完整记录（`../learning/notes/10_sglang源码参考.md` + [SGLang RadixAttention 论文笔记](../learning/papers/12_sglang_radixattention.md)）。

## 6. 页面导航

- [← 文档首页](00_index.md)
- [ViT 与图像 Patch 编码](03_vit和图像patch.md)
- [CLIP 与图文对齐](04_clip和图文对齐.md)
- [Qwen-VL 多模态输入](05_qwen_vl多模态输入.md)
- [多模态 Prefill/Decode](06_多模态prefill_decode.md)
- [多模态 KV Cache 管理](07_多模态kv_cache管理.md)
- [vLLM 多模态推理参考](08_vllm多模态推理参考.md)
- [已有引擎审计](01_已有引擎审计.md)
- [Paged Attention 基础](02_paged_attention基础.md)

---

minivLLM 多模态推理实验工作区 · Wave 4 / Task 14 · 2026-06-07
