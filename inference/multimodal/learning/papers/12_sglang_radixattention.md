# 12 — SGLang RadixAttention: 基于 Radix Tree 的前缀缓存

## 一句话总结

SGLang 提出 RadixAttention，用 Radix Tree（基数树）来管理和检索 KV cache 的前缀共享，相比 vLLM 的 hash-based caching 在层次化 prefix（如多轮对话中的渐进式前缀）场景下有更高的命中率，并支持自动前缀匹配而不需要手动指定共享范围。

## 关键 Idea

### 1. Radix Tree 作为 KV Cache 索引

Radix Tree（也叫 Patricia Trie）是一种压缩前缀树。SGLang 将被服务过的所有 prompt 的 token 序列作为 key，对应的 KV cache block 作为 value，组织成一个全局 Radix Tree：

```
           Root
          /    \
     "You are"  "Hello"
       /     \       \
  " a helpful" " an expert"  " world"
       |          |
   剩余 token  剩余 token
```

- 树的每条边是一个 token 子序列（压缩后）。
- 每个节点存储直到该节点的完整 KV cache block 列表。
- 新请求到达时，从 Root 开始逐 token 匹配，直到无法继续匹配。能匹配到的所有节点就是共享 prefix，其 KV cache 直接复用。

### 2. 自动前缀检测与共享

SGLang 的 key insight 是**不需要开发者手动指定哪些 prefix 共享**。所有到达系统的 prompt 自动在 Radix Tree 中记录。

例如多轮对话：
- Round 1 prompt: "System: You are a helpful assistant.\nUser: What is Python?"
- Round 2 prompt: "System: You are a helpful assistant.\nUser: What is C++?"

Round 2 的 prompt 在 "System: You are a helpful assistant.\nUser: What is " 处与 Round 1 完全匹配，这些 token 的 KV cache 自动复用，无需重新计算。

vLLM 的 hash-based caching 也能做到这一点，但 Radix Tree 的树形结构使得**部分匹配的复用**更加高效——即使两个 prompt 的 suffix 不同，共享的 prefix 部分仍然可以局部复用，不需要整个 block 完全一致。

### 3. 运行时架构

SGLang 的 runtime 包含以下核心组件：

- **Tokenizer Manager**：统一管理 tokenization，支持并行 tokenize。
- **Scheduler**：基于 Radix Tree 的请求调度，优先调度 prefix 命中率最高的请求。
- **Radix Cache**：维护 Radix Tree 和 KV cache 块的分配/释放。
- **Model Runner**：执行 prefill（仅计算 prefix 未命中部分）和 decode（逐 token 生成）。

显存管理同样采用类似 vLLM 的 block-based 分配，但 block 的生命周期由 Radix Tree 的节点引用计数决定：节点被引用的请求数为 0 时，其 KV block 被释放。

### 4. 与 vLLM PagedAttention 的对比

| 特性 | vLLM PagedAttention | SGLang RadixAttention |
|------|---------------------|-----------------------|
| KV 索引方式 | Hash table（block 级别） | Radix Tree（token 级别） |
| Prefix 共享粒度 | Block 级（≥block_size token） | Token 级（精确到每个 token） |
| 自动共享 | 需要显式配置同一 prefix | 自动，所有 prompt 隐式共享 |
| 部分匹配复用 | 有限（需 block 对齐） | 高效（树的分叉点即复用边界） |
| 实现复杂度 | 较低（hash 表） | 中等（树结构维护） |
| 适用场景 | 高度重复的 system prompt | 多轮对话、渐进式 prefix |

### 5. 多模态场景下的扩展

SGLang 的 Radix Tree 原生设计为文本 token 序列。在多模态场景下，visual token 的 prefix 共享面临额外挑战：

- 不同图像产生的 visual token 完全不同，Radix Tree 中的文本 prefix 到 visual token 的边界需要特殊处理。
- 如果两个请求共享相同的 system prompt + 相同的图像，则文本 prefix + visual token 都可以共享。
- SGLang 社区已有讨论多模态 prefix caching 的扩展方案，但尚未像纯文本那样开箱即用。

## 与本项目的关联

本项目的 KV cache 管理当前基于 vLLM 风格的 PagedAttention（block-based, correctness-first）。SGLang 的 RadixAttention 提供了更高级的 prefix 共享策略，尤其是在多轮对话中渐进式 prefix 共享非常高效。当 Task 11（多模态 KV cache 管理）完成后，Radix Tree 的思路可以借鉴到 visual token 的共享场景：如果两次推理使用完全相同的图像，visual token 的 KV 可以完全复用。
