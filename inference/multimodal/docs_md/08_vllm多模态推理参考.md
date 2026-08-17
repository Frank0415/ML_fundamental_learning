# 08 · vLLM 多模态推理参考

本项目用 vLLM 的 PagedAttention 设计校对 minivLLM 的 KV cache 管理。vLLM 从 v0.5.0 起逐步加入多模态推理支持，覆盖 LLaVA、Qwen-VL 和 InternVL 等 VLM。

## 1. vLLM 的多模态架构

vLLM 的多模态实现包含这些设计：

- **复用 PagedAttention**：VLM 的 KV cache 管理沿用与纯文本 LLM 完全相同的 block-based 机制。vLLM 没有为 VLM 单独设计一套 KV cache，而是将 visual token 视为特殊的 prompt token 写入相同的 block table。
- **Automatic Prefix Caching（APC）**：vLLM v0.6.0 引入自动前缀缓存，并支持多模态 prefix 共享。当两个请求的 system prompt 和视觉 token 之前的部分完全相同时，APC 自动复用 prefix KV。
- **多模态输入处理器**：vLLM 为每种 VLM 提供独立的 `MultiModalProcessor`，负责将图像/视频预处理后转换为 visual token，并嵌入到文本 token 序列中。这个过程对 KV cache 层是透明的。

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/multimodal/docs/assets/architecture/vllm_system_overview.png" alt="vLLM 系统总览图" style="width: 100%; max-width: 720px; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：vLLM PagedAttention 论文 Figure 4。Scheduler、KV Cache Manager 与多个 worker 的职责分工，正好对应多模态 serving 的整体运行时结构。
  </figcaption>
</figure>

## 2. vLLM 多模态推理流水线

```
用户请求: "描述这张图片" + image.jpg
  │
  ▼
MultiModalProcessor:
  - 图像 load + resize（按 max_pixels / min_pixels 约束）
  - 调用模型的 visual tower（HF processor）产出 visual embeddings
  - 将 visual embeddings 插入到 prompt token 序列的 <image> 位置
  │
  ▼
Scheduler:
  - 预处理后的 token 序列作为正常 prompt 进入调度队列
  - can_allocate() 检查 visual token + text token 的总 block 需求
  - 分配 block table（与纯文本相同逻辑）
  │
  ▼
ModelRunner:
  - prefill: visual token + text token 一起做前向，KV 写入 block
  - decode: 追加生成的文本 token 到 block table
  │
  ▼
Output Processor:
  - 解码生成 token → 文本输出
```

## 3. PagedAttention 在 VLM 上的应用

vLLM 的 PagedAttention 在 VLM 场景下无需修改，因为：

- visual token 和 text token 共享相同的 KV cache 维度。K/V 的形状是 `[block_size, num_kv_heads, head_dim]`，与 token 是"视觉"还是"文本"无关。
- Block table 的语义只是"逻辑位置 → 物理 block 的映射"，不需要区分 token 类型。
- 从 attention 计算的角度看，visual token K/V 和 text token K/V 没有区别，都是参与 \(QK^{\mathsf{T}}\) 计算的向量。

**visual token 的位置编码**需要单独处理。使用 M-RoPE 的模型（如 Qwen2-VL/Qwen3-VL）为 visual token 提供三维坐标 (t, h, w)，text token 仍使用一维位置。vLLM 通过 `AttentionMetadata` 的 `position_ids` 区分两者。

## 4. vLLM 的显存管理优势

在受限显存环境中，vLLM 的 PagedAttention 可以减少连续分配与碎片带来的浪费：

| 机制 | 对受限显存配置的价值 |
|------|----------------|
| Block 级分配 | 不需要为 visual token 预留完整的 max_seq_len 连续空间。visual token 在 prefill 时写入若干个 block，decode 阶段使用新 block。 |
| 碎片率 <4% | 若以一张中档显存卡为例，仅浪费约 480MB。在 contiguous 方案下碎片可能高达 5~6GB，直接阻塞推理。 |
| Automatic Prefix Caching | system prompt 的 KV 共享可以节省约 20~50MB 的重复 prefill 计算（取决于 prompt 长度），并减少 batch 服务中的同前缀计算。 |
| Swap Out / Recomputation | 当显存不足时，可将不活跃请求的 KV block swap 到 CPU 内存或直接释放（需要时重新 prefill 恢复）。 |

## 5. 我们这次实验中发现的 vLLM 兼容性问题

> **4 个 reference VLM 模型全部加载失败**
>
> 在我们使用 `run_qwen_vl_reference.py`（Task 10）做参考推理时，4 个候选模型（Qwen3-VL-4B、Qwen2.5-VL-3B、InternVL3.5-4B、SmolVLM2-2.2B）都因 `device_map="auto"` 依赖 `accelerate` 而加载失败。这是 vLLM/TGI 等推理引擎通常绕过的环境问题，它们自己管理设备映射，不依赖 HF transformers 的 `device_map="auto"`。
>
> 这次实验也暴露了**参考实现对环境的依赖**。本机是 macOS + MPS，没有 CUDA，基于 device_map 的高层加载路径会受阻。minivLLM 改用手动权重加载，绕开了这项依赖。

## 6. 页面导航

- [← 文档首页](00_index.md)
- [ViT 与图像 Patch 编码](03_vit和图像patch.md)
- [CLIP 与图文对齐](04_clip和图文对齐.md)
- [Qwen-VL 多模态输入](05_qwen_vl多模态输入.md)
- [多模态 Prefill/Decode](06_多模态prefill_decode.md)
- [多模态 KV Cache 管理](07_多模态kv_cache管理.md)
- [已有引擎审计](01_已有引擎审计.md)
- [Paged Attention 基础](02_paged_attention基础.md)
- [SGLang 多模态推理参考 →](09_sglang多模态推理参考.md)

---

minivLLM 多模态推理实验工作区 · Wave 4 / Task 14 · 2026-06-07
