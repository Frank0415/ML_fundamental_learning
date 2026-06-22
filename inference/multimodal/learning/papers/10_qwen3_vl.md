# 10 — Qwen3-VL: 效率优化与密集小模型

## 一句话总结

Qwen3-VL 在保持 Qwen2.5-VL 同级别多模态能力的前提下，通过 DeepStack 架构创新、密集（dense）小模型设计（2B/4B/8B）和 Interleaved-MRoPE 位置编码优化，大幅提升了推理效率，是 受限显存设备上运行 VLM 的理想选择。

## 关键 Idea

### 1. Dense 小模型策略

Qwen3-VL 的核心发布策略是提供多个密集（非 MoE）小模型尺寸：2B、4B、8B。与之前 Qwen2.5-VL 的最小 3B 模型不同，Qwen3-VL 专门为边缘设备和低显存场景做了优化：

- Qwen3-VL-4B 在 4B 参数量级做到接近 Qwen2.5-VL-7B 的多模态能力，这是参数的 "2 倍压缩比"。
- Qwen3-VL-2B 进一步缩小，适合 <8GB 显存的设备。
- 所有模型均为 dense（非 MoE），前向推理架构简单，不需要 MoE routing 的额外开销。

### 2. DeepStack：深度高效架构

Qwen3-VL 引入了 DeepStack 架构创新：

- 在保持总参数量不变的前提下，增加层数、减少每层的宽度（hidden dim）。更深的网络（更多层、更窄）比宽浅网络在相同参数量下有更好的表达能力。
- 但更深的网络意味着更大的 KV cache（每层都存 K/V），Qwen3-VL 通过减少 kv_heads 数量（从 Qwen2.5-VL 的 8 降到 4）来抵消层数增加带来的 KV 开销。
- Qwen3-VL-4B 约 36 层、hidden=2560、GQA kv_heads=4、head_dim=128。这个配置下每 token 的 KV 约 72KB（bf16），在 中档显存卡上 2048 序列长度仅占约 150MB。

### 3. Interleaved-MRoPE

Qwen3-VL 保留了 Qwen2-VL/Qwen2.5-VL 的 M-RoPE 框架，但做了两个关键优化：

- **Interleaved 分配**：M-RoPE 的三个维度（t/h/w）不再简单等分 head_dim，而是按交错（interleaved）方式分配频率段。这比简单等分在各个 spatial dimension 上的区分度更好。
- **长上下文支持**：Qwen3-VL 的原生上下文窗口达到 128K token，通过 YaRN（Yet another RoPE extensioN）进一步外推到 256K。这意味着可以处理长视频（多帧密集采样）或大量图片跨页文档。

### 4. 多图与视频

Qwen3-VL 的多图和视频处理与 Qwen2-VL/Qwen2.5-VL 一致，基于 M-RoPE 的 temporal 维度：

- 多图：每张图像分配一个独立的时间步，所有图片的 visual token 沿 t 维度合并到同一序列。
- 视频：视频帧均匀采样（如每秒 1 帧），每帧生成 visual token，沿 t 维顺序排列。
- 显存约束：每增加一张图像/一帧视频，visual token 数量翻倍。中等显存设备上多图/视频的总 visual token 预算需要严格评估。

## 与本项目的关联

Qwen3-VL-4B-Instruct 是本项目 minivLLM 加载的**主路径权重**。理解其 DeepStack 层数/kv_heads 配置对于 KV cache 显存预算的精确计算至关重要；理解 Interleaved-MRoPE 的三维位置编码风格对于正确分配 visual token 的 position_ids 是必需的。Qwen3-VL 的 128K 长上下文和轻量化设计完美匹配我们的 受限显存目标。
