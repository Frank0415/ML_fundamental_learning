# 05 — BLIP-2: Q-Former 桥接冻结视觉编码器与冻结 LLM

## 一句话总结

用 Q-Former（Querying Transformer）——一个轻量级的查询 Transformer——将冻结的 ViT 视觉编码器产出的海量视觉 token 压缩为少量固定数量的 query token，作为"软提示"直接输入冻结的大语言模型，全程不微调任何预训练组件。

## 关键 Idea

### 1. 三段式冻结架构

- **视觉编码器**：冻结的 ViT（CLIP 预训练或 EVA-CLIP）。
- **Q-Former**：一个可训练的轻量 Transformer（约 188M 参数），核心任务是将视觉 token 压缩为固定数量（默认 32 个）的 query embedding。
- **大语言模型**：冻结的 LLM（OPT 或 FlanT5），接收 Q-Former 产出的 query embedding 作为输入前缀。

### 2. Q-Former 的核心设计

Q-Former 包含两个共享 self-attention 的 Transformer 子模块：

- **图像 Transformer**：与冻结的 ViT 输出做 cross-attention，提取视觉信息。
- **文本 Transformer**：处理文本 token，同时与图像 Transformer 共享 self-attention 权重。

Q-Former 的输入是一组可学习的 query token（例如 32 个），它们不绑定任何具体像素，而是通过 cross-attention 从 ViT 输出中"抽取"最相关的视觉信息。产出 32 个 embedding 向量。

### 3. 两阶段训练

**第一阶段：vision-language 表示学习。** Q-Former 通过三种预训练任务（图文对比、图文匹配、图文生成）学习如何从视觉特征中提取与语言相关的表示，训练时 Q-Former 连接 ViT 但不连接 LLM。

**第二阶段：vision-to-language 生成学习。** 冻结 Q-Former，仅训练一个线性投影层将 Q-Former 的 query embedding 映射到 LLM 的 token embedding 空间。LLM 完全冻结。

### 4. 为什么需要 Q-Former

ViT 输出的 patch token 数量通常很大（如 256 或 576 个）。如果直接将这些 token 全部拼接到 LLM 的输入序列中，将急剧增加 KV cache 显存开销。Q-Former 将 256+ 个 token 压缩为 32 个，大幅降低了 LLM 的推理成本。这正是本项目在 12GB 显存约束下的核心优化思路。

## 与本项目的关联

本项目面临的核心显存瓶颈之一就是 visual token 过多。Q-Former 的压缩思路（大→小，用可学习 query 做信息蒸馏）与 Qwen-VL 的 VL Adapter 原理一致。理解 BLIP-2 的 token 压缩机制是理解如何在实际推理中控制 visual token 开销的关键。
