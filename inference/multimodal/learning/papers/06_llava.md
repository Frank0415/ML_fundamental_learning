# 06 — LLaVA: Visual Instruction Tuning（视觉指令微调）

## 一句话总结

用 CLIP vision encoder + 一个简单的线性投影层 + LLM 组成 VLM，通过 GPT-4 生成的高质量视觉指令微调数据来训练，证明最简洁的架构配合高质量数据就能达到强大多模态能力。

## 关键 Idea

### 1. 极简架构

LLaVA 是全部 VLM 中架构最简洁的方案：

```
图像 → CLIP ViT-L/14 → 线性投影 (W) → 视觉 token
文本 → 分词器 → 文本 token

视觉 token + 文本 token → 拼接 → LLM (Vicuna) → 生成回答
```

这里没有 Q-Former，没有 gated cross-attention，没有任何花哨的中间模块。CLIP 视觉编码器的输出经过一个可训练的线性层（projector）直接投影到 LLM 的 token embedding 空间，然后和文本 token 拼在一起送进 LLM。参数量最小，实现最简单。

### 2. 核心贡献：数据，不是架构

LLaVA 最重要的贡献不在模型架构，而在数据生成方法。它用 GPT-4/ChatGPT 对 COCO 图像生成三类指令数据：

- **对话**：多轮问答，模拟自然交互。
- **细节描述**：让模型描述图像中的细节。
- **复杂推理**：需要逻辑推理和常识的问题。

这些数据让 LLM 在保持文本能力的同时，学会了"看到"图像并基于图像推理。

### 3. 两阶段训练

**Stage 1：特征对齐预训练。** 仅训练投影层，冻结 CLIP 和 LLM。用 595K 图文对训练，目标是对齐视觉特征和文本 token 空间。

**Stage 2：端到端指令微调。** 解冻 LLM（CLIP 依然冻结），用 158K 指令数据微调。投影层和 LLM 同时更新。这一步让模型真正学会根据图像理解指令并生成回答。

### 4. 简单但有效

实验表明 LLaVA 在多个多模态 benchmark 上超过当时的 BLIP-2 和 Flamingo。关键启示：当 LLM 足够强大时，一个简单的投影就足以桥接视觉和语言；复杂架构（如 Q-Former）的价值在于 LLM 能力相对不足时，或者在极低显存场景中压缩 token。

## 与本项目的关联

LLaVA 路线（CLIP + Projector + LLM）是本项目最小 VLM 的首选架构，与 Qwen-VL 系列架构高度一致。我们不需要从零训练，但理解 LLaVA 的 token 拼接方式和两阶段训练，是理解 Qwen-VL 前向推理中 visual token 如何进入 LLM 的关键。12GB 显存约束下，LLaVA 的简洁性就是优势。
