# 04 — Flamingo

## 一句话总结

通过在每个 LLM Transformer 层中插入额外的 gated cross-attention，将预训练的视觉编码器输出的视觉特征注入冻结的大语言模型，实现了图文交错的少样本多模态推理，全程不修改 LLM 和视觉编码器的原始权重。

## 关键 Idea

### 1. 冻结一切，只训练连接层

Flamingo 的核心理念：冻结预训练的 LLM（Chinchilla 70B）和冻结预训练的视觉编码器（NFNet-F6），只在两者之间插入可训练的连接模块。这样做的好处是保留 LLM 的文本能力和视觉编码器的视觉能力，不会因为多模态训练而灾难性遗忘。

### 2. Perceiver Resampler（视觉特征重采样）

视觉编码器输出的 visual features 数量随图像分辨率变化（可能数百到数千个）。Perceiver Resampler 是一个小型 Transformer，用固定数量的可学习 latent queries 通过 cross-attention 从可变数量的视觉特征中提取出固定长度的视觉表示。这样无论图像多大，给 LLM 的 visual token 数都是固定的。

### 3. GATED XATTN-DENSE（门控交叉注意力层）

在 LLM 的每个 Transformer 层中，原有的 self-attention 和 FFN 之间插入一个新的交叉注意力层。这个层以 LLM 的 hidden state 作为 query，以 Perceiver Resampler 输出的视觉特征作为 key 和 value。额外插入一个 Tanh gating 机制：

```
h_new = h + tanh(α) × CrossAttn(h, visual_features)
```

初始化时 α=0，交叉注意力模块输出被完全抑制，确保训练早期模型行为等同于原始纯文本 LLM，随后逐步"打开"视觉通路。

### 4. 图文交错处理

Flamingo 支持任意顺序的图文交错序列（如：图像1 → 文字1 → 图像2 → 文字2），视觉 token 只在其对应位置影响后续文本生成，而不会被所有 token 看到。这种设计使其非常适合多轮视觉对话和少样本示范学习。

## 与本项目的关联

Flamingo 的 gated cross-attention 是"冻结 LLM 加视觉"路线的经典方案，对本项目在 12GB 显存受限条件下的架构选择有参考价值——如果我们需要避免全量训练，可以考虑类似的冻结 + 连接器策略。Perceiver Resampler 的固定 token 数量压缩思路也与 Q-Former（BLIP-2）和 Qwen-VL VL Adapter 一脉相承。
