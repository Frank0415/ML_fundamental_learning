# 03 — CLIP (Contrastive Language-Image Pre-training)

## 一句话总结

通过图文对比学习，将图像和文本分别编码后映射到同一向量空间，使匹配的图文对距离近、不匹配的距离远，从而实现零样本图像分类和图文检索。

## 关键 Idea

### 1. 双塔架构（Two-Tower Architecture）

CLIP 由两个独立的编码器组成：

- **Image Encoder**：ViT（Vision Transformer）或 ResNet，将图像映射为图像嵌入向量。
- **Text Encoder**：一个标准 Transformer，将文本描述映射为文本嵌入向量。

两个编码器的输出维度经线性投影对齐到同一 d 维空间。

### 2. 对比损失（InfoNCE / Contrastive Loss）

在一个 batch 的 N 个图文对中，图文对 (I_i, T_i) 是正样本，其余 N²−N 对是负样本。训练目标：最大化正样本对的余弦相似度，最小化负样本对的相似度。

```
L = (L_image→text + L_text→image) / 2
```

两条方向对称的 cross-entropy 损失，等价于一个 N 分类问题——在 N 个候选文本中找出正确的图像（反之亦然）。

### 3. 数据规模是关键

CLIP 的成功不来自精巧的架构创新，而来自 4 亿对图文数据的规模。自然语言监督比手工标注（如 ImageNet 标签）提供了丰富得多的语义信号，使得模型学习到了开放词汇的视觉概念。

### 4. 零样本推理

训练完成后，对任意新类别，只需构造"a photo of a {class_name}"的文本提示，图像编码与所有文本编码计算相似度，取最高者即为分类结果。不需要任何微调。

## 与本项目的关联

Qwen-VL 系列的视觉编码器是在 CLIP 预训练范式下训练的 ViT。理解 CLIP 的对齐机制，有助于理解后续 visual token 如何与 LLM 的 token embedding 空间桥接——无论用简单投影层（LLaVA）还是 Q-Former（BLIP-2），本质都是要把 CLIP 学到的视觉表示映射到 LLM 的文本 token 空间。
