# 02 — An Image is Worth 16x16 Words (ViT)

## 一句话总结

将图像切割为固定大小的 patch，每个 patch 展平后视为一个"视觉单词 token"，直接送入标准 Transformer 编码器——不需要任何 CNN 前置处理。

## 关键 Idea

### 1. Patch Embedding（图像切块嵌入）

输入图像 H×W×C 被切分为 N 个 P×P 的 patch（P=16 是默认值）：

```
N = (H/P) × (W/P)
```

每个 patch 展平为一个 P²×C 的向量，再通过一个可学习的线性投影矩阵 E 映射到 d_model 维度。这块线性投影等价于一个 stride=P、kernel_size=P 的卷积。

### 2. [CLS] Token 与 Position Embedding

在 patch 序列最前面添加一个可学习的 [class] token，其最终编码器的输出用于分类。所有 patch embedding 加上可学习的 1-D position embedding 后送入 Transformer encoder。

### 3. 纯 Transformer 编码器

标准的 Transformer encoder（多层的 multi-head self-attention + MLP），不加任何 CNN 归纳偏置（inductive bias），完全靠数据和注意力学习空间关系。前提条件：足够大的数据集（JFT-300M、ImageNet-21k）预训练。

### 4. 从图像分类到视觉骨干

ViT 最原始用于分类，但 patch embedding + Transformer 的组合后来成为几乎所有 VLM 的视觉编码器基础架构。CLIP 的视觉分支、Qwen-VL 的视觉编码器都从 ViT 架构出发演进。

## 与本项目的关联

本项目最终需要一个视觉编码器来处理输入图像并产出 visual token。ViT 是所有 VLM 视觉编码器的源头——理解 patch 切分的粒度、embedding 的投影方式、position embedding 的位置注入，是后续理解 CLIP ViT 和 Qwen-VL ViT-bigG 的前提。当我们控制 `max_pixels` 以适配 12GB 显存时，本质上是在控制 patch 数量（= visual token 数量）。
