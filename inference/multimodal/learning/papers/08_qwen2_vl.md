# 08 — Qwen2-VL: 动态分辨率与多维位置编码

## 一句话总结

Qwen2-VL 是 Qwen-VL 系列的重要跃迁，引入动态分辨率（以任意分辨率读取图像原文）和 M-RoPE（三维旋转位置编码）两大创新，使模型能够看清高分辨率图像的细节并精确感知 visual token 的二维空间位置。

## 关键 Idea

### 1. 动态分辨率：不再"缩略图"阅读

Qwen-VL 初代将任意图像 resize 到固定尺寸后切 patch，这导致高分辨率图像（如文档、表格、PPT 截图）中的小字完全不可读。Qwen2-VL 的核心改进是**根据图像原始长宽比和分辨率动态决定切分方式**。

具体做法：大图被切为若干子图（tile），每个子图独立经过 ViT 编码产生 visual token，最后将所有子图的 visual token 合并为一个序列。模型实际"看到"的是所有子图 patch 的完整集合，而不是一个被压缩到低分辨率的缩略图。

这意味着系统 prompt 里的 `min_pixels` / `max_pixels` 参数直接控制 visual token 预算：`max_pixels` 越大，子图越多，visual token 也越多，显存消耗越大。12GB 设备上必须把 `max_pixels` 控制在 512×512 以内。

### 2. M-RoPE：三维位置编码

RoPE（Rotary Position Embedding）原本只是一维位置编码，编码 token 沿文本序列的位置。Qwen2-VL 将其扩展为 **M-RoPE（Multimodal RoPE）**——三维位置编码：

- **时间维度（temporal）**：文本序列中的 token 位置索引。
- **高度维度（height）**：图像 token 在原图中的 Y 坐标。
- **宽度维度（width）**：图像 token 在原图中的 X 坐标。

每个 visual token 携带一个 (t, h, w) 三元组。模型在计算 attention 时，不同维度的 RoPE 旋转被分别施加到 query/key 的不同频率段上（即 head_dim 被等分给 t/h/w 三个维度）。这让模型在 attention 计算中显式感知 visual token 的空间位置关系。

文本 token 的 h 和 w 维度使用占位值（通常为零），保持与 visual token 的兼容。

### 3. 视频理解的自然延伸

M-RoPE 的 t 维度天然支持视频：视频的每一帧是一幅图像，不同帧沿 t 维度递增。Qwen2-VL 可以无缝处理多帧视频输入，不需要额外的架构调整。

## 与本项目的关联

Qwen3-VL 继承了 Qwen2-VL 的动态分辨率框架和 M-RoPE 位置编码。理解 `image_grid_thw`（image grid in temporal/height/width）和 visual token 预算的计算方法，是实现多模态 prefill 中对 visual token 正确分配 position_ids 和 KV cache 空间的前提。动态分辨率也直接决定了我们对 12GB 显存的 `max_pixels` 约束是否安全。
