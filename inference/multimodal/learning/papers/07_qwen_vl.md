# 07 — Qwen-VL 系列：从初代到 Qwen3-VL

## 一句话总结

Qwen-VL 系列从视觉编码器（ViT-bigG）+ 视觉-语言适配器（VL Adapter）+ 大语言模型（Qwen LLM）的三段式架构出发，经过 Qwen2-VL（动态分辨率 + M-RoPE）、Qwen2.5-VL（更强视觉感知）到 Qwen3-VL（效率优化），逐步演进为当前最强开源 VLM 家族之一。

## 关键 Idea

### 1. Qwen-VL 初代设计（2023）

- **视觉编码器**：ViT-bigG（约 1.9B 参数），从 CLIP 预训练初始化。
- **VL Adapter**：位置感知的视觉-语言适配器，使用 cross-attention 模块将可变数量的 ViT patch features 压缩为固定数量（256 个）的 visual token。不同于 LLaVA 的简单投影，VL Adapter 可以处理任意分辨率输入（resize 后直接切 patch 不限制固定数量）。
- **LLM**：Qwen-7B 基础语言模型。

关键特性：支持任意分辨率图像输入、支持 bounding box 定位与指称理解（referring/grounding）。

### 2. Qwen2-VL（2024）

引入两大改进：

- **动态分辨率**：不再将图像统一 resize 到固定尺寸。根据图像长宽比和分辨率动态决定切分方式，把大图切为若干子图分别编码，再合并 visual token。这让模型能看清高分辨率图像中的细节文字（OCR 场景）。
- **M-RoPE（Multimodal Rotary Position Embedding）**：将 RoPE 扩展为三维位置编码——时间维度（文本序列位置）、高度维度（图像 Y 坐标）、宽度维度（图像 X 坐标）。让模型显式感知视觉 token 的二维空间位置关系。

### 3. Qwen2.5-VL（2025）

- 进一步增强视觉感知能力：更强的视觉定位、更细粒度的目标检测和 OCR。
- 训练数据质量和规模的提升，尤其是在视频理解任务上。
- 更大的视觉编码器和更高效的 VL Adapter 设计。

### 4. Qwen3-VL（2026）

- 针对推理效率和部署场景优化，参数量更灵活（4B / 8B / 32B 等）。
- 视觉编码器更加轻量，推理速度提升，但保持同级别最强的多模态能力。
- 在 4B 参数级别做到了接近 Qwen2.5-VL-7B 的能力，是 受限显存设备上的理想选择。

## 与本项目的关联

minivLLM 加载的权重就是 Qwen3-VL-4B-Instruct。理解 Qwen-VL 系列的三段式架构是正确实现本项目多模态 token pipeline 的前提。具体而言：视觉编码器如何产出 visual token、VL Adapter 如何投影到 LLM 空间、visual token 如何与文本 token 拼接到同一序列——这三个步骤直接决定了我们前向推理的实现细节。Qwen3-VL 的轻量化设计正好匹配我们的 中等显存配置约束。
