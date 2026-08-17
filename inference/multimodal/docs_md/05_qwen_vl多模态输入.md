# 05 · Qwen-VL 多模态输入格式

理解 Qwen-VL 系列的输入格式是正确实现多模态 token pipeline 的前提。本文档覆盖 Qwen-VL、Qwen2-VL、Qwen2.5-VL 和 Qwen3-VL 四代模型的输入格式、特殊 token 用法、以及 visual token 的拼接规则。

## 1. 三段式架构概览

所有 Qwen-VL 系列共享相同的三段式架构流水线，只是各阶段的实现细节随版本演进：

```
输入图像
  │
  ▼
视觉编码器 (ViT)     ── Qwen-VL: ViT-bigG (14×14 patch)
  │                     Qwen2-VL/2.5-VL: 更大 ViT + 动态分辨率
  │                     Qwen3-VL: 轻量 ViT (~0.4B)
  │
  ▼
VL Adapter / Projector ── 将 visual token 投影到 LLM embedding 空间
  │                       输入维度: ViT hidden_dim (1664/1280)
  │                       输出维度: LLM hidden_dim (2560/3584/4096)
  │
  ▼
文本 Token 合并         ── visual token + text token → 拼接为一个序列
  │
  ▼
LLM (Qwen)             ── 生成回答
```

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/multimodal/docs/assets/architecture/qwen_vl_training_pipeline.png" alt="Qwen-VL 系列训练与架构流水线图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：Qwen-VL 论文 Figure 3。图中把 ViT、Cross-Attention Adapter 与 QwenLM 的三阶段训练关系画了出来，能直接对应本页的“三段式架构概览”。
  </figcaption>
</figure>

## 2. 特殊 Token 定义

Qwen-VL 系列使用以下特殊 token 来标记图像在文本序列中的位置：

| Token | Token ID | 含义 |
|-------|----------|------|
| `<\|vision_start\|>` | 151652 | 视觉 token 区域的开始标志 |
| `<\|vision_end\|>` | 151653 | 视觉 token 区域的结束标志 |
| `<\|image_pad\|>` | 151654 | 图像占位符（padding），在输入中被替换为视觉 token |
| `<\|video_start\|>` | 151655 | 视频 token 区域的开始 |
| `<\|video_end\|>` | 151656 | 视频 token 区域的结束 |
| `<\|im_start\|>` | 151644 | 对话轮次中的消息开始 |
| `<\|im_end\|>` | 151645 | 对话轮次中的消息结束 |

## 3. 输入序列的拼接规则

一个典型的多模态输入序列布局如下。其中 `<\|image_pad\|>` 在 tokenizer 阶段是单个文本 token，但在 forward 之前被替换为实际的 visual token embedding 向量序列：

```
[文本前缀 token ...]
<|vision_start|>
<|image_pad|> <|image_pad|> ... (共 N 个, N = num_visual_tokens)
<|vision_end|>
[文本后缀 token ...]
```

实际拼接时，`<\|image_pad\|>` 位置上放入 visual token 的 embedding，`<\|vision_start\|>` 和 `<\|vision_end\|>` 保持为正常的文本 token。因此 LLM 最终看到的输入是：

```
embed([text_prefix]) + embed([vision_start]) + visual_embeds + embed([vision_end]) + embed([text_suffix])
```

## 4. Visual Token 预算：image_grid_thw

Qwen2-VL 及之后的版本引入了 `image_grid_thw` 概念，这是一个三元组表示图像的 temporal/height/width 切分情况：

- **t（temporal）**：视频帧数。对于单张图像，t=1。
- **h（height grid）**：图像在高度方向上的 patch 数。如 224×224 图像、patch=14 时，h=16。
- **w（width grid）**：图像在宽度方向上的 patch 数。w=h=16 时，总 token = 256。

**动态分辨率下的 token 计算**：对于大图，Qwen2-VL 会将图像切为多个子图（tile），每个子图单独产生 visual token。总 token 数为 \(N_{\mathrm{visual}} = n_{\mathrm{tile}} \times h \times w\)。在受限显存配置下，`max_pixels=512×512` 大约产生 4 个子图，每个子图产生 \(16\times 16=256\) 个 visual token，总计约 1024 个 visual token。

## 5. 各版本输入格式的差异

| 版本 | 分辨率策略 | 位置编码 | Visual Token 投影方式 |
|------|------------|----------|----------------------|
| Qwen-VL | 固定 resize | 1D RoPE（合并到文本序列） | cross-attention VL Adapter → 256 个 token |
| Qwen2-VL | 动态分辨率 + 子图切分 | M-RoPE（3D: t, h, w） | MLP Projector → 可变的 visual token |
| Qwen2.5-VL | 更强的动态分辨率 | M-RoPE（同 Qwen2-VL） | 优化的 MLP Projector |
| Qwen3-VL | 继承动态分辨率 | Interleaved-MRoPE | 轻量 MLP Projector |

## 6. 页面导航

- [← 文档首页](00_index.md)
- [ViT 与图像 Patch 编码](03_vit和图像patch.md)
- [CLIP 与图文对齐](04_clip和图文对齐.md)
- [已有引擎审计](01_已有引擎审计.md)
- [Paged Attention 基础](02_paged_attention基础.md)
- [多模态 Prefill/Decode →](06_多模态prefill_decode.md)
- [多模态 KV Cache 管理](07_多模态kv_cache管理.md)
- [vLLM 多模态推理参考](08_vllm多模态推理参考.md)
- [SGLang 多模态推理参考](09_sglang多模态推理参考.md)

---

minivLLM 多模态推理实验工作区 · Wave 4 / Task 14 · 2026-06-07
