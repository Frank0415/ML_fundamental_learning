# 03 · ViT 与图像 Patch 编码

Vision Transformer（ViT）是几乎所有现代 VLM 视觉编码器的架构基础。它把一幅图像看作一个"序列"，用和文本 Transformer 几乎一样的机制来处理图像，只是输入从单词 token 变成了图像的 patch token。

## 1. 图像如何变成 token？

ViT 将图像切成固定大小的小方块（patch），把每个 patch 展开为一维向量，再用线性投影（patch embedding）映射到 Transformer 的 hidden dimension。这个过程可以写成卷积：`Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)`。卷积输出展平后得到形状为 `(num_patches, hidden_dim)` 的序列。

例如对于一幅 224×224 的 RGB 图像，当 patch 大小为 16×16 时：总共切出 \(14\times 14=196\) 个 patch，每个 patch 是 \(16\times 16\times 3=768\) 个像素值。patch embedding 将这 768 个值投影到 hidden_dim（如 768 或 1280），得到 196 个 token。

对于一幅 448×448 的图像，patch 大小仍为 16×16：总共 \(28\times 28=784\) 个 patch，visual token 数增加到 784 个。token 数量随分辨率平方增长，这也是为什么 受限显存设备必须严格控制 `max_pixels`。

<figure style="margin: 1rem 0 1.5rem;">
  <img src="https://raw.githubusercontent.com/Frank0415/Research/main/papers-source/multimodal/docs/assets/architecture/vit_model_overview.png" alt="ViT 模型总览图" style="width: 100%; border: 1px solid #d0d0d0; border-radius: 8px; background: #fff;" />
  <figcaption style="margin-top: 0.6rem; color: #555; font-size: 0.95rem;">
    来源：ViT 论文 Figure 1。原图把 patchify、线性投影、位置编码和 Transformer Encoder 主干一次性串起来，正好对应本节的输入管线说明。
  </figcaption>
</figure>

## 2. Patch 切分的直观表示

下面是一个 224×224 图像被 16×16 patch 切分的 ASCII 表示。每个编号代表一个 patch：

```
  0    1    2    3    4    5    6    7    8    9   10   11   12   13
 14   15   16   17   18   19   20   21   22   23   24   25   26   27
 28   29   30   31   32   33   34   35   36   37   38   39   40   41
 42   43   44   45   46   47   48   49   50   51   52   53   54   55
 56   57   58   59   60   61   62   63   64   65   66   67   68   69
 70   71   72   73   74   75   76   77   78   79   80   81   82   83
 84   85   86   87   88   89   90   91   92   93   94   95   96   97
 98   99  100  101  102  103  104  105  106  107  108  109  110  111
112  113  114  115  116  117  118  119  120  121  122  123  124  125
126  127  128  129  130  131  132  133  134  135  136  137  138  139
140  141  142  143  144  145  146  147  148  149  150  151  152  153
154  155  156  157  158  159  160  161  162  163  164  165  166  167
168  169  170  171  172  173  174  175  176  177  178  179  180  181
182  183  184  185  186  187  188  189  190  191  192  193  194  195

224×224 图像，patch_size=16 → 14×14 = 196 个 patch。
每个 patch 通过线性投影变成 hidden_dim 维向量，就是 196 个 visual token。
```

## 3. 位置编码：Patch 的空间顺序

文本 token 有天然的顺序（第一个词、第二个词...），但图像的 patch 是二维排列的。Transformer 本身不感知输入的位置，因此需要位置编码告诉模型"这个 patch 在图像的左上角，那个 patch 在右下角"。

ViT 使用可学习的位置嵌入（learned position embedding）：为每个 patch 位置预分配一个向量，与 patch embedding 相加。1D 的文本位置编码用 (t)，2D 的图像位置编码用 (h, w)，但 ViT 的做法是将 2D 坐标 flatten 成 1D 序列号，再用标准的 1D 位置嵌入。更现代的方案（如 Qwen-VL 的 M-RoPE）用显式三维编码同时表示高度和宽度。

还有一个特殊 token：**[CLS] token**。ViT 借鉴 BERT，在 patch 序列最前面加一个可学习的 [CLS] token。Transformer 编码后，[CLS] token 的最终 hidden state 被用作整个图像的全局表示（用于分类）。不过在现代 VLM 中，[CLS] token 往往不被使用，visual token 直接全部送入下游。

## 4. Qwen-VL 中的 ViT

Qwen-VL 系列使用的视觉编码器是 ViT-bigG（约 1.9B 参数），从 CLIP 预训练初始化。对于 Qwen3-VL，视觉编码器做了进一步的轻量化。基本参数如下：

| 参数 | ViT-bigG（Qwen-VL） | Qwen3-VL 视觉编码器 |
|------|---------------------|---------------------|
| hidden_dim | 1664 | 1280（更轻量） |
| 层数 | 48 | 约 32 |
| patch_size | 14 | 14 |
| 参数量 | 约 1.9B | 约 0.4B |

## 5. 页面导航

- [← 文档首页](00_index.md)
- [已有引擎审计](01_已有引擎审计.md)
- [Paged Attention 基础](02_paged_attention基础.md)
- [CLIP 与图文对齐 →](04_clip和图文对齐.md)
- [Qwen-VL 多模态输入](05_qwen_vl多模态输入.md)
- [多模态 Prefill/Decode](06_多模态prefill_decode.md)
- [多模态 KV Cache 管理](07_多模态kv_cache管理.md)
- [vLLM 多模态推理参考](08_vllm多模态推理参考.md)
- [SGLang 多模态推理参考](09_sglang多模态推理参考.md)

---

minivLLM 多模态推理实验工作区 · Wave 4 / Task 14 · 2026-06-07
