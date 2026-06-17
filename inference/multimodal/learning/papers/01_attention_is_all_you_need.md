# 01 — Attention Is All You Need (Transformer)

## 一句话总结

用纯注意力机制（self-attention）替代循环网络（RNN/LSTM）的序列建模方式，实现了完全并行化的训练，并有效捕获长程依赖。

## 关键 Idea

### 1. Scaled Dot-Product Attention

给定 query Q、key K、value V，注意力计算为：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

缩放因子 √d_k 防止点积过大导致 softmax 梯度饱和。这是所有现代 Transformer 的核心计算单元。

### 2. Multi-Head Attention

将 Q/K/V 分别投影到 h 个低维子空间，独立计算注意力，再拼接结果：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

不同 head 关注不同的表示子空间（语法、语义、位置等）。

### 3. Causal Mask（Decoder Self-Attention）

解码器（自回归生成）的 self-attention 会 mask 掉当前位置之后的 token，确保预测第 t 个 token 时只能看到 1..t-1。实现方式是对 QK^T 矩阵的上三角区域加上 −∞，softmax 后权重归零。

### 4. Positional Encoding

Transformer 没有循环也没有卷积，自身不感知序列顺序。论文用正弦/余弦函数为每个位置生成固定编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

将 PE 加在 token embedding 上后再送入 encoder/decoder。

## 与本项目的关联

minivLLM 中的 `layers/numpy/attention.py → Attn` 模块直接实现了 scaled dot-product attention 和 multi-head 计算。RoPE（旋转位置编码）替代了原始论文的正弦/余弦 PE，但 causal mask 在自回归推理时依然关键。理解原始 Transformer 的完整前向流程是理解 KV cache、paged attention 以及后续多模态注意力融合的基础——所有 VLM 的 LLM 骨干都建立在这篇论文之上。
