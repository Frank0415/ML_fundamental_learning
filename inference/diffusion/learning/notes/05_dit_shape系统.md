# DiT Shape 系统：从图像到 Tokens 再回来

> T11 学习笔记 — 记录 Diffusion Transformer 的核心 shape 变换逻辑  
> 最后更新：2026-06-07

---

## 1. 总览：一个 forward 中经历了什么

TinyDiT 的 forward 接收三个输入：

| 输入 | Shape | 含义 |
|------|-------|------|
| `x` | `(B, C, H, W)` | 噪声 latent（如 `(1, 4, 8, 8)`） |
| `t` | `(B,)` 或 `(B, 1)` | timestep，范围 `[0, 1]` |
| `text_tokens`（可选） | `(B, L, D_text)` | 文本条件 embedding |

输出：

| 输出 | Shape | 含义 |
|------|-------|------|
| `epsilon` | `(B, C, H, W)` | 预测噪声（与输入同 shape） |

---

## 2. Patchify：图像 → Token 序列

### 公式

```
x:  (B, C, H,     W    )
  → (B, C, H/p, p, W/p, p)      # 按 patch_size 分块
  → (B, H/p, W/p, C, p, p)      # 置换维度
  → (B, N, p*p*C)                # 展平 patches，N = (H/p) * (W/p)
```

### 示例

`(1, 4, 8, 8)` with `patch_size=2`:
- `N = (8/2) × (8/2) = 4 × 4 = 16`
- 每个 patch 展开为 `2 × 2 × 4 = 16` 维
- 结果：`(1, 16, 16)` → 若 `hidden_size=16`，则投影后为 `(1, 16, 16)`

### 关键约束

- `H` 和 `W` 必须能被 `patch_size` 整除
- 假设正方形 latent（`H==W`），unpatchify 才能恢复形状

### 与 ViT 的差别

| 维度 | ViT | DiT |
|------|-----|-----|
| 输入 | 真实像素 (3, H, W) | VAE latent (4, H/8, W/8) |
| patch 维度 | `p*p*3` | `p*p*4` (or `p*p*16`) |
| 位置编码 | 1D learnable | 2D sinusoidal（真实 DiT）/ learnable（toy） |

---

## 3. Timestep Embedding: `(B,)` → `(B, D_hidden)`

### 步骤

```
t: (B,) 或 (B, 1)
  → sinusoidal encoding: (B, D_hidden)
  → SiLU + Linear + SiLU + Linear: (B, D_hidden)
```

### 原理

Sinusoidal encoding 使用几何级数频率：

```
freqs[k] = exp(-log(10000) × 2k / D)
embedding[t, 2k]   = cos(t × freqs[k])
embedding[t, 2k+1] = sin(t × freqs[k])
```

time 值 `t` 映射为 `D_hidden` 维向量，使模型能区分不同噪声水平。

### 用途

这个 embedding 被送入每个 DiTBlock 的 `adaLN_modulation` MLP，生成 **6 组调制参数**：
- `shift_attn`, `scale_attn`, `gate_attn` → attention 子层
- `shift_ffn`, `scale_ffn`, `gate_ffn` → FFN 子层

每组都是 `(B, D_hidden)` 维向量。

---

## 4. Text Projection: `(B, L, D_text)` → `(B, L, D_hidden)`

```
text_tokens: (B, L, D_text)
  → nn.Linear(D_text, D_hidden): (B, L, D_hidden)
```

在 toy 实现中只是一个简单的线性投影。真实实现会经过 CLIP/T5/Gemma 等 text encoder，此处仅保留输入接口。

---

## 5. DiTBlock 内部 Shape 变换

### 输入

| 名称 | Shape | 来源 |
|------|-------|------|
| `x` | `(B, N, D)` | patch tokens + pos_embed |
| `t_emb` | `(B, D)` | timestep encoding + MLP |
| `text_tokens`（可选） | `(B, M, D)` | text projection |

### Attention 子层

```
x → LayerNorm → modulated = norm(x) * (1 + scale_attn) + shift_attn

if text_tokens 存在:
    [modulated || text_tokens] → JointAttention → [attn_out_img, attn_out_text]
    attn_out = attn_out_img
else:
    modulated → SelfAttention → attn_out

x = x + gate_attn * attn_out
```

### FFN 子层

```
x → LayerNorm → modulated = norm(x) * (1 + scale_ffn) + shift_ffn
  → Linear → GELU → Linear → ffn_out
x = x + gate_ffn * ffn_out
```

### 调制参数广播

调制参数 `shift`, `scale`, `gate` 都是 `(B, D)` → unsqueeze 为 `(B, 1, D)` → 与 `(B, N, D)` 的 token 序列广播。

---

## 6. DiTBlock 输入输出对照表

| 阶段 | 输入 Shape | 输出 Shape | 操作 |
|------|-----------|-----------|------|
| 1. Attn AdaLN | `x: (B,N,D)`, `t_emb: (B,D)` | `modulated: (B,N,D)` | norm + broadcast |
| 2. Attention | `modulated: (B,N,D)` [± text] | `attn_out: (B,N,D)` | self 或 joint |
| 3. Residual | `x, gate_attn, attn_out` | `x': (B,N,D)` | gate 调制残差 |
| 4. FFN AdaLN | `x': (B,N,D)`, `t_emb: (B,D)` | `modulated: (B,N,D)` | norm + broadcast |
| 5. FFN | `modulated: (B,N,D)` | `ffn_out: (B,N,D)` | Linear→GELU→Linear |
| 6. Residual | `x', gate_ffn, ffn_out` | `x'': (B,N,D)` | gate 调制残差 |

- Attention 的 K/V 维度：`(B, H, N+N_text, d)`  其中 `d = D/H`
- FFN 隐藏层：`mlp_hidden_dim = D * mlp_ratio`（默认 `= 4*D`）

---

## 7. Unpatchify: Token 序列 → 图像

```
tokens:  (B, N, p*p*C)
  → (B, grid, grid, C, p, p)       # 假设正方形，N = grid^2
  → (B, C, grid, p, grid, p)
  → (B, C, grid*p, grid*p) = (B, C, H, W)
```

**约束**：`N` 必须是完全平方数（toy 版本的简化假设）。

---

## 8. MMDiT Joint Attention 维度拼接示意

```
             B  N_img  D      B  N_text  D
               \      /         \      /
                \    /           \    /
             B  (N_img+N_text)  D
                     |
              JointAttention
                     |
             B  (N_img+N_text)  D
               /      \
              /        \
     B  N_img  D    B  N_text  D
```

**与真实 SD3 MMDiT 的区别**：
- SD3 是双流架构：image stream 和 text stream 各有独立的 MLP/Norm，通过 cross-attention 交换信息
- 我们的 toy 版本做简化：拼接 → unified attention → 拆分
- 两者关键差异是：toy 版本的 text tokens 未经 AdaLN 调制（真实版本有独立的 text AdaLN stream）

---

## 9. 与 minivLLM LLM Attention 的关键差异

| 维度 | minivLLM (LLM) | diffusion_engine (DiT) |
|------|----------------|------------------------|
| **Attention 类型** | GQA（组查询注意力） | 标准 MHA（每 head 独立 QKV） |
| **因果掩码** | causal mask（下三角） | 无 mask（全局 attend） |
| **KV Cache** | 存储历史 K/V | 无 cache（每步全刷新） |
| **位置编码** | 1D RoPE（旋转位置编码） | 2D sinusoidal（真实）/ learnable（toy） |
| **归一化** | 静态 RMSNorm | AdaLN（动态 scale/shift/gate） |
| **输入维度** | `(B, seq_len, D)`, seq_len 逐步增长 | `(B, N, D)`, N 固定不变 |
| **输出** | 逐 token 概率分布 | 完整 latent 预测（一步输出全部 N） |
| **复用价值** | 无（除 SiluAndMul 11 行） | — |

---

## 10. 完整 forward shape 流水线（总结）

```
Input:  x=(B,C,H,W), t=(B,), text=(B,L,D_text)
─────────────────────────────────────────────────────
patchify:      (B,C,H,W)     →  (B,N,P²C)
proj:          (B,N,P²C)     →  (B,N,D)       # 若 P²C≠D
pos_embed:     + (1,N,D)     →  (B,N,D)       # 加法注入
─────────────────────────────────────────────────────
t_emb:         (B,)          →  (B,D)         # sinusoidal
t_mlp:         (B,D)         →  (B,D)         # SiLU MLP
text_proj:     (B,L,D_text)  →  (B,L,D)       # Linear
─────────────────────────────────────────────────────
DiTBlock × depth:
  per block:   (B,N,D) + (B,D) + (B,L,D)?
                 ↓ adaLN + attention + gate+residual + adaLN + FFN + gate+residual
               (B,N,D)                       # shape 不变
─────────────────────────────────────────────────────
final_norm:    (B,N,D)       →  (B,N,D)
final_linear:  (B,N,D)       →  (B,N,P²C)
unpatchify:    (B,N,P²C)     →  (B,C,H,W)
─────────────────────────────────────────────────────
Output: epsilon=(B,C,H,W)  ← 与输入同 shape
```

---

## 11. Toy 简化清单（T18 总结时标注）

以下简化在本实现中有意保留，将在最终报告中显式标注：

1. **位置编码**：使用 `nn.Parameter` 可学习 embedding，而非原始 DiT 的 2D sinusoidal
2. **AdaLN-Zero**：未实现零初始化 gate（真实版本在训练开始时 gate→0，使模型等同恒等映射）
3. **MMDiT**：使用拼接式 joint attention 而非 SD3 的双流架构
4. **Patch 大小**：toy 版本仅测试 `patch_size=2`（代码中 `PatchEmbed` 支持变量 p，但模型假设 `hidden_size` 匹配）
5. **预测目标**：epsilon prediction 而非 v-prediction（两者均正确，区别在于训练时的噪声调度）
6. **Latent 假设**：正方形（`H==W`），非正方形 latent 会导致 unpatchify 失败

---

## 12. 实现验证结果（T11 产出）

- [x] `attention.py`：SelfAttention + JointAttention 实现并测试
- [x] `transformer_block.py`：DiTBlock AdaLN 调制实现
- [x] `dit.py`：TinyDiT 完整 forward（patchify → blocks → unpatchify）
- [x] `test_dit_shapes.py`：覆盖 shape / NaN / 不同 param 组合
- [x] 无 minivLLM 依赖
- [ ] torch 环境状态：**未安装**（Python 3.9.6, torch 缺失），测试代码已用 `pytest.mark.skipif` 包裹
