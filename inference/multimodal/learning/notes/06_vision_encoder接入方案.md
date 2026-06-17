# Vision Encoder 接入方案

> 状态：Wave 3 / Task 8 完成（inputs_embeds 路径已正式接入 minivLLM 引擎）

## 1. 方案概览

本任务构建的是**最小教学型管线**，不真实接入引擎。目标是为 Task 8 准备好：
1. 图像预处理输出
2. Visual token embedding
3. 序列布局（position_ids / attention_mask）
4. 各模块的 shape 契约

## 2. 模块与文件

| 模块 | 文件 | 功能 |
|------|------|------|
| 样例图片生成 | `sample_images/demo.jpg` | 224×224 RGB 测试图 |
| 图像预处理 | `image_preprocess.py` | resize + normalize → tensor |
| Patch Embed | `patch_embed_demo.py` | Conv2d 模拟 ViT patch embed |
| Visual Token | `visual_token_demo.py` | 两种模式（tiny-vit-random / clip-reference） |
| 序列构造 | `mm_sequence_builder.py` | 两种布局，输出 position_ids / mask |

所有文件位于 `experiments/mm_token_pipeline/`。

## 3. 管线数据流

```
demo.jpg (224×224)
  │
  ▼
image_preprocess.py ───→ (3, 224, 224) float32 tensor (CLIP normalized)
  │
  ▼
patch_embed_demo.py ───→ Conv2d(3, 768, 16, 16) → (1, 196, 768)
  │
  ▼
visual_token_demo.py ──→ tiny-vit-random: (1, 197, 192)
                      ──→ clip-reference:  (1, 50, 768) [不下载权重]
  │
  ▼
mm_sequence_builder.py ─→ input_ids, position_ids, attention_mask, visual_span
```

## 4. Shape 契约汇总

### 4.1 预处理输出

| 参数 | 值 |
|------|---|
| shape | `(3, 224, 224)` |
| dtype | `float32` |
| normalization | CLIP mean/std |
| 值域 | `[-1.5, 2.2]` 左右（视图片内容） |

### 4.2 Patch Embedding

| 参数 | 224×224, patch=16 | 224×224, patch=32 |
|------|-------------------|-------------------|
| num_patches | 196 (14×14) | 49 (7×7) |
| embed_dim | 768 (ViT-Base) | 768 |
| 输出 shape | `(1, 196, 768)` | `(1, 49, 768)` |

### 4.3 Visual Token

| 模式 | 含 CLS？ | 输出 shape | 说明 |
|------|---------|-----------|------|
| tiny-vit-random | 是 | `(1, 197, 192)` | 2 层 Transformer, hidden=192 |
| clip-reference | 是 | `(1, 50, 768)` | ViT-B/32, hidden=768, 随机权重 |

### 4.4 序列布局

| 布局 | 示例（num_visual=256, num_text=16） | 总长度 |
|------|-------------------------------------|--------|
| bos_image_text | `[BOS][IMG_START] + 256 visual + [IMG_END] + 16 text` | 275 |
| placeholder_expanded | `8 text + 256 visual + 8 text` | 272 |

## 5. Task 8 实现结果 — inputs_embeds 路径已正式接入

`Qwen3Model.forward` 现在正式接受 `inputs_embeds` 参数（`minivLLM/minivllm/model/qwen3.py:176-203`）：

```python
def forward(self, input_ids=None, positions=None, kv_cache=None,
            is_prefill=True, inputs_embeds=None):
    # 双输入冲突检测
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("cannot accept both input_ids and inputs_embeds...")
    # 双空检测
    if inputs_embeds is None and input_ids is None:
        raise ValueError("requires at least one of input_ids or inputs_embeds")
    # 分支选择
    if inputs_embeds is not None:
        hidden_states = inputs_embeds  # 跳过 embed_tokens
    else:
        hidden_states = self.embed_tokens(input_ids)
    # 后续 transformer 不变
    for i, layer in enumerate(self.layers):
        hidden_states, residual = layer(positions, hidden_states, residual,
                                        kv_cache=kv_cache, layer_idx=i,
                                        is_prefill=is_prefill)
    ...
```

`Qwen3.forward` 同样透传 `inputs_embeds` 参数。

**测试结果**：
- `text_parity`：input_ids 路径 == inputs_embeds 路径，max|diff| = 0.00e+00 — ✅
- `invalid_dual_input`：双输入正确抛出 ValueError — ✅
- `HF parity`（Task 4 回归）：verdict=IDENTICAL — ✅

## 6. 注意事项

- **不下载 HF 权重**：`clip-reference` 模式用 `CLIPVisionConfig` 构造，随机初始化，HF 模型 id 仅用于读取配置结构
- **不同动 minivLLM**：Task 7 的所有代码位于 `experiments/mm_token_pipeline/`，不触碰 `minivLLM/` 下任何文件
- **Python 环境**：使用 `minivLLM/.venv/bin/python`（macOS 无 `python` 命令）
- **特殊 token ID**：BOS=151643, IMG_START=151655, IMG_END=151656, IMG_PLACEHOLDER=151654
- **position_ids 规则**：视觉 token 和文本 token 使用连续递增的 position，这对 RoPE 是正确的

## 7. 后续：从教学管线到真实接入

| Wave | 任务 | 内容 |
|------|------|------|
| Wave 3 / Task 8 | `inputs_embeds` 接入 ✅ | 已完成：`Qwen3Model.forward` 正式接受 `inputs_embeds`，text_parity 通过 |
| Wave 4 / Task 9 | Vision encoder 真实加载 | 加载 HF vision model 权重（如 Qwen3-VL visual tower） |
| Wave 5 / Task 11 | 端到端 VLM 推理 | image → visual tokens → LLM → text output |
