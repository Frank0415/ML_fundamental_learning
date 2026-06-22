# Week 3 — Wave 3 进展报告

> 日期: 2026-06-07
> 关联任务: Task 7 — 多模态 Token Pipeline（教学管线）

---

## Task 7: 多模态 Token Pipeline 教学管线

### 目的

在将视觉编码器（vision encoder）正式接入 minivLLM 引擎之前，先构建一套完整的**教学型 token pipeline**，验证图像到 visual token 的完整数据流在数据形状（shape）和语义上是正确的。这个 pipeline 的所有代码位于 `experiments/mm_token_pipeline/` 下，不修改 `minivLLM/` 引擎一行代码。

### 实现

**5 个管线模块**：

| 模块 | 文件 | 功能 |
|------|------|------|
| 样例图片生成 | `sample_images/demo.jpg` | 224×224 RGB 测试图 |
| 图像预处理 | `image_preprocess.py` | resize + CLIP normalize → tensor |
| Patch Embed | `patch_embed_demo.py` | Conv2d(3,768,16,16) 模拟 ViT patch embed |
| Visual Token | `visual_token_demo.py` | tiny-vit-random (197×192) / clip-reference (50×768) |
| 序列构造 | `mm_sequence_builder.py` | 两种布局（bos_image_text / placeholder_expanded）→ input_ids + position_ids + mask |

**两种 visual token 模式**：
- **tiny-vit-random**：2 层随机 Transformer，hidden=192，产生 197 个 token（含 CLS）。用于快速验证管线 shape。
- **clip-reference**：使用 HF `CLIPVisionConfig` 构造 ViT-B/32，hidden=768，随机权重（不下载 HF 权重，仅读取配置结构），产 50 个 token。用于验证与真实 ViT 配置一致的 shape。

**两种序列布局**：
- **bos_image_text**：`[BOS][IMG_START] + 256 visual + [IMG_END] + 16 text` → 总长度 275。
- **placeholder_expanded**：`8 text + 256 visual + 8 text` → 总长度 272。模拟 `image_pad` token 被替换为 visual token 的展开过程。

### Shape 契约验证

所有模块的输入输出 shape 已在输出中明确打印，满足以下契约：

| 阶段 | 输入 shape | 输出 shape |
|------|-----------|------------|
| 预处理 | `(H, W, 3) uint8` | `(3, 224, 224) float32` |
| Patch Embed | `(1, 3, 224, 224)` | `(1, 196, 768)` |
| Visual Token (tiny) | `(1, 196, 768)` | `(1, 197, 192)` |
| Visual Token (clip) | `(1, 49, 768)` | `(1, 50, 768)` |
| 序列构造 | `visual_token + text_token` | `(1, total_len)` |

### 关键发现

1. **position_ids 规则**：视觉 token 和文本 token 使用连续递增的 position，这对 RoPE 是正确的。visual token 的二维空间位置（h, w）在 M-RoPE 中处理，不在 position_ids 中显式体现。
2. **特殊 token ID 映射**：BOS=151643, IMG_START=151655, IMG_END=151656, IMG_PLACEHOLDER=151654。这些 ID 在 `Qwen3-VL-4B-Instruct` 的 tokenizer 中定义。
3. **max_pixels 约束**：在受限显存配置下，`max_pixels` 应限制在 512×512 以内（约 1024 visual tokens）。超过此值时 visual token 的 KV cache 占用会挤压文本 token 的生成空间。
4. **clip-reference 模式无需下载权重**：通过 `CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")` 仅读取配置（不下载权重），用随机初始化的 Tensor 验证 shape，避免网络依赖。

### 与 Task 8 (inputs_embeds) 的衔接

Task 7 的教学管线产出的 visual token embedding（形状 `[1, num_visual, hidden_dim]`）可以直接作为 Task 8 的 `inputs_embeds` 参数传入 `Qwen3Model.forward()`。Task 8 的 `text_parity` 测试（input_ids 路径 == inputs_embeds 路径，max|diff|=0.00e+00）已经验证了输入 embedding 的拼接机制完全正确。

### 评测产物

**结果文件**（`experiments/mm_token_pipeline/results/`）：
- `pipeline_shape_contract.json` — 各阶段 shape 契约

**证据文件**（`.omo/evidence/`）：
- `task-7-pipeline.txt` — 5 模块全绿通过

### 状态

- [x] 5 个管线模块全部实现并运行通过
- [x] 两种 visual token 模式均可运行
- [x] 两种序列布局均可运行
- [x] Shape 契约验证通过
- [x] 证据文件写入
- [x] 未修改 minivLLM 引擎代码
- [ ] 真实 vision encoder 权重接入（留待 Task 9）
