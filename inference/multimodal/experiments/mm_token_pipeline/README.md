# 多模态 Token Pipeline

教学演示管线：图像 → visual tokens → 多模态序列构造。

## 脚本

| 脚本 | 功能 |
|------|------|
| `image_preprocess.py` | 图像预处理：resize + CLIP normalize → tensor |
| `patch_embed_demo.py` | ViT patch embedding 演示（Conv2d 模拟） |
| `visual_token_demo.py` | Visual token 生成（tiny-vit-random / clip-reference） |
| `mm_sequence_builder.py` | 多模态序列构造（两种布局） |

## 可复现命令

```bash
# 进入工作目录
cd inference/multimodal

# 使用 venv 中的 Python
PY=minivLLM/.venv/bin/python

# 1. 图像预处理
$PY experiments/mm_token_pipeline/image_preprocess.py \
    --image experiments/vlm_minimal_demo/sample_images/demo.jpg \
    --size 224 \
    --out experiments/mm_token_pipeline/results/preprocessed.pt

# 2. Patch Embedding 演示
$PY experiments/mm_token_pipeline/patch_embed_demo.py \
    --image experiments/vlm_minimal_demo/sample_images/demo.jpg \
    --size 224 --patch 16

# 3a. Visual Token — Tiny ViT 随机模拟
$PY experiments/mm_token_pipeline/visual_token_demo.py \
    --mode tiny-vit-random

# 3b. Visual Token — CLIP Reference 结构打印
$PY experiments/mm_token_pipeline/visual_token_demo.py \
    --mode clip-reference

# 4a. 序列构造 — bos_image_text 布局
$PY experiments/mm_token_pipeline/mm_sequence_builder.py \
    --layout bos_image_text --num-visual 256 --num-text 16

# 4b. 序列构造 — placeholder_expanded 布局
$PY experiments/mm_token_pipeline/mm_sequence_builder.py \
    --layout placeholder_expanded --num-visual 256 --num-text 16
```

## 依赖

- Python 3.10+
- PyTorch（已在 `minivLLM/.venv` 中）
- Pillow（`uv pip install pillow`）
- transformers（可选，仅 `clip-reference` 模式的结构打印需要）

## 输出

所有结果保存到 `experiments/mm_token_pipeline/results/`：
- `preprocessed.pt` — 预处理后的图像 tensor
- `sequence_bos_image_text.pt` / `.json` — bos_image_text 布局
- `sequence_placeholder_expanded.pt` / `.json` — placeholder_expanded 布局

## 示例图片

`experiments/vlm_minimal_demo/sample_images/demo.jpg` — 224×224 RGB JPEG（"Hello VLM" 文字 + 几何形状）
