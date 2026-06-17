#!/usr/bin/env python3
"""图像预处理脚本：读图 → resize → normalize → tensor。

用法：
    uv run python experiments/mm_token_pipeline/image_preprocess.py \\
        --image experiments/vlm_minimal_demo/sample_images/demo.jpg \\
        --size 224 \\
        --out experiments/mm_token_pipeline/results/preprocessed.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


CLIP_MEAN = [0.4815, 0.4578, 0.4082]  # CLIP 默认均值
CLIP_STD = [0.2686, 0.2613, 0.2758]   # CLIP 默认标准差


def load_and_resize(image_path: str, size: int) -> Image.Image:
    """读取图片并 resize 到 (size, size)，使用 BICUBIC 插值。"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return img


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image → (C, H, W) tensor 并做 CLIP normalize。"""
    arr = torch.from_numpy(np.array(img, dtype=np.float32))
    # (H, W, 3) → (3, H, W)
    arr = arr.reshape(img.size[1], img.size[0], 3)
    arr = arr.permute(2, 0, 1)  # (3, H, W)

    # 归一化到 [0, 1]
    arr = arr / 255.0

    # CLIP normalize
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=torch.float32).view(3, 1, 1)
    arr = (arr - mean) / std

    return arr


def main():
    parser = argparse.ArgumentParser(description="图像预处理：resize + normalize → tensor")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--size", type=int, required=True, choices=[224, 336],
                        help="目标尺寸 (224 或 336)")
    parser.add_argument("--out", default=None, help="输出 .pt 文件路径（可选）")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误：图片不存在 {args.image}", file=sys.stderr)
        sys.exit(1)

    # 读图 + resize
    img = load_and_resize(args.image, args.size)
    print(f"[读图] 路径={args.image}")
    print(f"[尺寸] 原始={Image.open(args.image).size}, 目标=({args.size},{args.size})")

    # → tensor + normalize
    tensor = image_to_tensor(img)
    print(f"[张量] shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(f"[统计] min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
          f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")

    # 保存
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save(tensor, args.out)
        print(f"[保存] → {args.out}")

    print("预处理完成 ✓")
    return tensor


if __name__ == "__main__":
    main()
