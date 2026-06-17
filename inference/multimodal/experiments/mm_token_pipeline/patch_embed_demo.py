#!/usr/bin/env python3
"""ViT Patch Embedding 演示：用 Conv2d 模拟图像切 patch 并嵌入。

用法：
    uv run python experiments/mm_token_pipeline/patch_embed_demo.py \\
        --image experiments/vlm_minimal_demo/sample_images/demo.jpg \\
        --size 224 --patch 16

说明：
    使用 Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    来模拟 ViT 的 patch embedding。输出 num_patches 与 embedding tensor 的 shape。
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


def patch_embed_demo(image_path: str, img_size: int, patch_size: int,
                     embed_dim: int = 768) -> dict:
    """对图片做 patch embedding，返回形状信息。

    Args:
        image_path: 图片路径
        img_size: 统一 resize 后的尺寸（正方形）
        patch_size: patch 大小（如 16 代表 16×16 的 patch）
        embed_dim: 嵌入维度（ViT-Base = 768, ViT-Large = 1024）

    Returns:
        dict: 包含 num_patches, embedding_shape 等信息
    """
    if img_size % patch_size != 0:
        raise ValueError(f"img_size ({img_size}) 必须能被 patch_size ({patch_size}) 整除")

    # 加载图片并 resize
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BICUBIC)

    # → tensor: (3, H, W), [0,1]
    arr = torch.from_numpy(np.array(img, dtype=np.float32))
    arr = arr.permute(2, 0, 1) / 255.0  # (H, W, 3) → (3, H, W)
    arr = arr.unsqueeze(0)  # → (1, 3, H, W)  batch=1

    # ViT patch embedding = Conv2d(3, embed_dim, patch_size, stride=patch_size)
    conv = torch.nn.Conv2d(
        in_channels=3,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size,
        bias=True,
    )

    with torch.no_grad():
        emb = conv(arr)  # (1, embed_dim, H/patch, W/patch)

    num_patches = (img_size // patch_size) * (img_size // patch_size)
    batch, dim, gh, gw = emb.shape

    # 常见 ViT 操作：flatten 空间维度 + transpose
    # (B, embed_dim, gh, gw) → (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
    emb_seq = emb.flatten(2).transpose(1, 2)  # (1, num_patches, embed_dim)

    return {
        "img_size": img_size,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "num_patches_h": img_size // patch_size,
        "num_patches_w": img_size // patch_size,
        "num_patches": num_patches,
        "conv_output_shape": tuple(emb.shape),
        "sequence_shape": tuple(emb_seq.shape),
    }


def main():
    parser = argparse.ArgumentParser(description="ViT Patch Embedding 演示")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--size", type=int, default=224, help="图片统一尺寸 (默认 224)")
    parser.add_argument("--patch", type=int, default=16, help="patch 大小 (默认 16)")
    parser.add_argument("--embed-dim", type=int, default=768,
                        help="嵌入维度 (默认 768, ViT-Base)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误：图片不存在 {args.image}", file=sys.stderr)
        sys.exit(1)

    result = patch_embed_demo(args.image, args.size, args.patch, args.embed_dim)

    print("=== Patch Embedding 演示 ===")
    print(f"图片尺寸: {result['img_size']}×{result['img_size']}")
    print(f"Patch 大小: {result['patch_size']}×{result['patch_size']}")
    print(f"嵌入维度: {result['embed_dim']}")
    print(f"Patch 网格: {result['num_patches_h']}×{result['num_patches_w']}")
    print(f"Patch 总数: {result['num_patches']}")
    print(f"Conv 输出 shape: {result['conv_output_shape']}  # (B, embed_dim, gh, gw)")
    print(f"序列 shape:     {result['sequence_shape']}  # (B, num_patches, embed_dim)")
    print("演示完成 ✓")


if __name__ == "__main__":
    main()
