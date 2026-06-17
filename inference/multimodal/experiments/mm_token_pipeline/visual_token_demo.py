#!/usr/bin/env python3
"""Visual Token 生成演示：两种模式。

用法：
    # 模式 1: 随机模拟 ViT-Tiny（不依赖任何预训练权重）
    uv run python experiments/mm_token_pipeline/visual_token_demo.py --mode tiny-vit-random

    # 模式 2: CLIP reference 结构打印（不下载 HF 权重）
    uv run python experiments/mm_token_pipeline/visual_token_demo.py --mode clip-reference

设计原则：
    - tiny-vit-random: 用 torch.nn.Conv2d(3, 192, 16, 16) 模拟 ViT patch embed，
      后接一个简单的 2 层 Transformer encoder（随机权重），输出 visual token shape。
    - clip-reference: 不下载 HuggingFace 权重。仅在 transformers 可用时打印
      CLIPVisionModel 的结构参数（层数、head 数、输出维度）；若不可用，直接输出
      shape 契约以供后续对齐。
"""

import argparse
import sys

import torch
import torch.nn as nn


# ── 共享的 shape 契约（CLIP ViT-B/32 的参数） ──
CLIP_CONTRACT = {
    "model": "openai/clip-vit-base-patch32",
    "image_size": 224,
    "patch_size": 32,
    "num_patches": 49,  # (224/32)^2
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "output_dim": 512,  # CLIP projection dim
    "visual_tokens": "49 + 1 (CLS token) = 50",
    "per_token_dim": 768,
}


def demo_tiny_vit_random(img_size: int = 224, patch_size: int = 16):
    """随机权重 Tiny ViT 模拟。

    结构：
        patch_embed: Conv2d(3, hidden_dim, patch, patch)
        pos_embed: 可学习位置编码
        transformer: 2 层 TransformerEncoder
    """
    hidden_dim = 192
    num_heads = 3
    num_layers = 2

    num_patches = (img_size // patch_size) ** 2
    print("=== Tiny-ViT 随机模拟 ===")
    print(f"图片尺寸: {img_size}×{img_size}")
    print(f"Patch 大小: {patch_size}×{patch_size}")
    print(f"Patch 数量: {num_patches}")
    print(f"隐藏维度: {hidden_dim}")
    print(f"Transformer 层数: {num_layers}")
    print(f"注意力头数: {num_heads}")
    print()

    # 1. Patch embedding
    patch_embed = nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size)

    # 2. 位置编码
    pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_dim))
    cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    # 3. Transformer encoder
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # 4. 构造 dummy 输入
    dummy_img = torch.randn(1, 3, img_size, img_size)  # (B, 3, H, W)

    with torch.no_grad():
        # Patch embed
        x = patch_embed(dummy_img)  # (1, hidden_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (1, num_patches, hidden_dim)

        # 添加 CLS token
        cls_tokens = cls_token.expand(1, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (1, num_patches+1, hidden_dim)

        # 添加位置编码
        x = x + pos_embed

        # Transformer
        x = transformer(x)

    print(f"输入图像 shape:    {dummy_img.shape}  # (B, 3, H, W)")
    print(f"Patch 嵌入后 shape: (1, {num_patches}, {hidden_dim})")
    print(f"加 CLS 后 shape:    {x.shape}  # (B, num_patches+1, hidden_dim)")
    print(f"最终 visual tokens: {num_patches + 1} 个 (含 CLS), 每个 {hidden_dim} 维")
    print("Tiny-ViT 随机模拟完成 ✓")

    return x


def demo_clip_reference():
    """打印 CLIP ViT 结构契约（不下载权重）。"""
    print("=== CLIP Reference 结构 ===")
    print()

    try:
        # 尝试 import 但不加载权重
        from transformers import CLIPVisionConfig

        config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
        print(f"模型名称: openai/clip-vit-base-patch32")
        print(f"image_size:          {config.image_size}")
        print(f"patch_size:          {config.patch_size}")
        print(f"hidden_size:         {config.hidden_size}")
        print(f"num_hidden_layers:   {config.num_hidden_layers}")
        print(f"num_attention_heads: {config.num_attention_heads}")
        print(f"intermediate_size:   {config.intermediate_size}")
        print(f"projection_dim:      {config.projection_dim}")

        # 用随机权重构造结构（不加载 HF 权重）
        from transformers import CLIPVisionModel
        model = CLIPVisionModel(config)  # 随机初始化，无网络请求

        dummy_img = torch.randn(1, 3, config.image_size, config.image_size)
        with torch.no_grad():
            out = model(dummy_img)

        print()
        print("--- 用随机权重 forward ---")
        print(f"输入 shape:   {dummy_img.shape}")
        print(f"last_hidden_state shape: {out.last_hidden_state.shape}")
        print(f"pooler_output shape:     {out.pooler_output.shape}")
        print(f"总 token 数: {out.last_hidden_state.shape[1]} (1 CLS + {config.image_size // config.patch_size}^2 patches)")
        print()
        print("说明：以上使用随机权重初始化 CLIPVisionModel，并无网络请求。")
        print("      实际训练 / 推理时应加载 HF pretrained weights。")

    except ImportError:
        print("CLIP reference 模式需要 transformers 库；当前仅打印 shape 契约。")
        print()
        for k, v in CLIP_CONTRACT.items():
            print(f"  {k}: {v}")
        print()
        print("契约说明：")
        print("  输入: (B, 3, 224, 224) → 输出: (B, 50, 768)")
        print("  50 tokens = 1 CLS + 49 patches (224/32=7, 7×7=49)")
        print("  pooler_output: (B, 768)  # CLS token 经过 projection")

    print()
    print("CLIP Reference 演示完成 ✓")


def main():
    parser = argparse.ArgumentParser(
        description="Visual Token 生成演示")
    parser.add_argument("--mode", required=True,
                        choices=["tiny-vit-random", "clip-reference"],
                        help="tiny-vit-random: 随机 ViT-Tiny 模拟; "
                             "clip-reference: CLIP 结构打印（不下载权重）")
    args = parser.parse_args()

    if args.mode == "tiny-vit-random":
        demo_tiny_vit_random()
    else:
        demo_clip_reference()


if __name__ == "__main__":
    main()
