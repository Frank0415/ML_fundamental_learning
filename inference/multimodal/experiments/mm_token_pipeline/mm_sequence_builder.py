#!/usr/bin/env python3
"""多模态序列构造器：两种布局，输出 position_ids、attention_mask、num_visual_tokens。

用法：
    # 布局 1: bos_image_text
    uv run python experiments/mm_token_pipeline/mm_sequence_builder.py \\
        --layout bos_image_text --num-visual 256 --num-text 16

    # 布局 2: placeholder_expanded
    uv run python experiments/mm_token_pipeline/mm_sequence_builder.py \\
        --layout placeholder_expanded --num-visual 256 --num-text 16

输出保存到 experiments/mm_token_pipeline/results/ 目录。
"""

import argparse
import json
import os

import torch

# ── 特殊 token ID（与 Qwen3 tokenizer 对齐，但此处仅作演示） ──
BOS_ID = 151643
IMG_START_ID = 151655   # <|vision_start|>
IMG_END_ID = 151656     # <|vision_end|>
IMG_PLACEHOLDER_ID = 151654  # <|image_pad|>
TEXT_BASE_ID = 24       # 文本 token 起始 ID（仅演示用）


def build_bos_image_text(num_visual: int, num_text: int) -> dict:
    """bos_image_text 布局：
        [BOS] [IMG_START] v_1 v_2 ... v_N [IMG_END] t_1 t_2 ... t_M

    视觉 token 在文本 token 之前，BOS 在最前面。
    """
    # 构造序列
    seq = []
    # BOS
    seq.append(BOS_ID)
    # IMG_START
    seq.append(IMG_START_ID)
    # Visual tokens (用占位符 ID)
    for i in range(num_visual):
        seq.append(IMG_PLACEHOLDER_ID)
    # IMG_END
    seq.append(IMG_END_ID)
    # Text tokens
    for i in range(num_text):
        seq.append(TEXT_BASE_ID + i)

    total_len = len(seq)
    input_ids = torch.tensor([seq], dtype=torch.long)  # (1, total_len)

    # position_ids: 从 0 开始顺序递增
    position_ids = torch.arange(total_len, dtype=torch.long).unsqueeze(0)

    # attention_mask: 双向注意力，全 1
    attention_mask = torch.ones(1, total_len, dtype=torch.bool)

    # causal_mask: 左下三角 (但此处仅返回 bool mask)
    causal_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))

    num_visual_tokens = num_visual + 2  # + IMG_START + IMG_END
    num_text_tokens = num_text

    return {
        "layout": "bos_image_text",
        "num_visual_tokens": num_visual_tokens,
        "num_text_tokens": num_text_tokens,
        "total_len": total_len,
        "sequence_structure": "BOS | IMG_START | v_1...v_N | IMG_END | t_1...t_M",
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "causal_mask": causal_mask,
        "visual_span": (2, 2 + num_visual),  # 视觉 token 在序列中的 [start, end) 区间
        "text_span": (3 + num_visual, total_len),  # 文本 token 的 [start, end)
    }


def build_placeholder_expanded(num_visual: int, num_text: int) -> dict:
    """placeholder_expanded 布局：
        t_1 ... t_k [IMG_PLACEHOLDER_POS] t_{k+1} ... t_M

    一个占位符被展开为 num_visual 个 visual token，插入文本序列中间。
    这里 k = num_text // 2（插入到文本中间）。
    """
    k = num_text // 2  # 插入位置

    # 构造序列
    seq = []
    # 前半段文本
    for i in range(k):
        seq.append(TEXT_BASE_ID + i)
    # Visual tokens（展开的占位符）
    for i in range(num_visual):
        seq.append(IMG_PLACEHOLDER_ID)
    # 后半段文本
    for i in range(k, num_text):
        seq.append(TEXT_BASE_ID + i)

    total_len = len(seq)
    input_ids = torch.tensor([seq], dtype=torch.long)  # (1, total_len)

    # position_ids: 0..total_len-1 顺序
    position_ids = torch.arange(total_len, dtype=torch.long).unsqueeze(0)

    # attention_mask: 双向注意力 → 全 1
    attention_mask = torch.ones(1, total_len, dtype=torch.bool)

    # causal_mask: 左下三角
    causal_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))

    return {
        "layout": "placeholder_expanded",
        "num_visual_tokens": num_visual,
        "num_text_tokens": num_text,
        "text_prefix_len": k,
        "text_suffix_len": num_text - k,
        "total_len": total_len,
        "sequence_structure": f"t_1...t_{k} | v_1...v_{num_visual} | t_{k+1}...t_{num_text}",
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "causal_mask": causal_mask,
        "visual_span": (k, k + num_visual),  # 视觉 token 区间
        "text_spans": [(0, k), (k + num_visual, total_len)],  # 文本区间
    }


def main():
    parser = argparse.ArgumentParser(description="多模态序列构造器")
    parser.add_argument("--layout", required=True,
                        choices=["bos_image_text", "placeholder_expanded"],
                        help="序列布局模式")
    parser.add_argument("--num-visual", type=int, default=256,
                        help="visual token 数量 (默认 256)")
    parser.add_argument("--num-text", type=int, default=16,
                        help="文本 token 数量 (默认 16)")
    args = parser.parse_args()

    if args.layout == "bos_image_text":
        result = build_bos_image_text(args.num_visual, args.num_text)
    else:
        result = build_placeholder_expanded(args.num_visual, args.num_text)

    # ── 打印 ──
    print(f"=== 多模态序列构造: {result['layout']} ===")
    print(f"布局结构: {result['sequence_structure']}")
    print(f"视觉 token 数: {result['num_visual_tokens']}")
    print(f"文本 token 数: {result['num_text_tokens']}")
    print(f"序列总长度: {result['total_len']}")
    print()

    print(f"input_ids shape:        {result['input_ids'].shape}")
    print(f"input_ids (前10+后5):   {result['input_ids'][0, :10].tolist()} ... {result['input_ids'][0, -5:].tolist()}")
    print(f"position_ids shape:     {result['position_ids'].shape}")
    print(f"position_ids (前10):    {result['position_ids'][0, :10].tolist()}")
    print(f"position_ids (最后5):   {result['position_ids'][0, -5:].tolist()}")
    print(f"attention_mask shape:   {result['attention_mask'].shape}")
    print(f"attention_mask[0]:      {result['attention_mask'][0].tolist()}")
    print(f"causal_mask shape:      {result['causal_mask'].shape}")
    print()
    print(f"视觉 token 在序列中的区间: {result['visual_span']}")
    if "text_span" in result:
        print(f"文本 token 区间:          {result['text_span']}")
    if "text_spans" in result:
        print(f"文本 token 区间 (分段):   {result['text_spans']}")

    # ── 保存到 results/ ──
    out_dir = "experiments/mm_token_pipeline/results"
    os.makedirs(out_dir, exist_ok=True)

    save_file = os.path.join(out_dir, f"sequence_{result['layout']}.pt")
    torch.save({
        "layout": result["layout"],
        "input_ids": result["input_ids"],
        "position_ids": result["position_ids"],
        "attention_mask": result["attention_mask"],
        "causal_mask": result["causal_mask"],
        "num_visual_tokens": result["num_visual_tokens"],
        "num_text_tokens": result["num_text_tokens"],
        "visual_span": result["visual_span"],
    }, save_file)
    print(f"\n已保存到: {save_file}")

    # 同时保存摘要 JSON
    summary_file = os.path.join(out_dir, f"sequence_{result['layout']}.json")
    summary = {k: v for k, v in result.items()
               if not isinstance(v, torch.Tensor)}
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要已保存到: {summary_file}")
    print("序列构造完成 ✓")


if __name__ == "__main__":
    main()
