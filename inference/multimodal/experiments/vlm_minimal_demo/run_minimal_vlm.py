#!/usr/bin/env python3
"""run_minimal_vlm.py — Wave 4 Task 9: End-to-end visual+text embeddings pipeline.

Modes:
  --mode text_parity        Assert input_ids path == inputs_embeds path (allclose)
  --mode invalid_dual_input  Assert dual input raises ValueError
  --mode prefill_only       Concat visual+text embeddings, forward once through Qwen3
  --mode prefill_decode     Prefill with KV cache, then decode max_new_tokens

本 demo 的目标是工程路径跑通；**不**保证语义质量。
随机 projector 输出可能没有语义。
No HF visual weights required — uses random tiny-ViT + random projector.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Qwen3Config

# ── Path setup ────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]  # multimodal/
_MINIVLLM_ROOT = _PROJECT_ROOT / "minivLLM"
if str(_MINIVLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIVLLM_ROOT))

from minivllm.model.qwen3 import Qwen3
from minivllm.core.kv_cache import KVCache

RESULTS_DIR = _SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_DIR = _PROJECT_ROOT / ".omo" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# ── Model config (Qwen3-0.6B mini, random init) ────────────────────────────
QWEN3_0_6B = dict(
    hidden_size=1024,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    intermediate_size=3072,
    vocab_size=151936,
    max_position_embeddings=4096 * 32,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000,
    hidden_act="silu",
    tie_word_embeddings=True,
)

ATOL = 1e-5
RTOL = 1e-4


# ══════════════════════════════════════════════════════════════════════════
# Utility: importlib helper
# ══════════════════════════════════════════════════════════════════════════

def _load_module(name: str, filepath: Path):
    """Load a Python module from an arbitrary file path via importlib."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════
# Visual embedding pipeline (tiny-ViT random + random projector)
# ══════════════════════════════════════════════════════════════════════════

def get_visual_embeddings(
    preprocessed_image: torch.Tensor,
    llm_hidden_size: int = 1024,
    patch_size: int = 16,
) -> torch.Tensor:
    """Random tiny ViT + random projector → visual embeddings.

    Args:
        preprocessed_image: (3, H, W) normalized tensor
        llm_hidden_size: target LLM hidden dim (1024 for Qwen3-0.6B)
        patch_size: ViT patch size (16 → 196 patches for 224×224)

    Returns:
        (1, num_visual, llm_hidden_size) visual embeddings
    """
    hidden_dim = 192
    num_heads = 3
    num_layers = 2
    img_h, img_w = preprocessed_image.shape[1], preprocessed_image.shape[2]
    num_patches = (img_h // patch_size) * (img_w // patch_size)
    num_visual = num_patches + 1  # + CLS token

    # Build tiny ViT (random weights)
    patch_embed = nn.Conv2d(3, hidden_dim, patch_size, stride=patch_size)
    pos_embed = nn.Parameter(torch.randn(1, num_visual, hidden_dim))
    cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # Random projector: hidden_dim → llm_hidden_size
    projector = nn.Linear(hidden_dim, llm_hidden_size)

    # Forward
    img_batch = preprocessed_image.unsqueeze(0)  # (1, 3, H, W)
    with torch.no_grad():
        x = patch_embed(img_batch)  # (1, hidden_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (1, num_patches, hidden_dim)

        cls_tokens = cls_token.expand(1, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (1, num_visual, hidden_dim)
        x = x + pos_embed
        x = transformer(x)
        x = projector(x)  # (1, num_visual, llm_hidden_size)

    return x


# ══════════════════════════════════════════════════════════════════════════
# Tokenization helper
# ══════════════════════════════════════════════════════════════════════════

def _get_tokenizer():
    """Load Qwen3 tokenizer. Returns None on failure."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", trust_remote_code=True
        )
        return tok
    except Exception:
        return None


def encode_prompt(tokenizer, prompt: str) -> list:
    """Encode text prompt to token IDs.

    Falls back to simple ord-based encoding if tokenizer unavailable.
    Semantic quality NOT guaranteed in either case — pipeline path demo only.
    """
    if tokenizer is not None:
        return tokenizer.encode(prompt, add_special_tokens=False)

    # Fallback: simple character-level encoding
    vocab_size = QWEN3_0_6B["vocab_size"]
    ids = []
    for ch in prompt:
        ids.append((ord(ch) % (vocab_size - 1)) + 1)
    return ids


# ══════════════════════════════════════════════════════════════════════════
# Distributed init
# ══════════════════════════════════════════════════════════════════════════

def _init_distributed():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


# ══════════════════════════════════════════════════════════════════════════
# Mode: text_parity (Task 8)
# ══════════════════════════════════════════════════════════════════════════

def run_text_parity(device: torch.device, seq_len: int = 8) -> dict:
    """Compare input_ids path vs inputs_embeds path; assert allclose."""
    cfg = Qwen3Config(**QWEN3_0_6B)
    model = Qwen3(cfg).to(device).eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, QWEN3_0_6B["vocab_size"], (seq_len,), device=device)
    positions = torch.arange(seq_len, device=device)

    with torch.no_grad():
        hidden_id = model(input_ids=input_ids, positions=positions)
        logits_id = model.compute_logits(hidden_id)

    with torch.no_grad():
        embed = model.model.embed_tokens(input_ids)
        hidden_emb = model(inputs_embeds=embed, positions=positions)
        logits_emb = model.compute_logits(hidden_emb)

    diff = (logits_id - logits_emb).abs()
    max_abs_diff = diff.max().item()
    passed = torch.allclose(logits_id, logits_emb, atol=ATOL, rtol=RTOL)

    top_diffs = []
    if not passed:
        flat_diff = diff.flatten()
        topk_vals, topk_idx = flat_diff.topk(min(10, flat_diff.numel()))
        for v, i in zip(topk_vals.tolist(), topk_idx.tolist()):
            top_diffs.append({
                "idx": i,
                "abs_diff": round(v, 8),
                "logits_id": round(logits_id.flatten()[i].item(), 8),
                "logits_emb": round(logits_emb.flatten()[i].item(), 8),
            })

    result = {
        "mode": "text_parity",
        "seq_len": seq_len,
        "max_abs_diff": round(max_abs_diff, 8),
        "passed": passed,
        "threshold": {"atol": ATOL, "rtol": RTOL},
    }
    if top_diffs:
        result["top_diffs"] = top_diffs

    status = "PASS" if passed else "FAIL"
    print(f"  seq_len={seq_len}: max|diff|={max_abs_diff:.2e}  {status}")
    if top_diffs:
        print(f"    top element diffs: {top_diffs[:3]}")
    print(f"  logits_id  range: [{logits_id.min().item():.6f}, {logits_id.max().item():.6f}]")
    print(f"  logits_emb range: [{logits_emb.min().item():.6f}, {logits_emb.max().item():.6f}]")

    json_path = RESULTS_DIR / "text_parity.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON written to: {json_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Mode: invalid_dual_input (Task 8)
# ══════════════════════════════════════════════════════════════════════════

def run_invalid_dual_input(device: torch.device, seq_len: int = 8) -> dict:
    """Assert ValueError when both input_ids and inputs_embeds are provided."""
    cfg = Qwen3Config(**QWEN3_0_6B)
    model = Qwen3(cfg).to(device).eval()

    torch.manual_seed(42)
    input_ids = torch.randint(0, QWEN3_0_6B["vocab_size"], (seq_len,), device=device)
    positions = torch.arange(seq_len, device=device)

    with torch.no_grad():
        embed = model.model.embed_tokens(input_ids)

    caught_error = None
    error_message = ""
    try:
        with torch.no_grad():
            model(input_ids=input_ids, inputs_embeds=embed, positions=positions)
    except ValueError as e:
        caught_error = "ValueError"
        error_message = str(e)
    except Exception as e:
        caught_error = type(e).__name__
        error_message = str(e)

    keywords = ["input_ids", "inputs_embeds", "cannot both"]
    has_keyword = any(kw.lower() in error_message.lower() for kw in keywords)
    passed = caught_error == "ValueError" and has_keyword

    result = {
        "mode": "invalid_dual_input",
        "seq_len": seq_len,
        "error_type": caught_error or "no_error",
        "error_message": error_message,
        "has_keyword": has_keyword,
        "keywords_checked": keywords,
        "passed": passed,
    }

    status = "PASS" if passed else "FAIL"
    print(f"  Error caught: {caught_error}")
    print(f"  Error message: {error_message[:200]}")
    print(f"  Keyword found: {has_keyword}")
    print(f"  {status}")

    json_path = RESULTS_DIR / "invalid_dual_input.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  JSON written to: {json_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Mode: prefill_only (Task 9)
# ══════════════════════════════════════════════════════════════════════════

def run_prefill_only(
    device: torch.device,
    image_path: str,
    prompt: str,
    tokenizer=None,
) -> dict:
    """Load image → visual embeddings → concat with text → forward once.

    Steps:
      1. Preprocess image (resize 224, normalize)
      2. Tiny-ViT random → visual embeddings (1, Nv, 1024)
      3. Tokenize prompt → text token IDs
      4. Build multimodal sequence: [vis_start] [v_1..v_Nv] [vis_end] [t_1..t_M]
      5. Embed text tokens + vision markers via model.model.embed_tokens
      6. Concatenate: [text_emb[:1], visual_emb, text_emb[1:]]
      7. Forward through model(inputs_embeds=...)
      8. Compute last-token logits, argmax
    """
    # ── Import image_preprocess via importlib ──────────────────────────
    pipeline_dir = _SCRIPT_DIR.parent / "mm_token_pipeline"
    img_prep = _load_module(
        "image_preprocess",
        pipeline_dir / "image_preprocess.py",
    )

    # ── Step 1: Preprocess image ──────────────────────────────────────
    print(f"\n  [Step 1] Preprocessing image: {image_path}")
    img_pil = img_prep.load_and_resize(image_path, size=224)
    img_tensor = img_prep.image_to_tensor(img_pil)  # (3, 224, 224)
    print(f"    Image tensor shape: {tuple(img_tensor.shape)}")

    # ── Step 2: Visual embeddings ─────────────────────────────────────
    print(f"\n  [Step 2] Generating visual embeddings (tiny-ViT random + projector)")
    visual_emb = get_visual_embeddings(img_tensor, llm_hidden_size=1024, patch_size=16)
    # visual_emb: (1, num_visual, 1024)
    num_visual = visual_emb.shape[1]
    print(f"    Visual embeddings shape: {tuple(visual_emb.shape)}")
    print(f"    num_visual_tokens: {num_visual}  (1 CLS + {num_visual - 1} patches)")

    # ── Step 3: Tokenize prompt ───────────────────────────────────────
    print(f"\n  [Step 3] Tokenizing prompt: {repr(prompt)}")
    text_token_ids = encode_prompt(tokenizer, prompt)
    num_text = len(text_token_ids)
    print(f"    Prompt tokens ({num_text}): {text_token_ids[:10]}{'...' if num_text > 10 else ''}")

    # ── Step 4: Get vision special token IDs ──────────────────────────
    if tokenizer is not None:
        vis_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vis_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    else:
        # Fallback: use IDs from mm_sequence_builder conventions
        vis_start_id = 151652
        vis_end_id = 151653

    print(f"    vision_start_id: {vis_start_id}, vision_end_id: {vis_end_id}")

    # ── Step 5: Build model ───────────────────────────────────────────
    print(f"\n  [Step 4] Building Qwen3 model (random init)")
    cfg = Qwen3Config(**QWEN3_0_6B)
    model = Qwen3(cfg).to(device).eval()
    hidden_size = QWEN3_0_6B["hidden_size"]

    # ── Step 6: Embed text tokens (including vision markers) ──────────
    # Sequence structure:
    #   Position 0:         vision_start → text embed
    #   Positions 1..Nv:    visual tokens → visual embed (from tiny ViT + projector)
    #   Position Nv+1:      vision_end   → text embed
    #   Positions Nv+2..:   text tokens  → text embed

    total_len = 2 + num_visual + num_text  # vis_start + Nv + vis_end + M
    print(f"\n  [Step 5] Building multimodal sequence")
    print(f"    Layout: VIS_START | v_1..v_{num_visual} | VIS_END | t_1..t_{num_text}")
    print(f"    Total sequence length: {total_len}")

    # Collect all text-embedded token IDs
    text_embed_ids = torch.tensor(
        [vis_start_id, vis_end_id] + text_token_ids,
        dtype=torch.long, device=device,
    )  # (2 + num_text,)

    with torch.no_grad():
        text_emb = model.model.embed_tokens(text_embed_ids)  # (2+M, 1024)

    # ── Step 7: Concatenate full inputs_embeds ────────────────────────
    print(f"\n  [Step 6] Concatenating visual + text embeddings")
    visual_flat = visual_emb.squeeze(0).to(device)  # (Nv, 1024)

    inputs_embeds = torch.cat([
        text_emb[0:1],       # vis_start (1, 1024)
        visual_flat,          # visual tokens (Nv, 1024)
        text_emb[1:],         # vis_end + text tokens (1+M, 1024)
    ], dim=0)  # (total_len, 1024)

    print(f"    inputs_embeds shape: {tuple(inputs_embeds.shape)}")

    # ── Step 8: Position IDs ──────────────────────────────────────────
    positions = torch.arange(total_len, device=device)

    # ── Step 9: Forward pass ──────────────────────────────────────────
    print(f"\n  [Step 7] Forward pass (prefill, no KV cache)")
    with torch.no_grad():
        hidden = model(inputs_embeds=inputs_embeds, positions=positions)
        # hidden: (total_len, hidden_size)
        logits = model.compute_logits(hidden[-1:])  # last token only
        last_logits = logits[0]  # (vocab_size,)
        predicted_token_id = int(last_logits.argmax().item())
        top5_vals, top5_ids = last_logits.topk(5)

    print(f"    hidden shape: {tuple(hidden.shape)}")
    print(f"    last-token logits shape: {tuple(last_logits.shape)}")
    print(f"    argmax token ID: {predicted_token_id}")
    print(f"    top-5 token IDs: {top5_ids.tolist()}")
    if tokenizer is not None:
        print(f"    top-5 decoded:  {[tokenizer.decode([tid]) for tid in top5_ids.tolist()]}")

    # ── Step 10: Save result ──────────────────────────────────────────
    result = {
        "mode": "prefill_only",
        "prompt": prompt,
        "image_path": image_path,
        "num_visual_tokens": num_visual,
        "num_text_tokens": num_text,
        "total_sequence_length": total_len,
        "sequence_layout": "VIS_START | v_1..v_N | VIS_END | t_1..t_M",
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "hidden_shape": list(hidden.shape),
        "last_logits_argmax_token_id": predicted_token_id,
        "top5_token_ids": top5_ids.tolist(),
        "visual_embedding_source": "tiny-vit-random + random-projector(192→1024)",
        "semantic_warning": (
            "本 demo 的目标是工程路径跑通；不保证语义质量。"
            "随机 projector 输出可能没有语义。"
        ),
    }

    json_path = RESULTS_DIR / "prefill_only.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON written to: {json_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Mode: prefill_decode (Task 9)
# ══════════════════════════════════════════════════════════════════════════

def run_prefill_decode(
    device: torch.device,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 8,
    tokenizer=None,
) -> dict:
    """Prefill with KV cache → decode loop generating max_new_tokens.

    Steps:
      1-7. Same as prefill_only, but forward with kv_cache + is_prefill=True
      8. Decode loop: for each step, embed last token, forward with is_prefill=False
      9. Collect generated token IDs
    """
    # ── Import image_preprocess via importlib ──────────────────────────
    pipeline_dir = _SCRIPT_DIR.parent / "mm_token_pipeline"
    img_prep = _load_module(
        "image_preprocess",
        pipeline_dir / "image_preprocess.py",
    )

    # ── Model config ──────────────────────────────────────────────────
    cfg = Qwen3Config(**QWEN3_0_6B)
    hidden_size = QWEN3_0_6B["hidden_size"]
    num_layers = QWEN3_0_6B["num_hidden_layers"]

    # head_dim: read from config — Qwen3Config computes it from num_key_value_heads
    # (128 for 1024/8), NOT from num_attention_heads
    cfg_head_dim = getattr(cfg, "head_dim", None) or (hidden_size // QWEN3_0_6B["num_key_value_heads"])

    # ── Step 1: Preprocess image ──────────────────────────────────────
    print(f"\n  [Step 1] Preprocessing image: {image_path}")
    img_pil = img_prep.load_and_resize(image_path, size=224)
    img_tensor = img_prep.image_to_tensor(img_pil)
    print(f"    Image tensor shape: {tuple(img_tensor.shape)}")

    # ── Step 2: Visual embeddings ─────────────────────────────────────
    print(f"\n  [Step 2] Generating visual embeddings (tiny-ViT random + projector)")
    visual_emb = get_visual_embeddings(img_tensor, llm_hidden_size=1024, patch_size=16)
    num_visual = visual_emb.shape[1]
    print(f"    Visual embeddings shape: {tuple(visual_emb.shape)}")
    print(f"    num_visual_tokens: {num_visual}")

    # ── Step 3: Tokenize prompt ───────────────────────────────────────
    print(f"\n  [Step 3] Tokenizing prompt: {repr(prompt)}")
    text_token_ids = encode_prompt(tokenizer, prompt)
    num_text = len(text_token_ids)
    print(f"    Prompt tokens ({num_text}): {text_token_ids[:10]}{'...' if num_text > 10 else ''}")

    # ── Step 4: Vision special tokens ─────────────────────────────────
    if tokenizer is not None:
        vis_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vis_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    else:
        vis_start_id = 151652
        vis_end_id = 151653

    total_len = 2 + num_visual + num_text
    print(f"\n  [Step 4] Sequence length: {total_len} "
          f"(1 vis_start + {num_visual} visual + 1 vis_end + {num_text} text)")

    # ── Step 5: Build model + KV cache ────────────────────────────────
    print(f"\n  [Step 5] Building model + KV cache")
    model = Qwen3(cfg).to(device).eval()

    # KV cache sizing: total_len + max_new_tokens with headroom
    kv_max_seq = max(total_len + max_new_tokens + 128, 512)
    kv_cache = KVCache(
        num_layers=num_layers,
        max_seq_len=kv_max_seq,
        num_kv_heads=QWEN3_0_6B["num_key_value_heads"],
        head_dim=cfg_head_dim,
        device=device,
        dtype=torch.float32,
    )
    print(f"    KV cache: {num_layers} layers × {kv_max_seq} max_seq "
          f"× {QWEN3_0_6B['num_key_value_heads']} kv_heads × {cfg_head_dim} head_dim")

    # ── Step 6: Build inputs_embeds ───────────────────────────────────
    print(f"\n  [Step 6] Building inputs_embeds")
    text_embed_ids = torch.tensor(
        [vis_start_id, vis_end_id] + text_token_ids,
        dtype=torch.long, device=device,
    )
    with torch.no_grad():
        text_emb = model.model.embed_tokens(text_embed_ids)

    visual_flat = visual_emb.squeeze(0).to(device)
    inputs_embeds = torch.cat([
        text_emb[0:1],       # vis_start
        visual_flat,          # visual tokens
        text_emb[1:],         # vis_end + text tokens
    ], dim=0)  # (total_len, 1024)

    print(f"    inputs_embeds shape: {tuple(inputs_embeds.shape)}")

    # ── Step 7: Prefill forward (is_prefill=True) ─────────────────────
    print(f"\n  [Step 7] Prefill forward (is_prefill=True, KV cache write)")
    positions = torch.arange(total_len, device=device)

    with torch.no_grad():
        hidden = model(
            inputs_embeds=inputs_embeds,
            positions=positions,
            kv_cache=kv_cache,
            is_prefill=True,
        )
        logits = model.compute_logits(hidden[-1:])
        next_token_id = int(logits[0].argmax().item())

    print(f"    Prefill hidden shape: {tuple(hidden.shape)}")
    print(f"    First generated token ID: {next_token_id}")
    if tokenizer is not None:
        print(f"    First token decoded: {tokenizer.decode([next_token_id])!r}")

    # ── Step 8: Decode loop (is_prefill=False) ────────────────────────
    print(f"\n  [Step 8] Decode loop (max_new_tokens={max_new_tokens})")
    generated_ids = [next_token_id]
    cur_pos = total_len

    for step in range(1, max_new_tokens):
        # Embed current token
        next_emb = model.model.embed_tokens(
            torch.tensor([next_token_id], device=device)
        )  # (1, 1024)
        pos_t = torch.tensor([cur_pos], device=device)

        with torch.no_grad():
            hidden = model(
                inputs_embeds=next_emb,
                positions=pos_t,
                kv_cache=kv_cache,
                is_prefill=False,
            )
            logits = model.compute_logits(hidden[-1:])
            next_token_id = int(logits[0].argmax().item())

        generated_ids.append(next_token_id)
        cur_pos += 1

        if tokenizer is not None:
            decoded = tokenizer.decode([next_token_id])
            print(f"    step {step}: pos={cur_pos-1}  token_id={next_token_id}  "
                  f"decoded={decoded!r}")
        else:
            print(f"    step {step}: pos={cur_pos-1}  token_id={next_token_id}")

    # ── Step 9: Summary ───────────────────────────────────────────────
    print(f"\n  [Step 9] Generated {len(generated_ids)} tokens")
    print(f"    Generated token IDs: {generated_ids}")
    if tokenizer is not None:
        full_decoded = tokenizer.decode(generated_ids)
        print(f"    Full decoded: {full_decoded!r}")

    # ── Step 10: Save result ──────────────────────────────────────────
    result = {
        "mode": "prefill_decode",
        "prompt": prompt,
        "image_path": image_path,
        "max_new_tokens": max_new_tokens,
        "num_visual_tokens": num_visual,
        "num_text_tokens": num_text,
        "total_sequence_length": total_len,
        "sequence_layout": "VIS_START | v_1..v_N | VIS_END | t_1..t_M",
        "inputs_embeds_shape": list(inputs_embeds.shape),
        "generated_token_ids": generated_ids,
        "num_generated_tokens": len(generated_ids),
        "kv_cache_config": {
            "num_layers": num_layers,
            "max_seq_len": kv_max_seq,
            "num_kv_heads": QWEN3_0_6B["num_key_value_heads"],
            "head_dim": cfg_head_dim,
        },
        "layer_idx_behavior": (
            "Qwen3Model.forward internally enumerates self.layers, "
            "passing layer_idx=i to each Qwen3DecoderLayer, "
            "which passes it to Qwen3Attn. No external layer_idx needed."
        ),
        "visual_embedding_source": "tiny-vit-random + random-projector(192→1024)",
        "semantic_warning": (
            "本 demo 的目标是工程路径跑通；不保证语义质量。"
            "随机 projector 输出可能没有语义。"
        ),
    }

    json_path = RESULTS_DIR / "prefill_decode.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  JSON written to: {json_path}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Task 9: End-to-end visual+text embeddings pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["text_parity", "invalid_dual_input", "prefill_only", "prefill_decode"],
        required=True,
    )
    parser.add_argument("--seq-len", type=int, default=8,
                        help="Sequence length for text_parity / invalid_dual_input")
    parser.add_argument("--image", type=str,
                        default="experiments/vlm_minimal_demo/sample_images/demo.jpg",
                        help="Path to input image (for prefill_* modes)")
    parser.add_argument("--prompt", type=str, default="请描述这张图片。",
                        help="Text prompt (for prefill_* modes)")
    parser.add_argument("--max-new-tokens", type=int, default=8,
                        help="Max tokens to generate in prefill_decode mode")
    args = parser.parse_args()

    _init_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {QWEN3_0_6B['num_hidden_layers']} layers, "
          f"hidden={QWEN3_0_6B['hidden_size']}, "
          f"heads={QWEN3_0_6B['num_attention_heads']}, "
          f"kv_heads={QWEN3_0_6B['num_key_value_heads']}")
    print(f"Threshold: atol={ATOL}, rtol={RTOL}")
    print(f"Mode: {args.mode}")

    # Load tokenizer once for prefill modes
    tokenizer = None
    if args.mode in ("prefill_only", "prefill_decode"):
        print(f"Loading tokenizer...")
        tokenizer = _get_tokenizer()
        if tokenizer is not None:
            print(f"  Tokenizer loaded: Qwen/Qwen3-0.6B (vocab={tokenizer.vocab_size})")
        else:
            print(f"  WARNING: Tokenizer unavailable, using fallback ord-based encoding")

    if args.mode == "text_parity":
        print(f"\n{' Text Parity Test ':-^50}")
        result = run_text_parity(device, args.seq_len)
        if not result["passed"]:
            print("\nFAIL: input_ids and inputs_embeds paths did not match!")
            sys.exit(1)

    elif args.mode == "invalid_dual_input":
        print(f"\n{' Invalid Dual Input Test ':-^50}")
        result = run_invalid_dual_input(device, args.seq_len)
        if not result["passed"]:
            if result["error_type"] is None or result["error_type"] == "no_error":
                print("\nFAIL: No error was raised when both input_ids and "
                      "inputs_embeds were provided!")
                sys.exit(2)
            else:
                print(f"\nFAIL: Wrong error type ({result['error_type']}) "
                      f"or missing keyword in message!")
                sys.exit(1)

    elif args.mode == "prefill_only":
        print(f"\n{' Prefill Only (Visual+Text Concat) ':-^50}")
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        result = run_prefill_only(device, args.image, args.prompt, tokenizer)

    elif args.mode == "prefill_decode":
        print(f"\n{' Prefill + Decode (KV Cache Generate) ':-^50}")
        if not os.path.exists(args.image):
            print(f"ERROR: Image not found: {args.image}", file=sys.stderr)
            sys.exit(1)
        result = run_prefill_decode(
            device, args.image, args.prompt, args.max_new_tokens, tokenizer,
        )

    print(f"\n{' ALL DONE ':-^50}")


if __name__ == "__main__":
    main()
