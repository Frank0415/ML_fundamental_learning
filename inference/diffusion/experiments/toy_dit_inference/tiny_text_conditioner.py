"""
tiny_text_conditioner.py — Toy 文本编码器（thin import wrapper）

直接 import diffusion_engine.core.text_conditioning.ToyTextConditioner。
T12 完整说明见 experiments/toy_dit_inference/README.md。
"""

# thin re-export from engine core
from diffusion_engine.core.text_conditioning import ToyTextConditioner  # noqa: F401
