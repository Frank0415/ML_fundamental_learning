"""
tiny_dit.py — Toy DiT 模型（thin import wrapper）

直接 import diffusion_engine.core.dit.TinyDiT。
T12 完整说明见 experiments/toy_dit_inference/README.md。
"""

# thin re-export from engine core
from diffusion_engine.core.dit import TinyDiT  # noqa: F401
