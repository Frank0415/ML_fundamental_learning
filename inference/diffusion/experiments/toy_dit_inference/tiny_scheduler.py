"""
tiny_scheduler.py — Rectified Flow 调度器（thin import wrapper）

直接 import diffusion_engine.core.scheduler.RectifiedFlowScheduler。
T12 完整说明见 experiments/toy_dit_inference/README.md。
"""

# thin re-export from engine core
from diffusion_engine.core.scheduler import RectifiedFlowScheduler  # noqa: F401
