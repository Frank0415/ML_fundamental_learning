# Reference Video Inference — Blocker (CogVideoX)

**日期**：2026-06-07（占位 — 真实运行日期待定）
**模型**：THUDM/CogVideoX-2b
**设备**：RTX 5070 Ti（12GB VRAM）（远程，待连接）
**执行者**：T15 系统尝试

## 失败原因
**环境未就绪**（占位 blocker）。当前开发环境为 macOS M5，不支持 CUDA。
本脚本需要在远程 RTX 5070 Ti 上运行。此 blocker 将在真实执行环境就绪后被覆盖或删除。

## 详细日志
```
[BLOCKER] CUDA 不可用。本脚本需要在 NVIDIA GPU 上运行。Mac M5 不支持 CUDA。
```

## 结论
这是占位 blocker，表示 T15 脚本已就绪但尚未在远程 GPU 上执行。
脚本 `run_cogvideox_if_possible.py` 已通过 `--help` 和代码结构自检。
CogVideoX-2B 为 Apache 2.0 协议，无授权障碍。待远程 RTX 5070 Ti 环境就绪后执行。

## 对后续的建议
- 在远程 RTX 5070 Ti 上运行此脚本
- 优先尝试默认小规格（16f×256×256, 8 steps, bf16, cpu_offload）
- CogVideoX-2B 是最无授权障碍的模型，应优先尝试
