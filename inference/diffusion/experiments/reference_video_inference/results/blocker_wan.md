# Reference Video Inference - Blocker (Wan)

**日期**：2026-06-07（占位 - 真实运行日期待定）
**模型**：Wan-AI/Wan2.1-T2V-1.3B
**设备**：可用的 CUDA GPU（中等显存配置）（远程，待连接）
**执行者**：T15 系统尝试

## 失败原因
**环境未就绪**（占位 blocker）。当前开发环境为 macOS M5，不支持 CUDA。
本脚本需要在远程 CUDA GPU 上运行。此 blocker 将在真实执行环境就绪后被覆盖或删除。

## 详细日志
```
[BLOCKER] CUDA 不可用。本脚本需要在 NVIDIA GPU 上运行。Mac M5 不支持 CUDA。
```

## 结论
这是占位 blocker，表示 T15 脚本已就绪但尚未在远程 GPU 上执行。
脚本 `run_wan_if_possible.py` 已通过 `--help` 和代码结构自检。
Wan2.1-1.3B 需要在 HF 接受许可协议。待远程 CUDA GPU 环境就绪后执行。

## 对后续的建议
- 在远程 CUDA GPU 上运行此脚本
- 先确认 HF token + Wan2.1 协议已接受
- 默认小规格（16f×256×256, 8 steps, bf16, cpu_offload）是 中等显存极限操作
- 如 OOM，优先尝试降分辨率到 192^2 和 8 帧
