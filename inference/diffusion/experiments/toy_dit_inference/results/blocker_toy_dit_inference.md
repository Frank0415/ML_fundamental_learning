# Toy DiT Inference — Blocker 记录

**日期**: 2026-06-07 04:18:50

**阻塞环节**: 环境依赖

**配置**: prompt='a cat', steps=4, device=cpu

## 错误信息

```
torch 未安装 (No module named 'torch')。
请安装: pip install torch>=2.7
或在 .venv 中: uv pip install torch
```

## 影响

- 阻塞 Pipeline smoke test
- 阻塞 T12 demo 运行

## 建议

- 在 `.venv` 中安装 torch: `uv pip install torch`
- 或在远程 RTX 5070 Ti 上运行
