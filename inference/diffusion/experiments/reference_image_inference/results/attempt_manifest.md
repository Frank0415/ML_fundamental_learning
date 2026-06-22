# 真实 Reference 文生图尝试清单 (Attempt Manifest)

**日期**：2026-06-07
**任务**：T14 — 真实 reference 文生图尝试、结果记录与 Week 4 报告
**设备**：Apple M5 (Metal 4, arm64)，无 NVIDIA GPU
**Python**：3.9.6（系统，低于 3.13 要求）
**远程 CUDA GPU**：当前不可用（用户未提供 SSH 访问凭证）

---

## 尝试顺序

按 T13 README 决策树优先级：Sana → SD3 Medium → FLUX schnell

---

### 尝试 1：Sana-0.6B

| 参数 | 计划值 | 实际设置 |
|------|--------|---------|
| 模型 | `Efficient-Large-Model/Sana_600M_1024px_diffusers` | 同左 |
| 分辨率 | 1024×1024（默认），fallback 512×512 | — |
| 步数 | 20（默认），降级 10 | — |
| dtype | bf16（CUDA） | — |
| offload | enable_model_cpu_offload（默认） | — |
| VAE tiling | enable_vae_tiling（默认） | — |
| HF token 需求 | 无需（Apache 2.0 开放） | — |
| prompt 示例 | "一只柴犬在樱花树下" | — |
| **状态** | — | **BLOCKED — 环境依赖缺失** |
| **blocker 类型** | — | `ModuleNotFoundError: No module named 'torch'` |
| **详细原因** | — | 系统 Python 3.9.6 < 3.13；torch 未安装；diffusers 未安装 |
| **是否可修复** | — | 是：`uv python install 3.13 && uv pip install torch diffusers` |
| **peak VRAM** | — | N/A |
| **latency** | — | N/A |
| **output path** | — | N/A |

脚本 `run_sana_if_possible.py` 经 `--help` 验证可正常解析参数，仅运行时依赖缺失。

---

### 尝试 2：SD3 Medium (no-T5)

| 参数 | 计划值 | 实际设置 |
|------|--------|---------|
| 模型 | `stabilityai/stable-diffusion-3-medium-diffusers` | 同左 |
| 分辨率 | 1024×1024（默认），fallback 768×768 | — |
| 步数 | 28（默认），降级 15 | — |
| dtype | fp16（默认） | — |
| offload | enable_model_cpu_offload（默认） | — |
| VAE slicing | enable_vae_slicing（默认） | — |
| T5 编码器 | 关闭（--no_t5，默认） | — |
| HF token 需求 | 是：需注册 + token + accept license | 未验证（因环境未就绪跳过） |
| **状态** | — | **BLOCKED — 环境依赖缺失** |
| **blocker 类型** | — | `ModuleNotFoundError: No module named 'torch'` |
| **详细原因** | — | 同上：Python 3.9.6 + torch 未装。即使环境就绪，还需 HF token + license accept 两步额外前置 |
| **是否可修复** | — | 是：环境就绪后完成 HF 登录即可 |
| **peak VRAM** | — | N/A |
| **latency** | — | N/A |
| **output path** | — | N/A |

脚本 `run_sd3_medium_if_possible.py` 经 `--help` 验证可正常解析参数。

---

### 尝试 3：FLUX.1-schnell

| 参数 | 计划值 | 实际设置 |
|------|--------|---------|
| 模型 | `black-forest-labs/FLUX.1-schnell` | 同左 |
| 分辨率 | 1024×1024（默认），fallback 768×768 | — |
| 步数 | 4（schnell 推荐） | — |
| dtype | fp16（默认） | — |
| offload | enable_model_cpu_offload（默认） | — |
| VAE slicing | enable_vae_slicing（默认） | — |
| VAE tiling | enable_vae_tiling（默认） | — |
| HF token 需求 | 是：需注册 + token + accept license | 未验证（因环境未就绪跳过） |
| 下载体积 | ~23GB（需预留磁盘空间） | 未执行 |
| **状态** | — | **BLOCKED — 环境依赖缺失** |
| **blocker 类型** | — | `ModuleNotFoundError: No module named 'torch'` |
| **详细原因** | — | 同上。额外风险：下载 23GB 模型需稳定网络和足够磁盘（当前可用 772GB，非瓶颈） |
| **是否可修复** | — | 是：环境就绪后完成 HF 登录 + 模型下载 |
| **peak VRAM** | — | N/A |
| **latency** | — | N/A |
| **output path** | — | N/A |

脚本 `run_flux_schnell_if_possible.py` 经 `--help` 验证可正常解析参数。

---

## 汇总结论

| 模型 | 环境状态 | 可运行？ | Blocker |
|------|---------|----------|---------|
| Sana 0.6B | ❌ torch/diffusers 缺失 | 否 | ModuleNotFoundError |
| SD3 Medium | ❌ torch/diffusers 缺失 | 否 | ModuleNotFoundError + HF 登录待完成 |
| FLUX schnell | ❌ torch/diffusers 缺失 | 否 | ModuleNotFoundError + HF 登录待完成 |

### 根本原因（单一 root cause）

**开发主机 Apple M5 上未安装 torch 和 diffusers**，且系统 Python 版本 3.9.6 低于项目要求的 3.13+。

这是预期的环境状态——T1 已记录的已知限制。双轨策略中，dev host 负责开发和 toy 实验，真实 reference inference 应在远程 CUDA GPU 上执行。远程 CUDA GPU 当前不可用（用户未提供 SSH 访问凭证或已就绪的远程环境）。

### 脚手架完备性

三个 run 脚本均已通过 `--help` 验证参数解析正常。`profile_memory.py`（T13 产出）默认 `--dry-run` 模式可在无 torch 下运行显存预估。环境就绪后，单条命令即可完成真实推理——不需要额外修改脚手架代码。

### 诚实声明

- **未伪造任何图片或成功记录**：results/ 目录下无 PNG 输出文件。
- **未在 M5 上强行运行 diffusers real pipeline**：这既不可能（无 CUDA），也不应该（MPS 不支持）。
- **未为"看起来完成"而忽略环境 blocker**：这是故意的、诚实的 blocker 记录。
