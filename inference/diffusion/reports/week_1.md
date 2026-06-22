# 第 1 周报告：前置环境验证与风险结论

> **日期**：2026-06-06  Week 1
> **来源任务**：T1 — 前置环境与风险验证
> **证据文件**：`.omo/evidence/task-1-env.txt`、`.omo/evidence/task-1-hf-check.txt`

---

## 1. 实际硬件环境

| 项目 | 实际值 | 计划值 | 结论 |
|------|--------|--------|------|
| **主机** | MacBook (franksAir.local) | — | — |
| **芯片** | Apple M5 | NVIDIA 可用的 CUDA GPU (Blackwell) | **关键偏差** |
| **GPU API** | Metal 4 | CUDA 12.x+ | **关键偏差** |
| **NVIDIA GPU** | 无（`nvidia-smi` 不可用） | 中档 CUDA 单卡 | **关键偏差** |
| **macOS** | 26.5 (Build 25F71) | — | 最新正式版 |
| **系统 Python** | 3.9.6 (`/usr/bin/python3`) | 3.13 | 不可用于 diffusion 项目 |
| **uv** | 0.11.17 (aarch64-apple-darwin) | — | 可用于管理 Python 3.13 |
| **PyTorch** | 未安装 | — | 需后续安装 |
| **磁盘 (diffusion/)** | 772 GiB 可用 (926 GiB 总量, 14% 使用) | — | 充足 |
| **HuggingFace** | 200 OK (可达) | — | 连通性正常 |

---

## 2. 与原计划的偏差说明 — 这是最关键的发现

**原计划里默认的是一张中档 CUDA 单卡，但当前开发宿主机是 Apple Silicon M5（Metal 4），没有 NVIDIA GPU。**

这意味着：

1. **本地开发的 GPU 后端截然不同**：M5 使用 Metal Performance Shaders (MPS) 而非 CUDA。PyTorch 在 Apple Silicon 上通过 `torch.device("mps")` 使用 GPU，但 MPS 后端的算子覆盖度、性能特征和显存管理方式与 CUDA 完全不同。

2. **M5 的统一内存（UMA）不等于独立 VRAM**：M5 的 CPU 和 GPU 共享同一片物理内存池，对于 PyTorch MPS 后端，峰值可用"显存"取决于系统总内存和 macOS 动态分配策略，而不是一块独立显卡上的固定显存。

3. **"中档单卡预算"的适用性变化**：
   - 若用户确认在远程 CUDA GPU 上运行真实 reference inference，按 85% 利用率做保守估算仍然成立。
   - 若在 M5 本地运行，需根据 M5 具体配置（统一内存大小）重新计算有效预算，且 MPS 后端的实际可用内存需实测。

4. **CUDA 专属特性完全缺失**：FlashAttention CUDA kernel、xformers、torch.compile CUDA 后端、fp8 硬件支持等均不可用，必须走纯 PyTorch 或 MPS 兼容路径。

**建议**：本报告的 fallback 策略（见第 8 节）将明确区分"本地 M5 玩具实验路径"与"远程 CUDA GPU 真实推理路径"。

---

## 3. Python 3.13 的获取路径

### 当前状态

- 系统 Python：`/usr/bin/python3` → Python 3.9.6（过于陈旧）
- `python3.13`：未在 PATH 中找到
- 旧引擎 `minivLLM/pyproject.toml` 声明 `requires-python = ">=3.13"`，`.python-version` 指定 `3.13`

### 推荐获取方式

`uv` (0.11.17) 已安装在 `/Users/franksair/.local/bin/uv`，可直接管理 Python 版本：

```bash
# 安装 Python 3.13（uv 自动下载）
uv python install 3.13

# 在 diffusion/ 项目根目录创建独立虚拟环境
uv venv --python 3.13 .venv

# 激活后安装依赖
uv pip install torch torchvision torchaudio
uv pip install diffusers transformers accelerate
```

**优势**：`uv` 全自动管理 Python 二进制，无需手动编译或 pyenv，且与旧引擎的 `.python-version` 约定一致。

**风险点**：
- 旧引擎 `minivLLM/` 也声明需要 Python ≥3.13，但新 diffusion 项目必须使用**独立**虚拟环境，不以任何方式依赖 `minivLLM/` 的依赖配置。
- macOS Apple Silicon 上某些 PyTorch 算子（特别是与 CUDA 强绑定的自定义 kernel）可能在 MPS 后端的 nightly build 中仍有不完整覆盖。

---

## 4. 磁盘空间结论

```
Filesystem      Size    Used   Avail Capacity iused ifree %iused  Mounted on
/dev/disk3s5   926Gi   121Gi   772Gi    14%    1.3M  8.1G    0%   /System/Volumes/Data
```

- **可用空间**：772 GiB → **充足**。
- **预期消耗**（估算）：
  - Python 3.13 + PyTorch + Diffusers 环境：~5-8 GiB
  - 单个图像 diffusion 模型（如 SD3 Medium、FLUX schnell）：~5-15 GiB
  - 单个视频 diffusion 模型（如 CogVideoX-2B、LTX-Video）：~5-10 GiB
  - 实验产出（图片、视频、日志）：~2-5 GiB
  - **合计估计**：~30-50 GiB，远在 772 GiB 可用空间内。

**结论**：磁盘空间不构成任何 blocker。

---

## 5. HF 连通性结论

```
HuggingFace status: 200
```

- **HuggingFace Hub 在当前网络可达**，HTTP 200 响应正常。
- 未测试大文件下载速度，但基础连通性已验证通过。
- **无代理/镜像要求**（至少未触发 403/5xx/连接超时）。

---

## 6. Gated 模型的前置清单（license / 手动操作）

以下 5 个模型是计划中 reference image/video inference 的关键候选。**所有模型都需要用户在 HF 上手动接受协议或申请访问，且需要 HF token 才能下载。不完成前置步骤则无法运行。**

| 序号 | 模型 | HF Repo | Gated? | 所需操作 | 对 T13/T14/T15 的影响 |
|------|------|---------|--------|----------|----------------------|
| 1 | **SD3 Medium** | `stabilityai/stable-diffusion-3-medium-diffusers` | **是** | 在 HF 上登录 → 访问模型页 → 点击"Agree and access repository" → 签署 Stability AI 社区许可 | 图像 reference 候选之一 (T13/T14) |
| 2 | **FLUX.1-schnell** | `black-forest-labs/FLUX.1-schnell` | **是** | 在 HF 上登录 → 访问模型页 → 接受 Apache 2.0 许可下的 gate 条款 | 图像 reference 候选之一 (T13/T14) |
| 3 | **Wan2.1-T2V-1.3B** | `Wan-AI/Wan2.1-T2V-1.3B` | **是** | 登录 HF → 访问模型页 → 同意使用条款 | 视频 reference 候选之一 (T15) |
| 4 | **LTX-Video** | `Lightricks/LTX-Video` | **是** | 登录 HF → 访问模型页 → 同意 Lightricks 社区许可 | 视频 reference 首选 (T15) |
| 5 | **CogVideoX-2B** | `THUDM/CogVideoX-2b` | **是** | 登录 HF → 访问模型页 → 同意 THUDM 许可协议 | 视频 reference 候选之一 (T15) |

**HF Token 设置**：

```bash
# 在终端设置（推荐）
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"

# 或在 Python 中
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxxxxx")
```

**若用户未完成上述操作**：
- T13 脚手架脚本应检测 HF token 是否存在，缺失则输出明确指引并退出（不静默失败）。
- T14/T15 真实 inference 尝试应以 `HF_TOKEN 未设置 / 模型未授权` 作为合法的 blocker 记录，而非"运行失败"。

> **注**：Sana-0.6B（`Efficient-Large-Model/Sana_600M_1024px_diffusers`）本身是开放模型，gated 要求较低，但也建议在 HF 上登录以避免速率限制。

---

## 7. 中档单卡的有效预算公式

即使在远程 CUDA GPU 上，也不该按满额显存去计划：

```
有效预算 = 标称显存 × 0.85
```

**按 85% 做保守估算的理由**：
- OS/驱动预留约 5%~8%
- PyTorch CUDA context + allocator 预留约 3%~5%
- 推理峰值通常超出"模型参数内存"的 20%~30%（因 activation、中间张量、VAE decode 等）

**超过 10.2 GB 的降级策略**（优先级从高到低）：
1. **降低 resolution**：512×512 → 384×384 或 256×256
2. **减少推理步数**：28 steps → 8~12 steps（配合 few-step / distilled 模型）
3. **降低 dtype**：fp32 → fp16/bf16（建议默认 bf16 或 fp16）
4. **CPU offload**：`enable_model_cpu_offload()`（牺牲 latency）
5. **VAE tiling/slicing**：`enable_vae_tiling()` + `enable_vae_slicing()`
6. **文本编码器 offload**：T5-XXL → T5-small 或跳过 T5
7. **blocker 记录**：若以上全部无效，记录 OOM 为 blocker，不无限调参

---

## 8. Fallback 策略：M5 统一内存 vs 远程 CUDA GPU 双轨

### 双轨策略概述

由于实际开发环境（M5 Metal）与计划目标硬件（可用的 CUDA GPU CUDA）严重不匹配，必须采用双轨策略：

| 轨道 | 硬件 | 用途 | 可运行内容 |
|------|------|------|-----------|
| **轨道 A：本地 M5** | Apple M5 + Metal 4 (MPS) | toy 实验、小规模验证、开发与测试 | toy rectified flow (T10)、toy DiT inference (T12)、scheduler/attention/pipeline 单元测试 |
| **轨道 B：远程 CUDA GPU** | NVIDIA 可用的 CUDA GPU (CUDA) | 真实 reference inference、性能 profiling | SD3 Medium / FLUX schnell / CogVideoX-2B / LTX-Video 等真实模型推理 (T14/T15)、优化实验 (T16/T17) |

### 轨道 A：M5 本地（toy 实验路径）

- **优势**：零延迟开发反馈、无需网络传输、统一内存利于大 latent buffer 实验
- **限制**：MPS 后端不兼容 CUDA 专属算子、真实 diffusers pipeline 部分组件可能需要 CUDA 兼容性适配
- **M5 统一内存预算**：
  - 若 M5 为 16 GB 统一内存配置，扣除 macOS 开销后 PyTorch 可用约 10-12 GB
  - 若 M5 为 24/32 GB 统一内存配置，基本不构成瓶颈
  - **需用户确认 M5 实际统一内存大小**才能在报告中填写具体预算

**T10+ 在 M5 上的预期**：
- T10 (toy rectified flow)：完全可在 M5 上运行，纯 PyTorch 无 CUDA 依赖
- T11 (toy DiT)：完全可在 M5 上运行，纯 PyTorch 无 CUDA 依赖
- T12 (toy DiT inference with pipeline)：完全可在 M5 上运行
- T16 (prompt cache / latent buffer / scheduler benchmark)：可在 M5 上运行 toy 规模

### 轨道 B：远程 CUDA GPU（真实推理路径）

- **优势**：CUDA 完整生态、FlashAttention、xformers、torch.compile 全功能可用
- **前提条件**：用户需在远程机器上配置好 Python 3.13 + PyTorch CUDA + diffusers 环境，并解决 HF token 与模型下载
- **不可跳过**：T14 (真实 image reference) 和 T15 (视频 reference) 的最终执行必须在 可用的 CUDA GPU 或等效 CUDA 设备上完成

### 用户确认清单

在进入 T10 之前，请用户确认以下事项：

- [ ] M5 的统一内存大小是多少？（16/24/32/其他 GB？）
- [ ] 远程 CUDA GPU 是否已可用？若不可用，何时可用？
- [ ] 远程机器上的 HF token 是否已配置？
- [ ] 远程机器的 Python/CUDA/PyTorch 版本是否就绪？
- [ ] T14/T15 是否接受在 M5 上尝试（以 MPS 后端，可能在部分模型上失败或性能极差）？

---

## 9. 对 T10+ 任务的明确执行建议

### 总体原则

```
Toy 实验 (T10/T11/T12)     → M5 本地直接跑
单元测试 (全部 *_test*.py)  → M5 本地直接跑
优化实验 (T16/T17 toy 规模) → M5 本地直接跑
真实 image 推理 (T14)       → 优先远程 CUDA GPU，次选 M5 + blocker 记录
真实 video 推理 (T15)       → 仅远程 CUDA GPU（M5 MPS 后端视频模型支持度未知）
```

### T10: scheduler / rectified flow / toy rectified flow

- **可在 M5 本地执行**：所有代码纯 PyTorch（`nn.Module`、ODE 求解、trajectory plotting），不依赖 CUDA 或 Diffusers
- `scheduler.py` 的 Euler 和 rectified flow update 完全与设备无关
- 建议使用 `torch.device("mps")` 编译并测试，确保 MPS 后端 shape 兼容

### T11: attention / transformer_block / tiny DiT

- **可在 M5 本地执行**：自注意力、cross-attention、patchify/unpatchify 均为纯 PyTorch
- 注意：如果后续需要使用 FlashAttention 做性能优化，MPS 不支持，需退化到标准 `scaled_dot_product_attention`（PyTorch 内置 SDPA 在 MPS 上可用）
- text conditioning 最小接口可在 M5 上开发和测试

### T12: text conditioning / pipeline / memory manager / toy DiT inference

- **可在 M5 本地执行**：pipeline 主循环是组装已有模块，不引入新硬件依赖
- `memory_manager.py` 的显存统计在 MPS 后端上行为与 CUDA 不同（MPS 没有 `torch.cuda.memory_stats()` 等 API），需写 MPS 适配版本或默认用 `torch.mps` 对应 API

### T13: reference image inference 脚手架

- **不可在 T13 下载模型**（MUST NOT），但可编写和验证脚本结构
- `--help`、参数解析、`torch.device` 自动检测（cuda > mps > cpu）可在 M5 上完成
- **注意**：`run_sd3_medium_if_possible.py` 等脚本应在 `--help` 中明确标注"需远程 CUDA GPU 或等效 CUDA 设备"

### T14: 真实 reference 文生图尝试

- **强烈建议在远程 CUDA GPU 上执行**
- 若用户坚持在 M5 上尝试，必须使用 `torch.device("mps")`，并在结果中明确记录"MPS 后端，非 CUDA，性能不可比"
- 若因 MPS 不兼容导致失败，应记录为 blocker（类别：`MPS_UNSUPPORTED`）而非"模型无法运行"

### T15: 视频 reference 脚手架与尝试

- **必须在远程 CUDA GPU 上执行**：视频模型（特别是 CogVideoX 和 Wan）对 CUDA 依赖较深，MPS 后端大概率不支持
- LTX-Video 相对轻量，可能是 M5 上唯一可尝试的视频模型，但仍需实际验证

### T16/T17: 优化实验

- **toy 规模可在 M5 本地执行**：prompt cache、latent buffer manager、scheduler benchmark 的 toy 版本
- **真实规模对照必须在一块可用的 CUDA GPU 上执行**：CFG batching latency/VRAM 对比、attention memory benchmark 的真实 activation 大小、VAE tiling 的真实效果
- 优化结论应标注执行环境（MPS vs CUDA），避免误导

---

## 10. 环境验证命令速查

以下命令的原始输出已保存到 `.omo/evidence/task-1-env.txt`：

```bash
python3 --version          # Python 3.9.6 (system, not usable)
uv --version               # uv 0.11.17
sw_vers                    # macOS 26.5
system_profiler SPDisplaysDataType | grep -E "Chipset|Vendor|Metal"  # Apple M5, Metal 4
df -h .                    # 772 GiB available
nvidia-smi                 # command not found (expected)
```

HF 连通性证据已保存到 `.omo/evidence/task-1-hf-check.txt`。

---

## 11. 旧引擎环境状态（仅供审计参考）

| 项目 | 状态 |
|------|------|
| `minivLLM/.venv` | 不存在 — 虚拟环境未创建 |
| `minivLLM/pyproject.toml` | 存在，声明 `requires-python = ">=3.13"`, 依赖 `torch>=2.11.0`, `transformers>=5.8.0`, `huggingface-hub>=1.14.0` |
| `minivLLM/.python-version` | 内容 `3.13` |

**结论**：旧引擎为 Qwen3 LLM 推理引擎，与 diffusion 项目弱相关。新 diffusion 项目将使用**独立的** Python 3.13 虚拟环境（通过 `uv` 管理），不以任何方式依赖 `minivLLM/` 的依赖配置。

---

## 12. 环境就绪判定

| 检查项 | 状态 | 待办 |
|--------|------|------|
| Python 3.13 可用 | ❌ 未安装 | `uv python install 3.13` |
| PyTorch (MPS) 可用 | ❌ 未安装 | `uv pip install torch` |
| HuggingFace 可达 | ✅ 200 OK | — |
| 磁盘充足 | ✅ 772 GiB | — |
| NVIDIA/CUDA 可用 | ❌ 无 | 需远程 CUDA GPU |
| 旧引擎虚拟环境 | ❌ 未创建 | 不需要（新项目独立） |

---

## 13. 风险登记表

| 风险 ID | 描述 | 严重度 | 缓解措施 |
|---------|------|--------|---------|
| R1 | 开发环境为 M5 Metal 而非 可用的 CUDA GPU CUDA | **高** | 双轨策略：toy 实验在 M5，真实推理在远程 CUDA GPU |
| R2 | M5 MPS 后端可能不支持部分 diffusers pipeline 组件 | **中** | T14 若在 M5 执行，需有 MPS_UNSUPPORTED blocker 分类 |
| R3 | Python 3.13 与 PyTorch diffusers 的兼容性未知 | **低** | `uv` 可选择安装不同 Python 版本作为 fallback |
| R4 | 远程 CUDA GPU 的可用性未知 | **中** | 需用户确认；若不可用，M5 toy-only 路线仍需完成核心任务 |
| R5 | Gated 模型需要用户手动接受 license | **低** | 已在报告第 6 节列出清单，T13 脚本将检测并引导 |

---

> **下一步**：等待用户确认第 8 节 M5 统一内存大小 + 远程 CUDA GPU 可用性，然后推进 T2（旧引擎审计）。
