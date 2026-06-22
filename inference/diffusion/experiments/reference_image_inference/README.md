# 真实文生图 Reference Inference 实验

> **目标**：在 中等显存配置 约束下，用 HuggingFace Diffusers 跑通至少一个真实文生图推理。
> **最低标准**：**只要跑通一个 image reference 即满足最低要求。**
> **负责任务**：T13（脚手架+前置下载说明）+ T14（真实推理尝试）
> **执行环境**：远程 CUDA GPU（中等显存配置）— dev host Mac M5 不支持 CUDA

---

## 1. 目标与最低标准

### 核心目标
1. 验证至少一个现代文生图模型在一块可用的 CUDA GPU 上能否完成推理。
2. 记录成功/失败参数（resolution、steps、dtype、offload、peak VRAM、latency）。
3. 为 `diffusion_engine/`（自研引擎）提供真实对照——对比 diffusers pipeline 与我们的 toy pipeline 在数据流与显存上的差异。
4. 失败时如实记录 blocker（OOM、gated、下载失败、CUDA 不支持），不伪造成功。

### 最低成功标准（一句话）
**只要跑通一个 image reference（任意模型 × 任意分辨率），即满足 T13+T14 的最低交付要求。**

- ✅ 跑通 512×512 也算成功。
- ✅ 跑通 4 步也算成功。
- ❌ 不要为了追踪多模型成功而无限消耗时间。
- ❌ 不要把 diffusers pipeline 的调用当成 `diffusion_engine/` 的最终成果。

---

## 2. 优先级与决策树

> 以下按**可行性**从高到低排列。首先尝试 1，失败再尝试 2，依此类推。

### 决策树

```
是否已安装 torch + diffusers？
 ├── 否 → 安装依赖：uv sync（Python 3.13+），然后重试
 └── 是
      │
      是否在 CUDA 设备上？
      ├── 否（MPS / CPU）→ 记录 blocker：MPS 不支持 diffusers real pipeline，
      │                          切换到远程 CUDA GPU
      └── 是（可用的 CUDA GPU）
           │
           尝试 Sana（首选，开放模型，显存友好）
           ├── 成功 → 🎉 记录结果，任务完成
           └── 失败
                │
                尝试 SD3 Medium（no-T5 + CPU offload）
                ├── 成功 → 🎉 记录结果，任务完成
                └── 失败
                     │
                     尝试 FLUX.1-schnell（4 步，但 VRAM 偏高）
                     ├── 成功 → 🎉 记录结果
                     └── 失败 → 📋 三模型都失败？记录详细 blocker
```

### 2.1 首选：Sana（开放模型，对中低显存更友好）

| 属性 | 值 |
|------|-----|
| **HF ID（首选）** | `Efficient-Large-Model/Sana_600M_1024px_diffusers` |
| **HF ID（备用）** | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` |
| **License** | Apache 2.0 |
| **是否需要 HF token / 接受协议** | **否**（完全开放，无 gated） |
| **是否需要 Python 3.13+** | 是 |
| **资源档位** | ✅ 非常适合（<6 GB VRAM） |
| **推荐 resolution** | 1024×1024（默认），降级 512×512 |
| **推荐 steps** | 20（默认），降级 10 |
| **推荐 dtype** | bf16（CUDA），fp32（MPS/CPU fallback） |
| **推荐 offload** | enable_model_cpu_offload（默认） |
| **推荐 VAE tiling** | enable_vae_tiling（默认） |
| **核心优势** | 单文本编码器 Gemma-2B（仅 4GB），无 T5-XXL 负担；linear attention（O(n) 而非 O(n²)）；Apache 2.0 无限制 |

### 2.2 次选：SD3 Medium（no-T5 路径）

| 属性 | 值 |
|------|-----|
| **HF ID** | `stabilityai/stable-diffusion-3-medium-diffusers` |
| **License** | Stability AI Community License |
| **是否需要 HF token** | **是**：需注册 HF 账号 → 创建 access token → 访问 model card 页面 accept license |
| **是否需要 Python 3.13+** | 是 |
| **资源档位** | ✅ 可行（no-T5 约 4.3GB，含 T5 约 15GB+ 不可行） |
| **推荐 resolution** | 1024×1024（默认），降级 768×768 |
| **推荐 steps** | 28（默认），降级 15 |
| **推荐 dtype** | fp16（默认） |
| **推荐 offload** | enable_model_cpu_offload（默认） |
| **推荐 VAE slicing** | enable_vae_slicing（默认） |
| **核心策略** | 必须关闭 T5：删除 `text_encoder_3` 加载，否则 VRAM 直接爆炸 |

### 2.3 第三选：FLUX.1-schnell

| 属性 | 值 |
|------|-----|
| **HF ID** | `black-forest-labs/FLUX.1-schnell` |
| **License** | Apache 2.0 |
| **是否需要 HF token** | **是**：需注册 HF 账号 → 创建 access token → 访问 model card 页面 accept license |
| **是否需要 Python 3.13+** | 是 |
| **资源档位** | ⚠️ 偏紧（约 10GB，需 sequential offload） |
| **推荐 resolution** | 1024×1024（默认），降级 768×768 |
| **推荐 steps** | 4（schnell 推荐只用 4 步） |
| **推荐 cfg_scale** | 1.0（schnell 不使用 CFG 或很低） |
| **推荐 dtype** | fp16（默认） |
| **推荐 offload** | enable_sequential_cpu_offload（必需） |
| **量化备选** | 社区 GGUF Q4 量化可大幅降低 VRAM；若 fp16 OOM，优先尝试 quantized 路径 |

---

## 3. 手动前置步骤清单

### 3.1 所有模型都需要的步骤

```bash
# 1. 确保 Python 3.13+ 可用
uv python install 3.13

# 2. 创建虚拟环境并安装依赖
cd /path/to/diffusion
uv sync

# 3. 验证 torch + diffusers 可 import
python -c "import torch; print(torch.__version__); import diffusers; print(diffusers.__version__)"
```

### 3.2 仅 gated 模型（SD3 / FLUX）需要的额外步骤

```
步骤 1：注册 Hugging Face 账号
  → https://huggingface.co/join

步骤 2：创建 Access Token
  → https://huggingface.co/settings/tokens
  → 选 "Read" 权限即可
  → 复制 token（hf_xxx...）

步骤 3：本地登录
  → 运行 huggingface-cli login
  → 粘贴 token

步骤 4：访问 Model Card 页面并 Accept License
  → SD3: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
  → FLUX: https://huggingface.co/black-forest-labs/FLUX.1-schnell
  → 页面中点击 "Agree and access repository" 按钮
  → 等待页面刷新确认状态变为 "You have been granted access"

步骤 5：验证访问权限
  → huggingface-cli whoami （确认登录状态）
  → 首次运行脚本时，diffusers 会自动下载模型到 ~/.cache/huggingface/
```

### 3.3 Sana（无 gated）只需步骤 1

Sana 是 Apache 2.0 开放模型，无需 token、无需 accept license。
下载命令：首轮运行时 diffusers 自动下载，或手动：

```bash
# 手动预下载 Sana 0.6B（推荐，避免运行时等）
huggingface-cli download Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --local-dir ~/.cache/huggingface/hub/models--Efficient-Large-Model--Sana_600M_1024px_diffusers
```

---

## 4. 模型下载与磁盘空间预估

| 模型 | HF ID | 下载大小 | 解压后大小 | Gated |
|------|-------|---------|-----------|-------|
| Sana 0.6B | `Efficient-Large-Model/Sana_600M_1024px_diffusers` | ~2 GB | ~3 GB | 否 |
| Sana 1.6B | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | ~4 GB | ~6 GB | 否 |
| SD3 Medium | `stabilityai/stable-diffusion-3-medium-diffusers` | ~5 GB | ~7 GB | 是 |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | ~23 GB | ~25 GB | 是 |

> **注意**：下载缓存默认存放在 `~/.cache/huggingface/`。确保磁盘有足够空间。
> FLUX 下载需约 25GB 空间；SD3 需约 7GB；Sana 需约 3~6GB。

### 手动下载命令（可选，用于预缓存）

```bash
# Sana（首选）
huggingface-cli download Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --local-dir ./models/Sana_600M

# SD3 Medium（需先 accept license + login）
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers \
  --local-dir ./models/SD3_Medium

# FLUX.1-schnell（需先 accept license + login）
huggingface-cli download black-forest-labs/FLUX.1-schnell \
  --local-dir ./models/FLUX_schnell
```

---

## 5. 本目录脚本总览

| 脚本 | 功能 | 默认 dry-run |
|------|------|-------------|
| `run_sana_if_possible.py` | Sana 文生图推理 | 否（尝试跑真实模型） |
| `run_sd3_medium_if_possible.py` | SD3 Medium 文生图推理 | 否 |
| `run_flux_schnell_if_possible.py` | FLUX.1-schnell 文生图推理 | 否 |
| `profile_memory.py` | 显存预测与 profiling | **是**（`--dry-run` 默认 True） |
| `sample_prompts.txt` | 中英双语 prompt 样本 | N/A |

### 快速开始

```bash
# 1. 先看各脚本的帮助信息（了解参数和前置要求）
python run_sana_if_possible.py --help
python run_sd3_medium_if_possible.py --help
python run_flux_schnell_if_possible.py --help

# 2. 用 profile_memory 预估显存（dry-run，安全）
python profile_memory.py --script sana --prompt "一只柴犬在樱花树下" --resolution 1024x1024

# 3. 尝试跑 Sana（首选，无 gated）
python run_sana_if_possible.py --prompt "一只柴犬在樱花树下" --output_dir results

# 4. 若 Sana 失败，尝试 SD3（需先完成 HF login + accept license）
python run_sd3_medium_if_possible.py --prompt "一只柴犬在樱花树下" --no_t5 --output_dir results
```

---

## 6. 失败处理与降级路径

### 6.1 OOM（Out of Memory）

```
OOM 发生 → 逐级降级，每级尝试一次：

Level 1: 降 resolution
  1024×1024 → 768×768 → 512×512

Level 2: 减 steps
  20 → 15 → 10 → 4

Level 3: 启用更激进的 offload
  enable_model_cpu_offload → enable_sequential_cpu_offload

Level 4: 切更小模型
  Sana 1.6B → Sana 0.6B
  SD3 Medium → 尝试 SD3 但等待更小变体
  FLUX schnell → 社区 GGUF Q4 量化路径

连续 3 次 OOM → 记录 blocker，不再调参
```

### 6.2 Gated Repo 拒绝（401 / 403）

```
症状：HFValidationError / RepositoryNotFoundError / 401 Unauthorized

排查步骤：
1. huggingface-cli whoami — 确认已登录
2. 访问 model card 页面 — 确认已点击 "Agree and access repository"
3. 检查 token 权限 — 至少需要 "Read" scope
4. 等待 5 分钟 — HF 权限同步可能有延迟
5. 仍失败 → 记录 blocker：模型 gated，尝试下一个模型
```

### 6.3 torch / diffusers 未安装

```
症状：ModuleNotFoundError: No module named 'torch' / 'diffusers'

解决：
  uv sync  # 从 pyproject.toml 安装所有依赖
  或
  uv pip install torch diffusers transformers accelerate

仍失败 → 记录 blocker：环境依赖缺失，DevOps 任务
```

### 6.4 设备不支持（MPS / CPU）

```
症状：脚本检测到 device != cuda

处理：
  - MPS (Mac M5)：diffusers 真实 pipeline 在 MPS 上可能部分算子缺失或报错。
    不要尝试在 Mac 上跑真实 diffusers 模型——只跑 toy 实验。
  - 远程 CUDA GPU：SSH 过去执行。
  - 记录 blocker：当前设备不支持 CUDA，需远程执行。
```

---

## 7. Dev Host vs 远程 CUDA GPU 双轨策略

| 维度 | Dev Host (Mac M5) | 远程 CUDA GPU |
|------|-------------------|-----------------|
| **用途** | 开发、写代码、跑 toy 实验 | 跑真实 diffusers reference inference |
| **Python** | 3.13+ via uv | 3.13+ via uv |
| **后端** | MPS (Metal 4) | CUDA |
| **可跑什么** | `profile_memory.py --dry-run` | 所有 `run_*_if_possible.py` 脚本 |
| **不可跑什么** | 真实 diffusers 模型（MPS 不支持） | N/A |
| **连接方式** | N/A | SSH |
| **PROFILE_MEMORY** | 在 dev host 上以 `--dry-run` 模式预估 | 在远程上以 `--no-dry-run` 模式实测 |

---

## 8. 结果目录结构（预期）

```
experiments/reference_image_inference/
├── README.md                          # 本文件
├── run_sana_if_possible.py            # Sana 推理脚本
├── run_sd3_medium_if_possible.py      # SD3 Medium 推理脚本
├── run_flux_schnell_if_possible.py    # FLUX.1-schnell 推理脚本
├── profile_memory.py                  # 显存预估与 profiling 工具
├── sample_prompts.txt                 # 中英 prompt 样本
└── results/                           # 输出目录（运行时创建）
    ├── sana_<timestamp>_<seed>.png           # Sana 成功图片
    ├── sd3_medium_<timestamp>_<seed>.png     # SD3 成功图片
    ├── flux_schnell_<timestamp>_<seed>.png   # FLUX 成功图片
    ├── <name>_profile.json                   # profile_memory 输出
    └── blocker_<model>.md                    # 失败时记录
```

---

## 9. 失败 / Blocker 记录模板

如果任何模型在 中等显存配置 约束下无法完成推理，在 `results/blocker_<model>.md` 中记录：

```markdown
# Reference Image Inference — Blocker: <模型名>

**日期**：YYYY-MM-DD
**模型**：<HF model ID>
**设备**：可用的 CUDA GPU（中等显存配置）

## 尝试配置
- dtype: fp16 / fp32 / bf16
- offload: model_cpu_offload / sequential_cpu_offload / 关闭
- vae_slicing: yes / no
- vae_tiling: yes / no
- resolution: W×H
- steps: N, cfg_scale: X.X

## 错误类型
[OOM / CUDA error / 401 Unauthorized / 模型加载失败 / MPS 不支持 / ...]

## 峰值 VRAM
（nvidia-smi 或 torch.cuda.max_memory_allocated() 读数）

## 降级尝试
1. 降 resolution → [结果]
2. 减 steps → [结果]
3. 开 offload → [结果]
4. 换模型 → [结果]

## 结论
[可复现 / 硬件限制 / 需更大显存 / gated 未授权 / ...]
```

---

## 10. 安全说明

- **不要**在脚本中硬编码 HF token。
- **不要**把 token 提交到 git 仓库。
- **不要**在 `--help` 或 README 中写"开箱即跑"——gated 模型需要手动授权。
- **不要**把 diffusers pipeline 的调用包装成 `diffusion_engine/` 的成果。
- `diffusers` 在本目录中的作用是 **reference**（对照），不是我们的引擎。

---

## 11. 与 diffusion_engine 的关系

| 维度 | diffusers（本目录） | diffusion_engine/（自研） |
|------|---------------------|--------------------------|
| **角色** | Reference（对照） | 学习实现（产物） |
| **模型** | 官方预训练权重 | Toy 规模（TinyDiT） |
| **文本编码器** | CLIP / T5 / Gemma（预训练） | ToyTextConditioner（随机映射） |
| **VAE** | 官方 SDXL/AE VAE | ToyVAE（3 层 Conv2d） |
| **用途** | 验证 资源档位、记录真实参数 | 理解原理、学习架构 |

---

## 12. 参考

- HF 模型页面：
  - Sana: <https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers>
  - SD3 Medium: <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>
  - FLUX.1-schnell: <https://huggingface.co/black-forest-labs/FLUX.1-schnell>
- Diffusers pipeline 文档：<https://huggingface.co/docs/diffusers>
- 本任务计划：`.omo/plans/modern-diffusion-inference-roadmap.md` T13–T14
- 论文卡片：
  - `learning/papers/01_scaling_rectified_flow_transformers_sd3.md`
  - `learning/papers/02_flux_architecture_notes.md`
  - `learning/papers/03_sana.md`
