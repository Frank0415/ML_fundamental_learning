# Toy DiT Inference 实验

> **状态**：T12 完成 - 完整 pipeline demo 已实现。
> **优先级**：P0（Wave 3 首个端到端实验）
> **负责任务**：T12 - text conditioning / pipeline / memory manager + toy DiT inference

---

## 实验目的

用最小 DiT（2 层、hidden=64、patch_size=2）跑完整的 denoising-to-decoding pipeline，验证 diffusion_engine 的以下模块能端到端协同：

| 模块 | 文件 | 验证目标 |
|------|------|---------|
| **Scheduler** | `diffusion_engine/core/scheduler.py` | RectifiedFlow 的 timestep 序列与 Euler step |
| **TinyDiT** | `diffusion_engine/core/dit.py` | Patchify → AdaLN modulation → Unpatchify 全流程 |
| **Text Conditioner** | `diffusion_engine/core/text_conditioning.py` | Prompt embedding + cache 机制 |
| **Memory Manager** | `diffusion_engine/core/memory_manager.py` | Latent buffer 预分配 + 显存统计 |
| **VAE Decoder** | `diffusion_engine/core/vae_stub.py` | Toy ConvTranspose2d decode（8x upsample） |
| **Pipeline** | `diffusion_engine/core/pipeline.py` | 6 步主循环：encode → init → loop → decode |

**期望产出**：
- 一张 64×64 RGB 图片（内容不要求美观，但需通过数值检查：无 NaN、无 inf、值域合理）
- 一次完整推理的显存统计 snapshot

---

## 与 toy_rectified_flow 实验的差异

| 维度 | toy_rectified_flow（T10） | toy_dit_inference（T12） |
|------|--------------------------|-------------------------|
| **框架** | 纯 numpy，零 torch 依赖 | torch nn.Module 全栈 |
| **数据类型** | 2D 点云 (N, 2) | 图像 latent (B, 4, H/8, W/8) |
| **向量场** | 人工解析函数 v(x, t) | TinyDiT 神经网络 v_θ(x, t, c) |
| **条件注入** | 无（纯无条件 flow） | 文本条件（toy random embedding） |
| **调度器** | 直接调用 rectified_flow_sample() | Pipeline 内聚 RectifiedFlowScheduler |
| **解码** | 无需解码（点云即输出） | ToyVAE 8x upsample → RGB image |
| **CFG** | 无 | Batched / Sequential 双模式 |
| **Cache** | 无 | Prompt embedding cache + latent ping-pong buffer |
| **产出** | 2D 散点图 + JSON 轨迹 | RGB 图片 + 显存统计 JSON |

**相同之处**：都使用 rectified flow ODE（Euler 积分，t ∈ [1, 0]），都是确定性推理。

---

## 运行命令

### 前置条件

```bash
# 确认在 .venv 中
source .venv/bin/activate

# 确认 torch 已安装（若未安装则记录 blocker）
python -c "import torch; print(torch.__version__)"
```

### 4 步快速 smoke test

```bash
cd /path/to/diffusion
python experiments/toy_dit_inference/infer_tiny_dit.py \
    --prompt "a cat" --num_steps 4 --seed 0
```

### 标准运行（28 步，SD3 默认步数）

```bash
python experiments/toy_dit_inference/infer_tiny_dit.py \
    --prompt "a cat sitting on a chair" \
    --num_steps 28 --cfg_scale 7.5 --seed 0
```

### Sequential CFG（显存更低）

```bash
python experiments/toy_dit_inference/infer_tiny_dit.py \
    --prompt "a cat" --mode sequential --num_steps 4
```

### 启用 memory profiling

```bash
python experiments/toy_dit_inference/infer_tiny_dit.py \
    --prompt "a cat" --num_steps 4 --profile
```

### 查看帮助

```bash
python experiments/toy_dit_inference/infer_tiny_dit.py --help
```

---

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt` | str | "a cat sitting on a chair" | 正向文本提示 |
| `--negative_prompt` | str | "" | 负向文本提示（默认空，无负面引导） |
| `--num_steps` | int | 28 | ODE 去噪步数。步数越多，质量越高但越慢。SD3 常用 28。 |
| `--cfg_scale` | float | 7.5 | CFG 引导强度。1.0 为无引导，> 1 为有引导。典型值 4-10。 |
| `--height` | int | 64 | 输出图像高度（像素，需为 8 的整数倍） |
| `--width` | int | 64 | 输出图像宽度（像素，需为 8 的整数倍） |
| `--seed` | int | 0 | 随机种子（保证确定性输出） |
| `--mode` | str | batched | CFG 模式：`batched`（一次 forward，显存高）或 `sequential`（两次 forward，显存低） |
| `--output_dir` | str | results/ | 输出目录（自动创建） |
| `--device` | str | cpu | 计算设备：`cpu` / `mps` / `cuda` |
| `--profile` | flag | False | 启用 memory profiling 记录显存统计 |

---

## 结果目录结构

运行后 `results/` 目录下生成：

```
results/
├── tiny_dit_<prompt>_s<步数>_cfg<引导>_seed<种子>.png   # 生成的 RGB 图片
├── results_summary.json                                   # 运行摘要（JSON）
└── blocker_toy_dit_inference.md                           # 仅在 blocker 时出现
```

### 如何判断成功

1. **图片非全黑/全白**：值域应在 [0, 1] 附近（或经 clamp 后可显示）。Toy 模型无训练，像素值可能偏灰/偏随机，但不应全零或全 NaN。
2. **数值健康**：图片无 NaN、无 inf、mean 和 std 在合理范围。
3. **确定性**：相同 `--seed` 两次运行产生 bit-exact 相同输出（纯确定性 ODE + 固定 embedding）。
4. **Cache 统计**：日志中 `hits` 和 `misses` 统计合理（首次运行 `hits=0, misses=1`，第二次相同 prompt 运行 `hits=1`）。

---

## 内部流程说明

### Pipeline 6 步

```
1. Encode prompt → cond/uncond embedding（ToyTextConditioner，含 cache）
2. Init latent → 用 --seed 生成噪声 (1, 4, H/8, W/8)
3. Scheduler timesteps → RectifiedFlowScheduler 生成 t ∈ [1.0, 0.0] 序列
4. Denoising loop（num_steps 次）:
   a. TinyDiT forward（batched 或 sequential CFG）
   b. CFG in vector field: v_cfg = v_uncond + s × (v_cond - v_uncond)
   c. Euler step: latents = latents + dt × v_cfg
5. Decode → ToyVAE.decode(latents) → (1, 3, H, W) RGB image
```

### CFG 在 vector field 层面

**正确做法**（本实现采用）：
```
v_cfg = v_uncond + cfg_scale × (v_cond - v_uncond)
latents = latents + dt × v_cfg
```

**错误做法**（不要在 latent 层面做 CFG！）：
```
# ❌ 不要在 latent 上插值
x_cfg = x_uncond + cfg_scale × (x_cond - x_uncond)
```

详见 `learning/notes/06_cfg和negative_prompt.md`。

### Batched vs Sequential CFG

| 维度 | Batched | Sequential |
|------|---------|------------|
| Forward 次数 | 1 次（batch × 2） | 2 次（单 batch） |
| 显存占用 | 高（2× latent batch） | 低（单 batch） |
| 速度 | 快 | 慢（2× forward） |
| 数值结果 | 相同（确定性 ODE + 相同 model） | 相同 |

---

## 前置依赖

- Python ≥ 3.13
- torch ≥ 2.7（核心依赖，未安装则记录 blocker）
- numpy（必需）
- pytest（仅测试需要）
- torchvision 或 Pillow（可选，用于保存图片；不可用时 fallback 到原始 tensor）

**不需要**：diffusers、transformers、CUDA/GPU（toy 规模极小，CPU 可运行）。

---

## 失败/Blocker 模板

如果本实验在任何环节失败，`results/blocker_toy_dit_inference.md` 会自动生成，包含：

- 阻塞环节描述
- 运行配置
- 错误日志
- 建议修复方案

### 常见 blocker

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `ModuleNotFoundError: No module named 'torch'` | torch 未安装 | `uv pip install torch` |
| `ImportError: diffusion_engine.core` | import path 不正确 | 确保从 `diffusion/` 根目录运行 |
| NaN 在输出中 | AdaLN 调制数值不稳定（toy 模型可能） | 检查 DiTBlock 初始化，或使用 `torch.autograd.detect_anomaly()` |
| MPS 不可用 | Mac 上 torch 未编译 MPS 后端 | 使用 `--device cpu` |

---

## 参考

- T12 计划：`.omo/plans/modern-diffusion-inference-roadmap.md` T12 章节
- T10 实验：`experiments/toy_rectified_flow/README.md` - 对比参考
- 学习笔记：
  - `learning/notes/06_cfg和negative_prompt.md` - CFG 原理与实现
  - `learning/notes/07_text_encoder和prompt_embedding_cache.md` - cache 设计
  - `learning/notes/08_latent_buffer和显存预算.md` - memory budget
- 论文：[2403.03206] Scaling Rectified Flow Transformers (SD3)
- 核心模块：`diffusion_engine/core/pipeline.py`、`text_conditioning.py`、`memory_manager.py`、`vae_stub.py`
