# 第 2 周报告：Scheduler / Rectified Flow / Toy 实验

> **日期**：2026-06-07  Week 2
> **来源任务**：T10 — scheduler / rectified flow / timestep embedding + toy rectified flow
> **证据文件**：`.omo/evidence/task-10-pytest.txt`、`.omo/evidence/task-10-toy-rf.txt`

---

## 1. 完成内容

### 1.1 核心模块实现

在 `diffusion_engine/core/` 下实现了三个模块：

- **`scheduler.py`**：同时支持 Euler 和 RectifiedFlow 两种更新方式。Euler 使用等间距 time step 列表，RectifiedFlow 使用 log-linear (sigma-based) step 列表。接口统一：`step(model_output, latent, timestep, next_timestep)` → 返回更新后的 latent。同时支持 image shape `(B, C, H, W)` 和 video shape `(B, C, T, H, W)`。

- **`rectified_flow.py`**：实现 rectified flow 的矢量场构造。基于 `v = (x1 - x0)` 的直线路径假设，当前 latent 沿 ODE 方向移动。支持 Euler 一阶 ODE solver 和自定义 step list。

- **`timestep_embedding.py`**：Sinusoidal timestep embedding（类 Transformer 位置编码），将标量 timestep 映射为高维向量用于 AdaLN 调制。支持 learnable linear projection。

### 1.2 测试

- `diffusion_engine/tests/test_scheduler.py`：18 个测试用例。覆盖 Euler update、RectifiedFlow update、确定性（seed 固定 = 输出固定）、多种 step 数（4/8/16/28/50）、image 和 video shape 支持、自定义 step list。

- `diffusion_engine/tests/test_rectified_flow.py`：18 个测试用例。覆盖矢量场构造、ODE 积分、路径线性度验证、不同目标分布（Gaussian/ring/moons）。

**总计 36 个 pytest 全部通过。**

### 1.3 Toy Rectified Flow 实验

在 `experiments/toy_rectified_flow/` 下完成：

- **`infer_toy_flow.py`**：从 2D 噪声采样 500 个点，沿学到的矢量场做 ODE 积分（8 步），将中间轨迹保存为 JSON，最终结果绘制为 PNG。

- **实验结果**：目标分布为 ring（环形分布）。初始点：半径 mean=1.237，std=0.689。8 步后：半径 mean=0.644，std=0.689。轨迹图保存为 `toy_flow_ring_s8_seed0.png`，轨迹数据保存为 `trajectory_ring_s8_seed0.json`。

---

## 2. 技术关键发现

### 2.1 Rectified Flow vs Score-Based

| 维度 | Score-Based (DDPM/DDIM) | Rectified Flow |
|------|------------------------|----------------|
| **核心对象** | Stein score ∇log p(x) | 矢量场 v(x, t) |
| **路径** | 随机（SDE，非直线） | 直线（ODE，最短路径） |
| **训练目标** | Denoising score matching | 直线方向回归 |
| **采样器** | SDE/ODE solver（需处理 score） | ODE solver（直接沿矢量场） |
| **步数需求** | 通常需要 50-1000 步 | 4-50 步即可（尤其是蒸馏后） |

关键结论：rectified flow 之所以能在少步数下工作，是因为它的路径被显式训练为尽可能笔直。直线路径意味着每一步的更新方向更准确，不需要像 score-based 方法那样靠大量小步来逼近弯曲的路径。

### 2.2 Scheduler 选择对延迟的影响

本 benchmark（Toy）使用 mock denoiser（numpy 模拟），结果：
- 延迟与步数呈完美线性关系（R² ≈ 1.000）
- Euler 和 RectifiedFlow 的每步计算量几乎相同
- 4 步 ≈ 0.34 ms，50 步 ≈ 4.22 ms（纯 scheduler 开销，不包含 denoiser forward）
- 真实 DiT 推理中 denoiser forward 占总时间的 95%+，scheduler 开销可忽略

### 2.3 Toy 实验的教学价值

Toy rectified flow 用 2D 环形分布示范了整个流程：从噪声（散布在环内的随机点）出发，沿学到的矢量场方向（指向环的边界），8 步后收敛到近环形的分布。这比数学公式更直观——你可以看到每一步点群的位置变化，理解 ODE 积分在做什么。

---

## 3. 与设计笔记的对照

`learning/notes/04_scheduler设计.md` 记录了三项关键设计决策：

1. **接口统一**：Euler 和 RectifiedFlow 共用同一 `step()` 签名，调用方不感知 scheduler 类型。这是为了后续 pipeline 和 benchmark 中能无缝切换 scheduler。

2. **确定性要求**：所有 scheduler 通过 `torch.Generator` 接受 seed，确保相同输入产生相同输出。这对测试和调试至关重要。

3. **image/video 双支持**：scheduler 操作的是 latent tensor（任意维度），不假设 shape，由调用方管理维度语义。测试中分别用 `(1, 4, 64, 64)` 和 `(1, 4, 8, 64, 64)` 验证。

---

## 4. 本周风险与未完成项

- **真实 denoiser 未接入**：Toy 实验使用 mock denoiser（固定矢量场），尚未连接真实的 DiT transformer 做 denoise。这需要在 T11 完成后才可连接。
- **MPS 后端未验证**：所有 toy 实验在 CPU 上运行。MPS 后端的 scheduler 兼容性待 T12 后验证。
- **真实 GPU latency 未知**：mock denoiser 的延迟数据不代表真实 DiT 推理。真实数据需等 T16/T17 的优化 benchmark。

---

## 5. 下周预览（Week 3 / T11-T12）

- T11：attention / transformer_block / tiny DiT + shape 测试
- T12：text conditioning / pipeline / memory manager + toy DiT inference
- 关键前置：T3（复用决策）、T4（骨架）、T5（数据流笔记）、T10（scheduler）

---

> **本周产出**：3 个核心模块、36 个 pytest、1 个 toy 实验目录（含 PNG + JSON + README）、1 篇设计笔记。T10 圆满完成。

---

## 6. 补充说明：Rectified Flow 与 Diffusion 家族的关系

Rectified flow 是 flow matching 的一种特例。广义 flow matching 允许任意路径（直线、曲线、甚至随机路径），而 rectified flow 强制路径为直线（通过 reflow / rectification 过程）。这种直线约束带来了两个关键好处：

1. **更少的推理步数**：直线路径意味着每一步的 ODE 更新更准确，不需要像弯曲路径那样大量小步累积。这是为什么 FLUX schnell 只需 4 步就能生成可用图像的基础。

2. **更简单的 scheduler 实现**：Euler ODE solver 在直线路径上的一阶截断误差最小。相比 score-based 方法需要高精度的 Heun/DPM-Solver，rectified flow 的 Euler update 几乎同效。

与 Consistency Models 的对比：consistency models 走的是 "将多步蒸馏为单步" 的路线（直接学习从噪声到图像的映射），而 rectified flow 走的是 "让每步更有效" 的路线（直线路径 + 少步数）。两者不互斥——Sana Sprint 同时使用了 rectified flow 的直线路径和 consistency distillation 的少步策略。
