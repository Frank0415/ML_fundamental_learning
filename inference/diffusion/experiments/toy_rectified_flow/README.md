# Toy Rectified Flow 实验

> **状态**：T10 完成，可运行。
> **优先级**：P0（Wave 2 首个可运行实验）
> **负责任务**：T10 - scheduler / rectified flow / timestep embedding + toy rectified flow

---

## 实验目的

在完全可控的 toy 场景（2D 分布间概率传输）上验证 `diffusion_engine/core/` 三个模块的正确性：

1. **Rectified Flow ODE**（`diffusion_engine/core/rectified_flow.py`）：验证 rectified flow 的 Euler ODE 积分能否将 2D 高斯噪声沿向量场推送到目标分布。
2. **Scheduler**（`diffusion_engine/core/scheduler.py`）：验证 EulerScheduler 和 RectifiedFlowScheduler 在 t 空间和 sigma 空间的 step 行为一致。
3. **Timestep Embedding**（`diffusion_engine/core/timestep_embedding.py`）：验证 sinusoidal embedding 的 shape 和数值范围。

**核心思想**：不训练任何模型，使用**人工设计的合成向量场**代替学习到的 v_θ(x, t)，观察 ODE 轨迹的行为。这使我们可以在零训练成本下验证调度器和 ODE 积分器的正确性。

**期望产出**：
- 一张 2D 散点图，展示从高斯噪声出发沿 rectified flow ODE 积分到达目标分布（圆环/原点/双中心/螺旋）的轨迹。
- 轨迹 JSON 数据，可供后续定量分析。

---

## 运行命令

### 快速验证（8 步，ring 目标）

```bash
cd /path/to/diffusion
python experiments/toy_rectified_flow/infer_toy_flow.py \
    --num_steps 8 --seed 0 --target_type ring \
    --output_dir experiments/toy_rectified_flow/results
```

### 标准运行（28 步，SD3 默认步数）

```bash
python experiments/toy_rectified_flow/infer_toy_flow.py \
    --num_steps 28 --seed 0 --target_type ring
```

### 查看帮助

```bash
python experiments/toy_rectified_flow/infer_toy_flow.py --help
```

### 其他目标分布

```bash
# 原点吸引（最简单的"去噪"）
python experiments/toy_rectified_flow/infer_toy_flow.py --target_type origin

# 双中心（演示分布分裂）
python experiments/toy_rectified_flow/infer_toy_flow.py --target_type dual_center

# 螺旋（演示非保守向量场）
python experiments/toy_rectified_flow/infer_toy_flow.py --target_type spiral --num_samples 1000
```

### 单独绘图（已有 JSON 时）

```bash
python experiments/toy_rectified_flow/plot_trajectory.py \
    experiments/toy_rectified_flow/results/trajectory_ring_s28_seed0.json \
    experiments/toy_rectified_flow/results/my_plot.png \
    20
```

---

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_steps` | int | 28 | ODE 积分步数。步数越多，轨迹越精细，越接近真实 ODE 解。SD3 常用 28。 |
| `--seed` | int | 0 | 随机种子。相同 seed 保证完全可复现（确定性 ODE + 确定性向量场）。 |
| `--dim` | int | 2 | 数据维度。当前仅 2D 支持可视化。 |
| `--num_samples` | int | 500 | 初始噪声点数量。越多越能展示分布全貌。 |
| `--target_type` | str | ring | 目标分布类型：`ring`（圆环）、`origin`（原点）、`dual_center`（双中心）、`spiral`（螺旋）。 |
| `--output_dir` | str | results/ | 输出目录，自动创建。 |
| `--record_every` | int | 1 | 每多少步记录一次轨迹（用于 JSON）。默认每步记录。 |

---

## 结果解释

### 输出文件

运行后 `results/` 目录下生成：

```
results/
├── trajectory_<target>_s<步数>_seed<种子>.json  # 完整轨迹数据
├── toy_flow_<target>_s<步数>_seed<种子>.png      # 分布快照图
├── toy_flow_<target>_s<步数>_seed<种子>_trajectories.png  # 单独轨迹线图
└── results_summary.md                              # 文本摘要
```

### 如何判断成功

1. **收敛性**：在 ring/origin 目标下，最终分布应紧贴目标流形（ring 上所有点 r≈3.0，origin 上所有点 ≈ (0,0)）。
2. **确定性**：使用相同 `--seed` 运行两次产生完全相同的输出文件（bit-exact，因为纯 numpy 确定性 ODE）。
3. **轨迹光滑**：ODE 积分路径应是光滑曲线，不出现跳跃或折线（步数少时折线是预期行为）。
4. **力场强度时间依赖**：t=1 附近力较弱（点缓慢移动），t→0 时力增强（点快速收敛）。

### 与 SD3 真实推理的对应关系

| Toy 实验 | 真实 SD3 推理 |
|---------|-------------|
| 合成向量场 v(x,t) | MMDiT 模型 v_θ(x_t, t, c) |
| 2D 点 (500×2) | 图像 latent (1×4×64×64) |
| Euler ODE (28 步) | Euler ODE (28 步) |
| t∈[1,0] 线性 schedule | t∈[1,0] 线性 schedule |
| 确定性 ODE | 确定性 ODE（CFG 仍保持确定性） |

**关键差异**：toy 向量场是人工设计的解析函数，不依赖文本条件；真实推理中 v_θ 由 MMDiT 神经网络计算，包含 text prompt 条件注入。

---

## 前置依赖

- Python ≥ 3.13
- numpy（必需）
- matplotlib（可选，用于绘图；如不可用，JSON 仍会生成）
- pytest（仅测试需要）

**不需要**：torch、diffusers、transformers、CUDA/GPU。

---

## 参考

- T5 笔记：`learning/notes/03_diffusion推理数据流.md` - scheduler 的 timestep 约定
- 计划详情：`.omo/plans/modern-diffusion-inference-roadmap.md` T10 章节
- 论文：[2403.03206] Scaling Rectified Flow Transformers for High-Resolution Image Synthesis（SD3 论文）
- 核心模块：`diffusion_engine/core/scheduler.py`、`rectified_flow.py`、`timestep_embedding.py`
