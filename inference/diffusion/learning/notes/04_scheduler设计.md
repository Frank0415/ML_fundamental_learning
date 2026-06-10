# scheduler 设计笔记 — Euler 与 Rectified Flow 调度器

> **对应任务**：T10
> **产出日期**：2026-06-07
> **前置阅读**：`learning/notes/03_diffusion推理数据流.md`（第 6 节 scheduler update）

---

## 1. 为什么需要 scheduler

在 diffusion 推理中，scheduler 是连接"模型前向"和"latent 更新"的中间层。它的职责是：

1. **定义时间步序列**：决定在哪些时间点上调用模型，以及每步的步长。
2. **执行 ODE/SDE 积分**：将模型输出的 vector field（或 noise prediction）转化为 latent 的更新。
3. **控制采样质量**：步数越多、步长越密，ODE 积分越精确，输出质量越高。

一个好的 scheduler 接口应该：
- **纯函数化**：步进方法不依赖隐藏状态（除时间序列本身），便于测试和调试。
- **易于 DiT forward 引用**：接受 `(v, x_t, step_index)` 即可更新，签名简单。
- **时间空间正交**：让上游搞清楚当前在 sigma 空间还是 t 空间。

---

## 2. 两类 scheduler 的设计权衡

| 维度 | EulerScheduler（sigma 空间） | RectifiedFlowScheduler（t 空间） |
|------|---------------------------|-------------------------------|
| **时间空间** | sigma ∈ [σ_max, σ_min]，值域差异大 | t ∈ [1, 0]，统一归一化空间 |
| **间隔方式** | log-linear（sigma 是对数间距） | linear（t 是线性间距） |
| **适用模型** | DDPM / DDIM / score-based 模型 | rectified flow / flow matching 模型 |
| **目标模型** | LDM、Stable Diffusion 1.x/2.x | SD3、FLUX、Sana |
| **噪声注入** | 可选（确定性 ODE / 随机 SDE） | 确定性 ODE（无噪声） |
| **复杂度** | 需理解 α_t / σ_t 缩放关系 | 公式最简：dx = v·dt |

### 为什么 Rectified Flow 的 t 空间更简单

Rectified flow 的核心构造是线性插值路径：

```
x_t = (1-t)·x_0 + t·x_1
```

其中 x_0 是干净数据，x_1 是噪声。这个路径的三个好处：

1. **t 的物理意义清晰**：t=1 就是 100% 噪声，t=0 就是 0% 噪声，线性关系。
2. **ODE 形式极简**：dx_t/dt = x_1 - x_0 = v_θ(x_t, t)，不需要 σ/α 的指数缩放。
3. **训练与推理同构**：训练时用 (x_0, x_1) pair 做 flow matching，推理时沿同样路径反向积分。

相比之下，DDPM 的 sigma 空间需要维护 α_t、σ_t、信噪比等多个量，且这些量依赖于训练时的噪声 schedule 设计（linear / cosine / etc.）。

---

## 3. 与 score-based 的关系

### Score-based 去噪

在 score-based 框架下（DDPM、NCSN），模型预测的是 **score function** 或等效的 **噪声 ε_θ**：

```
score = ∇_x log p_t(x) ≈ -ε_θ(x_t, t) / σ_t
```

反向 SDE/ODE 使用 score 来更新 latent，公式中涉及 α_t、σ_t 的导数。

### Rectified Flow

Rectified flow 绕过了 score 的概念，直接预测 **vector field**：

```
v_θ(x_t, t) ≈ (x_1 - x_0)  （在 t 处的条件期望）
```

这使得推理公式变得极其简单：

```
dx_t = v_θ(x_t, t) dt
x_{t_next} = x_t + (t_next - t) * v_θ(x_t, t)    （Euler 法）
```

### 等价性

在特定参数化下，两者可以通过缩放互相转换。例如在 variance-preserving SDE 下，ε_θ 与 v_θ 的关系为：

```
v_θ = ε_θ / t    （rectified flow 的 v-prediction 参数化特例）
```

但在我们的实现中，**rectified flow scheduler 直接接收 v_θ，不做任何缩放**。这使得接口与 SD3/FLUX 的 DiT 模型输出直接对齐。

---

## 4. 为什么 RF update 在 t∈[0,1] 上更简单

对于 Euler 法 ODE 求解器，从 t=1 积分到 t=0 的过程是：

```python
for each step:
    x = x + dt * v_theta(x, t)   # dt < 0 (t 递减)
```

这比 DDPM 的反向公式简单得多，后者需要：

```python
x = (x - (sqrt(1-α_bar_t) - ...) * ε) / sqrt(α_bar_t)
```

**三个关键简化**：

1. **不需要 α/σ 缩放**：rectified flow 的 ODE 是 `dx/dt = v`，而不是 `dx/dt = f(t)x + g(t)²·score`。
2. **timestep 的物理意义统一**：t=1 总是噪声，t=0 总是数据，不依赖训练时的超参数。
3. **确定性**：rectified flow 推理是纯 ODE（无扩散项），给定 x_1 后每一步都是确定的，便于可复现性控制。

---

## 5. step 公式的统一

尽管 sigma 空间和 t 空间的语义不同，Euler 法的数值更新公式是一样的：

```
x_next = x_t + (time_next - time_current) * v
```

其中 `time` 可以是 sigma 也可以是 t。区别在于 `time_next - time_current` 的分布：
- EulerScheduler: sigma 的指数差，步长先小后大（sigma 小的时候变化慢，大的时候变化快）。
- RectifiedFlowScheduler: t 的等间距差，每步步长相同（-1/num_steps）。

这种统一性使得 scheduler 类的实现非常简洁：只需存储时间序列 + 提供 step 方法。

---

## 6. 边界防御

我们在实现中强制执行了以下防御：

| 边界情况 | 处理 |
|---------|------|
| num_steps ≤ 0 | 抛 ValueError |
| 步索引 i 越界 | 抛 IndexError，明确给出有效范围 |
| timesteps 非单调 | rectified_flow_sample 中检测并抛 ValueError |
| timesteps 长度不足 | 最少 2 个点方可执行至少 1 步 |
| dtype 一致性 | step 方法输出 dtype 与输入 x_t 一致 |

---

## 7. 确定性种子策略

- 两个 scheduler 类都接受 `seed` 参数，用于初始化 `numpy.random.RandomState`。
- EulerScheduler 在 `stochastic=True` 模式下用种子控制噪声注入。RectifiedFlowScheduler 是确定性 ODE（无噪声），种子不影响积分结果。
- 在 `rectified_flow_sample` 中也保留了种子参数，为未来可能的后处理采样器预留接口。
- **实际使用时**：设置 `--seed` 保证相同的初始噪声分布和相同的 ODE 轨迹（完全可复现）。

---

## 8. 对未来模块的影响

T10 的 scheduler 设计直接决定了 T11–T12 的 DiT 接口：

```python
# DiT forward 将与 scheduler 配合：
v = dit.forward(latent=z_t, timestep=t, text_embeds=prompt_embeds)
z_next = scheduler.step(v, z_t, step_index)
```

这意味着：
- **DiT 不需要感知 sigma 空间**：它只接收 t∈[0,1]（rectified flow）并输出 v。
- **CFG 在 v 层面做**：`v_cfg = v_uncond + s*(v_cond - v_uncond)`，然后将 v_cfg 传给 scheduler.step。
- **scheduler 不接触 text conditioning**：保持职责单一。

这种分层设计确保 scheduler 可以独立测试（如 toy rectified flow），DiT 可以独立开发（如 toy DiT inference），最后在 pipeline 中拼接。
