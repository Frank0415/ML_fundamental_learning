# 04 · Rectified Flow 与 Flow Matching 入门

> 这一页从概率传输的直觉写到数学形式，再对照 score-based 去噪和 diffusion_engine 的实现。

## 1. 从"去噪"到"概率传输"

传统 DDPM 从去噪出发：前向扩散过程写作 \(x_t = \alpha_t x_0 + \sigma_t \epsilon\)，模型训练为预测噪声 \(\epsilon\)，反向过程再用概率流 ODE 或 Langevin SDE 求解。

**Rectified flow 换个角度看：这是两个概率分布之间的传输问题。**

给定数据分布 \(p_{\mathrm{data}}\)（如所有自然图像的分布）和噪声分布 \(p_{\mathrm{noise}}\)（如 \(\mathcal{N}(0, I)\)），rectified flow 要找到一条从前者到后者的"最直"的流动路径。沿这条路径，概率质量可以从一个分布被"推送"到另一个分布。

## 2. 直线流动

Rectified flow 的起点是一个简单的事实：如果你知道某张图 \(x_0\) 和某个噪声 \(x_1\) 是"配对"的，那它们之间的直线就是最好的路径：

\[
x_t = (1-t)x_0 + t x_1, \qquad t \in [0, 1]
\]

在 \(t=0\) 时，\(x_t\) 是数据；在 \(t=1\) 时，\(x_t\) 是噪声。（注意：本文档中 \(t=0\) 为数据，\(t=1\) 为噪声，与 SD3 论文和 diffusion_engine 的实现一致。）

沿这条直线的速度场（vector field）就是常数：

\[
v(x_t, t) = \frac{d x_t}{d t} = x_1 - x_0
\]

训练目标：学习一个神经网络 \(v_\theta(x_t, t)\)，使它在 \((x_t, t)\) 处的输出接近条件期望：

\[
v_\theta(x_t, t) \approx \mathbb{E}\left[x_1 - x_0 \mid X_t = x_t\right]
\]

**训练目标所需数据**：这个目标不需要知道任意时刻的 score function 或噪声水平，只需要"数据-噪声"配对；配对可以通过对 \(p_{\mathrm{data}}\) 和 \(p_{\mathrm{noise}}\) 独立采样获得。

## 3. Rectified Flow 的训练（概览，不在本项目中实现）

虽然 diffusion_engine 不实现训练，但理解训练有助于理解推理：

每次训练迭代：

\[
\begin{aligned}
x_0 &\sim p_{\mathrm{data}} && \text{（一张真实图片）} \\
x_1 &\sim \mathcal{N}(0, I) && \text{（一个随机噪声）} \\
t &\sim \operatorname{Uniform}[0, 1] && \text{（随机时间点）} \\
x_t &= (1-t)x_0 + t x_1 && \text{（线性插值）} \\
\mathrm{loss} &= \left\lVert v_\theta(x_t, t) - (x_1 - x_0)\right\rVert^2 && \text{（回归目标向量场）}
\end{aligned}
\]

与 DDPM 的差异：DDPM 的 loss 是 \(\left\lVert \epsilon_\theta(x_t, t) - \epsilon\right\rVert^2\)，其中 \(x_t\) 的分布取决于 \(\alpha_t/\sigma_t\) schedule。Rectified flow 的 \(x_t\) 是线性插值，\(t\) 服从均匀分布，训练目标更直接。

## 4. Rectified Flow 的推理

推理时，从噪声 \(x_1 \sim \mathcal{N}(0, I)\) 出发，沿 ODE 反向积分：

推理算法（Euler 法）从 \(t=1\) 向 \(t=0\) 离散步进：

\[
\begin{aligned}
x_t &= \text{随机噪声}, && t=1 \\
v_t &= v_\theta(x_t, t) && \text{（模型预测向量场）} \\
x_{t_{\mathrm{next}}} &= x_t + (t_{\mathrm{next}} - t) v_t && \text{（Euler 步）}
\end{aligned}
\]

\(x_0\) 即为生成结果。

由于 ODE 是可逆的、确定的：
- **相同的噪声 + 相同的模型 + 相同的积分器 → 相同的输出**（完全可复现）。
- **不需要随机噪声注入**（区别于 DDPM 的 SDE 采样器）。
- **步数=质量权衡**：步数越多越接近真实 ODE 解，但计算量越大。SD3 默认 28 步已足够。

## 5. 与 Score-Based 去噪的系统对比

| 维度 | Score-Based / DDPM | Rectified Flow |
|------|-------------------|----------------|
| **模型预测目标** | 噪声 \(\epsilon_\theta\) 或 score \(\nabla \log p_t\) | 向量场 \(v_\theta = x_1 - x_0\) |
| **时间空间** | sigma（需 log-linear 间隔） | \(t \in [0, 1]\)（线性间隔） |
| **前向路径** | \(x_t = \alpha_t x_0 + \sigma_t \epsilon\) | \(x_t = (1-t)x_0 + t x_1\) |
| **训练 t 采样** | 取决于 noise schedule | \(\operatorname{Uniform}[0, 1]\) |
| **推理更新公式** | \(x = (x - \sigma_t \epsilon_\theta) / \alpha_t\) 或等效 | \(x = x + (t_{\mathrm{next}} - t)v\) |
| **推理随机性** | SDE 模式有噪声注入，ODE 模式确定 | 确定性 ODE（无噪声） |
| **代表模型** | Stable Diffusion 1.x/2.x | SD3, FLUX, Sana |

## 6. Rectified Flow 为什么"更直"

"Rectified"（矫正）的含义：
1. **第一条流（1-Rectified Flow）**：用随机配对 \((x_0, x_1)\) 训练第一个流。存在瓶颈，不同 \(x_0\) 可能匹配到同一个 \(x_1\) 导致路径交叉。
2. **第二条流（2-Rectified Flow）**：用第一条流生成的 \((x_0, x_1)\) 作为训练对，重新训练。这些路径几乎不交叉（得到"矫正"），所以 ODE 积分更直、更稳定、可以用更少的步数。

实践中，SD3/FLUX/Sana 使用 1-rectified flow（一次训练），路径通常比 DDPM 更直。2-rectified flow 理论上还能继续矫正路径，但在大规模图像生成中的收益是否超过额外训练成本仍是一个开放问题。

## 7. CFG 在 Vector Field 层面

与 DDPM 一致，CFG 也在 **模型输出层面**（不是 latent 层面）做：

\[
v_{\mathrm{cfg}} = v_{\mathrm{uncond}} + s\left(v_{\mathrm{cond}} - v_{\mathrm{uncond}}\right)
\]

其中 \(s\) 是 CFG scale（通常 1.0-7.5）。\(s=0\) 退化为 unconditional，\(s\to\infty\) 过度放大条件信号（失真）。

这在 rectified flow 框架下仍然有效：\(v_\theta\) 虽然不再是"噪声预测"，但它仍然是"从当前状态到目标的方向"。加强 \(v_{\mathrm{cond}}\) 的贡献就是加强文本引导的方向。

> **结论**：Rectified flow 将扩散生成重新定义为两个概率分布之间的最优传输问题。它用简单的线性插值路径替代 DDPM 复杂的 \(\alpha_t/\sigma_t\) schedule，使得训练更稳定、推理公式极简（\(d x = v\,d t\)）。CFG 仍然在向量场层面做，这与 score-based 模型一致。现代主力模型（SD3、FLUX、Sana）全部基于 rectified flow 范式。

## 8. 在 diffusion_engine 中的实现映射

> **和我的 diffusion_engine 的关系**：本页描述的理论直接映射到 diffusion_engine 的以下模块：

| 概念 | diffusion_engine 文件 |
|------|----------------------|
| 向量场 \(v_\theta(x,t)\) | `core/rectified_flow.py` - `rectified_flow_step()` |
| Euler ODE 积分循环 | `core/rectified_flow.py` - `rectified_flow_sample()` |
| \(t\in[0,1]\) 时间调度 | `core/scheduler.py` - `RectifiedFlowScheduler` |
| Sigma 空间调度 | `core/scheduler.py` - `EulerScheduler` |
| Timestep 嵌入 | `core/timestep_embedding.py` - `sinusoidal_embedding()` |
| Toy 演示 | `experiments/toy_rectified_flow/` |

**diffusion_engine 的时间参数**：diffusion_engine 直接使用 rectified flow 的 \(t\in[0,1]\) 作为默认时间空间，不进行 \(\sigma\) 转换。上游 DiT 模型只需输出 \(v_\theta\)（向量场），scheduler 直接用于 Euler 积分。这与 SD3 论文中描述的 inferencer 对齐。

## 9. Rectified Flow 在视频生成模型中的应用

SD3、FLUX、Sana 等文生图模型采用 Rectified Flow；不少视频模型也使用 rectified flow / flow matching：

| 模型 | 框架 | 推理步数 | Video Latent Shape 示例 | 资源档位 |
|------|------|---------|------------------------|------------|
| **Wan2.1-1.3B** | Flow Matching | 50 步 | `(1, 16, 21, 60, 104)` @ 81f×480p | 极限可行 (~8.2 GB) |
| **CogVideoX-2B** | DDPM / Flow Matching | 50 步 | `(1, 4, 13, 90, 60)` @ 49f×720×480 | 可行 (~8-9 GB) |
| **LTX-Video 2B** | Flow Matching (distilled) | 4~8 步 | `(1, 4, 15, 22, 15)` @ 121f×720×480 | 可行 (~6 GB) |
| **HunyuanVideo 1.5** | Flow Matching | 10~20 步 (distilled) | `(1, 16, 33, 90, 160)` @ 129f×720×1280 | 偏紧 (需量化+offload) |

**视频与图像的 Flow Matching 差异**：

- **Latent 维度**：图像 latent 是 4D `(B,C,H,W)`，视频 latent 是 5D `(B,C,T,H,W)`。多出的时间轴使 token 数乘上了 \(T_{\mathrm{latent}}\) 因子。
- **Attention 规模**：图像 DiT 在 1024px 下的 token 数约 4096，视频 DiT 在 720p×49f 下的 token 数可达 17,550（CogVideoX）甚至 32,760（Wan 1.3B）。视频 attention 矩阵的内存压力按 \(O(n^2)\) 增长。
- **CFG Scale**：视频模型通常使用更高的 CFG（5.0~7.0 vs 图像的 3.5~5.0），意味着 text guidance 在视频中更重要，但也意味着每步 cond+uncond 双 forward 更"贵"。
- **Few-step 蒸馏**：LTX-Video 和 HunyuanVideo 1.5 的蒸馏版本把 50 步降到 4-20 步，直接减少 denoiser 的调用次数。

## 10. 延伸阅读

- **SD3 论文**：[2403.03206] Scaling Rectified Flow Transformers - 第 2-3 节详细阐述了 rectified flow 的理论基础和在实际图像生成中的应用。
- **Flow Matching 原始论文**：[2210.02747] Flow Matching for Generative Modeling - 建立 flow matching 框架的理论基础。
- **Rectified Flow 原始论文**：[2209.03003] Flow Straight and Fast - 提出"矫正"概念和两阶段训练。
- **diffusion_engine 设计笔记**：`learning/notes/04_scheduler设计.md` - 分析两类 scheduler 的权衡和实现细节。
