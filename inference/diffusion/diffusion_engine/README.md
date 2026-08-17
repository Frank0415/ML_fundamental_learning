# diffusion_engine：现代 Diffusion 推理引擎

> **状态**：骨架就绪，核心模块待 T10-T12 填充。
> **定位**：从零构建的最小 diffusion 推理引擎，与 `minivLLM/` 完全独立。

---

## 与 minivLLM 的关系

**diffusion_engine 与 minivLLM 是两个独立项目，不存在代码依赖或继承关系。**

`minivLLM/` 是一个 LLM 推理引擎（Qwen3-0.6B），架构设计从根上服务于自回归语言模型。经 T2 逐模块审计（`reports/engine_inventory.md`），其 14 个模块中仅 `SiluAndMul`（11 行）可被直接复制到本引擎，其他 13 个模块在注意力范式、迭代循环、条件注入三个方面与扩散模型存在结构性差异。

### 不复用的核心理由

| 差异维度 | 为什么 minivLLM 不能用 |
|---------|----------------------|
| **Attention** | GQA + causal mask + RoPE → DiT 需要 full attention + AdaLN modulation |
| **迭代循环** | 自回归（逐 token 追加，KV cache 是关键） → 扩散是迭代去噪（每步刷新 latent，KV cache 无用） |
| **Norm** | 静态 RMSNorm → DiT 需要 AdaLN-Zero（timestep + text 动态生成 scale/shift） |

### 仅有的交点

- **代码复用**：`minivLLM/minivllm/layers/numpy/activation.py` 的 `SiluAndMul`（11 行），将在 T11 复制到 `diffusion_engine/layers/activation.py`。
- **风格参考**（不复制代码）：`Linear` 薄 wrapper、`RMSNorm` residual 融合接口、`RotaryEmbedding` `@torch.compile` 缓存模式。

### 目录隔离

```
diffusion/
├── minivLLM/              # 只读，审计用途，永不动
└── diffusion_engine/      # 新引擎，独立 Python 包
    ├── README.md          # 本文件
    ├── layers/            # 算子层（待 T11 创建）
    ├── core/              # 核心模块（待 T10-T12 填充）
    └── tests/             # 单元测试（待 T10-T12 填充）
```

两份 `pyproject.toml` 完全独立，使用各自的虚拟环境。

---

## 已实现模块

> 目前 diffusion_engine 处于骨架阶段，以下模块等待 Wave 2（T10）和 Wave 3（T11-T12）填充。

尚无已实现模块。`SiluAndMul` 将在 T11 从 minivLLM 复制到 `diffusion_engine/layers/activation.py`。

---

## 未来模块（按实现顺序）

### Wave 2（T10）- Scheduler 与 Rectified Flow

| 模块 | 文件 | 职责 | 关键接口 |
|------|------|------|---------|
| Scheduler | `core/scheduler.py` | 噪声调度器（Euler/Heun/RF ODE） | `add_noise()`, `step()`, `set_timesteps()` |
| Rectified Flow | `core/rectified_flow.py` | Rectified Flow 矢量场训练/推理 | `sample_ode()`, `compute_flow_loss()` |
| Timestep Embedding | `core/timestep_embedding.py` | Fourier/sinusoidal timestep 编码 | `get_timestep_embedding(t, dim)` |

### Wave 3（T11-T12）- Transformer Block 与 Pipeline

| 模块 | 文件 | 职责 | 关键接口 |
|------|------|------|---------|
| Attention | `core/attention.py` | DiT full attention（所有 patches 互相 attend） | `forward(x, c)` - AdaLN modulated |
| Transformer Block | `core/transformer_block.py` | DiT block（AdaLN-Zero + FFN + attention） | `forward(x, t_emb, c_emb)` |
| Text Conditioning | `core/text_conditioning.py` | 文本编码器条件注入（pooled + seq） | `encode_text(prompt)`, `get_conditioning()` |
| Pipeline | `core/pipeline.py` | 完整推理 pipeline（调度 + 去噪 + VAE 解码） | `__call__(prompt)` → PIL Image |
| Memory Manager | `core/memory_manager.py` | 显存追踪与 buffer 管理 | `allocate()`, `free()`, `stats()` |
| Activation | `layers/activation.py` | 激活函数（从 minivLLM 复制 SiluAndMul） | `SiluAndMul.forward(x)` |

### Wave 4（T16-T17）- 系统优化辅助

| 模块 | 文件 | 职责 |
|------|------|------|
| (实验脚本在 `experiments/` 下，不在 engine core 内) | - | - |

---

## 设计原则

1. **最小主义**：每个模块只实现扩散推理所需的最小功能，不引入无关抽象。
2. **shape 明确**：所有接口都标注 tensor shape 约定（latent: `B C H W`；video latent: `B C T H W`）。
3. **可测试**：`diffusion_engine/tests/` 覆盖 scheduler shape、attention output、pipeline smoke。
4. **纯 PyTorch**：无 C++/CUDA/Triton 扩展（Mac M5 开发环境限制）。高性能 attention 通过 `torch.nn.functional.scaled_dot_product_attention` 走 PyTorch 2.x 内置路径。
5. **与 diffusers 兼容**：接口风格参考 `diffusers` 库的 pipeline/scheduler 设计，但不直接依赖它作为 engine 内部实现。

---

## 测试策略

- `diffusion_engine/tests/test_scheduler.py`：验证 add_noise 和 step 的 shape 与数值范围。
- `diffusion_engine/tests/test_shapes.py`：验证 attention/transformer_block 输入输出 shape。
- `diffusion_engine/tests/test_pipeline.py`：端到端 smoke test（最小 DiT + toy latent → 输出图像非全零）。

运行：`python -m pytest diffusion_engine/tests -q`
