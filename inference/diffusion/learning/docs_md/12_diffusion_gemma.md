# 12 · DiffusionGemma 架构与推理

本文详细探讨 Google DeepMind 最近开源的 **DiffusionGemma**，一个基于离散扩散（Discrete Diffusion）的全新文本生成模型。我们将从自回归的显存瓶颈、文本去噪机制的演进、Encoder-Denoiser 动态架构、分块自回归采样、以及采样调度器等多个维度进行全方位深度解析。

---

## 1. 自回归的瓶颈与扩散的计算绑定

### 自回归大型语言模型（AR LLM）的显存带宽噩梦
传统的自回归大型语言模型（如 GPT-4, Llama 3, Gemma 等）在文本生成时采用**逐个 Token 顺序生成**的机制。这种机制在单用户推理（Batch Size = 1）时，会面临极端的**显存带宽瓶颈（Memory-Bandwidth Bound）**：
* **算术强度极低**：在生成每个 Token 的前向传播中，GPU 必须从其显存（HBM/VRAM）中将全部模型参数（数十 GB）读取 to SRAM/Tensor Cores 中。而读入的权重仅与当前这一个 Token 的激活向量进行一次矩阵乘法。
* **计算单元闲置**：由于显存读取速度（如 1.5 - 3 TB/s）远慢于 GPU 计算单元的算力上限（如数百 TFLOPS），这导致 GPU 的 Tensor Cores 大部分时间都在处于闲置状态，等待数据从显存加载。
* **吞吐与延迟的权衡**：通过 Batching 增加并发可以提高硬件利用率，但由于必须顺序等待每个 Token 生成，单用户的端到端延迟（Latency）完全没有被优化，依然受限于单次前向传播的读显存耗时。

### DiffusionGemma 的计算绑定（Compute-Bound）设计
DiffusionGemma 颠覆了这一推理范式。它不是每次前向传播预测一个 Token，而是**一次性预测和提炼一个包含 256 个 Token 的完整画布（Canvas）**：
* **算术强度暴增**：前向传播处理的是长度为 256 的序列，GPU 可以将模型参数加载一次后，在 Tensor Cores 中并行进行 256 个位置的注意力计算 and 矩阵乘法。这使得计算密度大幅提升。
* **转换瓶颈**：通过将推理从**显存带宽绑定**转化为**计算绑定（Compute-Bound）**，DiffusionGemma 充分释放了现代 GPU 的 Tensor Cores 算力，使得单用户能以极高的速度并生成 Token。
* **性能对比**：在 NVIDIA H100 GPU (FP8) 上，DiffusionGemma 可实现超过 **1100+ Tokens/秒** 的超高生成速度；即使在消费级显卡（如 NVIDIA GeForce RTX 5090）上，也能实现 **700+ Tokens/秒** 的推理表现。

---

## 2. 文本去噪机制的演进

将扩散（Diffusion）模型从连续空间（如图像像素）引入离散空间（如文本 Token）并非易事。在图像扩散中，我们可以向像素添加高斯噪声（使得红色像素稍微变蓝），但在文本中，无法将 Token `"The"` 稍微模糊成另一个 Token。为此，学术界与工业界探索了两种主要的离散文本去噪路径：

### 路径 A：掩码扩散（Masked Diffusion）
类似于 BERT 的 Masked Language Modeling (MLM)，该机制在训练时将文本序列中的部分 Token 替换为特殊的 `[MASK]` 标记：
* **前向过程**：根据时间步逐步增加 `[MASK]` 比例。
* **逆向去噪**：模型预测 `[MASK]` 位置的真实 Token。在每一步中，采样器根据模型预测的置信度，仅保留最确信的部分 Token，替换掉对应的 `[MASK]`，并在下一步继续预测剩余的 `[MASK]`。
* **致命缺陷**：掩码扩散在去噪时缺乏**自纠错能力（Self-Correction）**。一旦某个 `[MASK]` 在早期步骤中被模型错误地填充为某个具体 Token，这个 Token 就会被锁定，无法在后续步骤中被修改。这类似于自回归模型“一错再错”的幻觉效应。

### 路径 B：均匀状态扩散（Uniform State Diffusion — DiffusionGemma 的选择）
为了克服掩码扩散的缺陷，DiffusionGemma 采用了更先进的**均匀状态扩散**机制：
* **噪声定义**：此处的“噪声”不再是唯一的 `[MASK]` 标记，而是**从词表中随机、均匀抽样的任意破坏性 Token**。
* **前向过程**：在训练时，输入文本中的部分 Token 会被随机替换成词表中的乱码 Token，比例随着扩散时间步变化。
* **逆向去噪与自纠错**：
  * 在每个去噪步，模型会对画布（Canvas）中**所有位置**的 Token 进行重新预测并输出概率分布。
  * 如果模型对当前位置的预测置信度较高，且满足采样器阈值，则将该位置替换为新预测的 Token。
  * 如果模型对某些位置的预测概率极低（认为这是噪声），采样器会主动对其进行**再噪声化（Re-noising）**——即再次用词表中的随机 Token 覆盖它们。
  * **优势**：由于没有锁定机制，早期步骤填充的错误 Token，如果随着周围上下文明朗化后导致置信度下降，可以在后续步骤中被重新判定为“噪声”并进行修改和自我纠正。

| 特性维度 | 掩码扩散 (Masked Diffusion) | 均匀状态扩散 (Uniform State Diffusion) | 自回归生成 (Autoregressive) |
| :--- | :--- | :--- | :--- |
| **噪声表现** | 显式 `[MASK]` 标记 | 隐式均匀随机 Token 替换 | 无扩散噪声 |
| **去噪动作** | 逐级填充 `[MASK]` 空间 | 预测全画布，不满足阈值处重填随机噪声 | 逐个 Token 尾部追加 |
| **自纠错能力** | ❌ 无法修改已填充的 `[MASK]` |  **允许持续修改和纠正错误预测** | ❌ 无法回溯和修改历史 Token |
| **生成步数** | 取决于采样策略（通常 20-50 步） | 固定或自适应步数（如 48 步） | 与生成文本长度 $N$ 严格一致 |

---

## 3. 网络架构：Encoder-Denoiser 动态模式切换

DiffusionGemma 并未重新设计一套复杂的网络，而是巧妙地在底层重用了 **Gemma 4 26B A4B** 开源模型 checkpoint（混合专家架构 MoE，拥有 30 层，词表大小 262K，单 Token 激活 3.8B 参数/8 个专家），并通过一个**Encoder-Denoiser 动态模式切换补丁**实现了双重功能的融合：

```
                    ┌────────────────────────┐
                    │  Prompt: "Write a..."  │
                    └───────────┬────────────┘
                                │
                         (Encoder Mode)
                       Causal Attention
                                │
                                ▼
                       ┌────────────────┐
                       │  Static KV     │ (Stored in Memory)
                       │  Cache         │
                       └────────│───────┘
                                │
                     (Cross-Attention Guidance)
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │ Denoiser Loop (48 steps)  ▼                           │
    │                                                       │
    │  [Random Canvas]  ──►  (Denoiser Mode)  ──► [Denoised │
    │    (256 tokens)      Bidirectional Attention  Canvas] │
    │                                                       │
    │         ▲                                             │
    │         └─────────── Self-Conditioning ───────────────┘
    └───────────────────────────────────────────────────────┘
```

### A. 编码器模式（Encoder Mode）
* **职责**：处理用户的输入 Prompt，提取语义特征，为后续去噪过程提供上下文指导。
* **机制**：在此模式下，Gemma 架构采用标准的**因果自注意力（Causal Self-Attention）**，即每个 Token 只能看到其前驱 Token。
* **输出**：输入 Prompt 经过前向传播后，生成对应的键值缓存（**KV Cache**）并写入内存。这个 KV Cache 在后续的去噪循环中保持**静态只读**，不参与迭代更新。

### B. 去噪器模式（Denoiser Mode）
* **职责**：接收一个包含 256 个 Token 的随机噪声画布（Canvas），结合 Encoder 提供的 KV Cache，逐步去噪还原出真实的响应文本。
* **机制**：去噪器将因果自注意力机制替换为**双向自注意力机制（Bidirectional Self-Attention）**。由于不再使用因果 Mask，画布上的 256 个 Token 在前向传播时可以实现 **All-to-All** 的互相可见。
* **跨注意力**：画布 Token 作为 Query (Q)，去噪器通过 Cross-Attention 机制去访问并读取由 Encoder 模式生成的 Prompt KV Cache (K/V)。

### C. 自我调节与历史记忆（Self-Conditioning）
去噪器如何感知它在上一步做出的预测？
1. 在第 $t$ 步，去噪器输出全画布每个位置的 Logits。
2. 这些 Logits 经过 Softmax 转化为概率分布。
3. 概率分布与模型的**词嵌入表（Embedding Matrix）**相乘，为画布每个位置生成一个加权的融合特征向量（融合特征包含了上一步预测的概率分布信息）。
4. 该融合特征向量通过一个轻量级的前馈网络（FFNN）进行映射，并直接**加到第 $t+1$ 步的输入 Token Embedding 中**。
5. 这为模型提供了清晰的“预测历史记忆”，稳定了画布在不同去噪步之间的收敛轨迹。

---

## 4. 分块自回归扩散（Block Autoregressive Diffusion）

由于去噪器单次只能处理 256 Token 长度的画布，如何生成长文本？
DiffusionGemma 将**局部并行扩散**与**全局自回归**相结合，实现了**多画布采样（Multi-Canvas Sampling）**：

```python
# 核心数据流概念伪代码
prompt = "Please write a long sci-fi story..."
kv_cache = encoder.prefill(prompt)  # 首次 Prefill 生成初始 KV Cache

while not generate_finished:
    # 1. 初始化 256 长度的随机画布
    canvas = initialize_random_tokens(length=256)
    
    # 2. 逆向迭代去噪循环 (例如 48 步)
    for step in range(num_denoising_steps):
        # 融合上一步的概率分布特征
        conditioned_embeddings = embed(canvas) + self_conditioning(prev_probs)
        
        # 去噪器进行双向 All-to-All 注意力计算，并 cross-attend 到只读的 kv_cache
        logits = denoiser.forward(conditioned_embeddings, kv_cache)
        
        # 采样器决定保留哪些 token，重新对低置信度位置进行 noise 覆盖
        canvas, prev_probs = sampler.step(logits, canvas)
        
        if adaptive_stopping.should_stop():
            break
            
    # 3. 256 Canvas 去噪完成，识别到结束符或填满
    finalized_block = canvas
    
    # 4. 自回归扩展：将 finalized_block 视为 prompt 的延续，送入 Encoder 
    # 追加并更新静态 KV Cache 缓存。之后该 KV Cache 作为下一画布的上下文。
    kv_cache = encoder.incremental_prefill(finalized_block, kv_cache)
    
    if eos_token in finalized_block:
        generate_finished = True
```

这种机制结合了**扩散模型在单块内部极高的并行吞吐量**与**自回归模型处理长序列时的优异上下文连贯性**。

---

## 5. 采样器与调度器参数（Best Practices）

DiffusionGemma 的核心控制中心在于其采样器（Sampler）与调度器（Scheduler）的设计，官方推荐了以下标准超参数配置：

### A. 温度调度器（Logits Temp Scheduler）
为了平衡去噪过程中的**探索（Exploration）**与**收敛（Exploitation）**，Logits 会被除以一个动态温度 $T$。
* 在 48 步去噪过程中，温度 $T$ 采用**线性衰减（Linear Decay）**，由 **0.8 逐渐降低至 0.4**。
* **直觉**：在去噪早期，画布充斥大量噪声，较高的温度有利于模型多样化地尝试不同的 Token 组合；在去噪后期，画布逐渐清晰，较低的温度可以“锁紧”预测分布，消除模糊性，实现精确的 Token 沉淀。

### B. 熵界采样器（Entropy-Bounded Sampler）
用于精确控制去噪每一步中保留哪些 Token、丢弃哪些 Token。
1. **熵值计算**：计算画布上每个位置的预测概率分布的香农熵（Entropy）。熵越低，代表模型对该位置的预测越自信（如 99% 的概率是 `"LLM"`，熵接近 0）。
2. **排序过滤**：将画布的 256 个位置按熵值**从低到高（从最自信到最不自信）进行排序**。
3. **阈值累加**：从最自信的 Token 开始累加。当累加的互信息上限（Mutual Information Bound）超过设定的**熵界（Entropy Bound = 0.1）**时，停止接纳。
4. **再噪声化**：所有被接纳的 Token 保持预测值，未被接纳的 Token 一律判定为噪声，**使用全新的随机 Token 重新覆盖（Re-noising）**。

### C. 自适应早停机制（Adaptive Stopping）
不需要强行跑满 48 步。如果画布提前收敛，系统会触发自适应早停，以节省宝贵的算力：
同时满足以下两点时停止：
* **置信度达标**：全画布所有 Token 的平均熵低于 **0.005**（代表模型对全画布内容已极度确信）。
* **预测稳定**：最高概率的 Token 预测结果在连续 **2** 步中完全一致（代表画布状态已不再发生任何漂移和修改）。

---

## 6. 案例分析：Sudoku Solver（数独求解器）

为什么 Diffusion 架构适合解决强约束的多变量协同任务（如数独）？

```
  自回归模型 (AR) 顺序预测：                   DiffusionGemma 并行协同去噪：
  ┌───┬───┬───┐                               ┌───┬───┬───┐
  │ 5 │ 3 │ ? ├──► 必须立刻决定第三格         │ 5 │ 3 │ 4 │  所有格点在 All-to-All 
  └───┴───┴───┘    无法考虑未来冲突           ├───┼───┼───┤  注意力下同时互相制约，
  ┌───┬───┬───┐                               │ 6 │ ? │ 8 │  若后期发现冲突，可通过 
  │ 6 │ ? │ 8 │    一旦写错，无法修改         ├───┼───┼───┤  Re-noising 机制在后续去噪 
  └───┴───┴───┘                               │ ? │ 9 │ ? │  步中将错误擦除并重写。
                                              └───┴───┴───┘
```

* **自回归的痛点**：数独需要满足横、竖、九宫格的三重 intersecting 约束。自回归模型必须从左到右硬性预测。当它预测到第 80 格时，如果发现和第 1 格冲突，它**无法进行回溯（Backtracking）或自我擦除**，导致最终求解失败率极高。
* **DiffusionGemma 的解法**：
  * **全局上下文感知**：Denoiser 采用双向注意力，首尾 Token 互相可见，信息对称流动，在一前向传播中同时考量所有格点的冲突约束。
  * **容错与自我修正**：如果第 10 步填入的数字在第 20 步被判定大概率冲突，它的置信度（熵）会变差，并在下一步中被重置为噪声数字重新求解。
* **微调效果**：DiffusionGemma 26B 基础模型在数独任务上的正确率为 ~0%。但通过 Google DeepMind 开源的 **Hackable Diffusion** 基于 JAX 的监督微调（SFT）配方训练后，数独**求解成功率飙升至 80%**，且步数大幅缩短。

---

## 7. 推理与部署管线

### vLLM 部署标准命令
vLLM 官方已原生集成 DiffusionGemma。可通过以下 OpenAI 兼容的 Server 命令行一键启动：

```bash
vllm serve google/diffusiongemma-26B-A4B-it \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN \
  --generation-config vllm \
  --hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}' \
  --diffusion-config '{"canvas_length": 256}' \
  --enable-chunked-prefill
```

### Hugging Face Transformers 推理代码
开发者可以使用标准 `transformers` 库中的 `DiffusionGemmaForBlockDiffusion` 进行推理：

```python
from transformers import DiffusionGemmaForBlockDiffusion, AutoProcessor
import torch

MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# 1. 初始化处理器与离散扩散专用模型类
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. 准备对话模版
messages = [
    {"role": "user", "content": "Explain the core difference between AR and Diffusion LLMs."}
]

input_ids = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# 3. 运行多画布扩散采样推理
output = model.generate(**input_ids, max_new_tokens=512)

# 4. 打印解码文本
text = processor.decode(output[0], skip_special_tokens=False)
print(text)
```

---

## 8. 与 diffusion_engine 的概念映射

本项目自写引擎 `diffusion_engine` 实现的诸多核心思想与 DiffusionGemma 的设计存在高度的概念重合，为读者理解底层代码提供了直观的印证：

* **双向注意力与因果注意力的动态切换**：去噪器模式切换的本质，类似于我们在 `diffusion_engine/core/attention.py` 中实现的灵活 Causal Mask 开关。自回归预处理（Causal Prefill）使用有因果 Mask 的 Attention，而逆向去噪（Bidirectional Denoising）将 Mask 彻底移除，变为 All-to-All 交互。
* **Prompt Embedding Cache 与推理管线控制**：项目在 `diffusion_engine/core/text_conditioning.py` 实现了 Prompt Cache 机制，只在首次 encode 时运行 text encoder，随后将 embedding 固化，在全循环中复用，这正是 DiffusionGemma 保持 KV Cache 静态只读的核心逻辑。
* **Scheduler 控制**：我们在 `diffusion_engine/core/scheduler.py` 实现的 Euler 降噪路径，是基于时间步的迭代逼近。这与 DiffusionGemma 中 Logits 随温度 $T$ 衰减、步步逼近确定性文本的过程在本质上是相通的。

---

## 9. 参考资源与链接

本章内容主要整理并参考自以下 DiffusionGemma 官方发布的技术资料与社区指引：
1. **Google 官方开发者指南**：[DiffusionGemma: The Developer Guide](https://developers.google.com/en/diffusiongemma-the-developer-guide)
2. **可视化原理解析**：[A Visual Guide to DiffusionGemma](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma)
3. **Hugging Face 模型卡片**：[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
