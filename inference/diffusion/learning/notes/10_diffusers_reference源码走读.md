# Diffusers Reference Pipeline 源码走读

> **前提**：本文档基于 diffusers ≥ 0.32.0 的公开 API 整理。**不需要 import diffusers**，只写明 API 路径和参数签名。目的是让 T13/T15 执行者和 `diffusion_engine/` 开发者理解"diffusers 的 pipeline 长什么样、怎么调用、和我们自己的引擎是什么关系"。
> **定位**：diffusers pipeline 是我们的 **reference（参考实现）**，不是我们的成果。我们用 diffusers 验证模型能跑、记录 VRAM 和 latency，但自己的 `diffusion_engine/` 走独立实现路线。

---

## 1. Diffusers Pipeline 的总体设计

### 1.1 统一的入口模式

所有 diffusers pipeline（图像和视频）都遵循同一个 template：

```python
from diffusers import SomePipeline

pipe = SomePipeline.from_pretrained(
    "org/model-name",
    torch_dtype=torch.float16,          # 权重 dtype
    variant="fp16",                      # 下载 fp16 变体（如可用）
)

# 可选：内存优化
pipe.enable_model_cpu_offload()          # 或 enable_sequential_cpu_offload()
pipe.enable_vae_slicing()                # VAE decode 逐 patch 处理
pipe.enable_vae_tiling()                 # VAE decode 分 tile 处理

# 推理
output = pipe(
    prompt="一只柴犬在樱花树下",
    num_inference_steps=28,              # 步数
    guidance_scale=4.5,                  # CFG scale
    height=1024,                         # 像素高度
    width=1024,                          # 像素宽度
    # 视频额外参数 ↓
    num_frames=16,                       # （仅视频 pipeline）
    generator=torch.Generator().manual_seed(42),
).frames[0]  # 或 .images[0]
```

### 1.2 三个核心方法

| 方法 | 作用 | 说明 |
|------|------|------|
| `from_pretrained(repo_id, **kwargs)` | 下载/加载模型权重，构建 pipeline | 自动根据 `model_index.json` 组装子模块（vae, text_encoder, tokenizer, transformer, scheduler） |
| `enable_model_cpu_offload()` | 逐模块 CPU offload | 一次只保持一个子模块在 GPU，其他在 CPU。最省显存但最慢。 |
| `enable_sequential_cpu_offload()` | 逐子层 CPU offload（更细粒度） | 比 model_cpu_offload 更慢但更省。视频模型的极限手段。 |
| `enable_vae_tiling()` | VAE decoder 分 tile | 将大图像切成 tile 分别 decode，降低 VAE 峰值显存。 |
| `enable_vae_slicing()` | VAE decoder 分 slice | 类似 tiling，但方向不同。两者可同时开启。 |

`from_pretrained` 的核心行为：
1. 下载 `config.json`（pipeline 总配置）和 `model_index.json`（子模块映射）。
2. 根据 `model_index.json` 逐个下载子模块（vae、text_encoder、transformer 等）。
3. 组装为完整 pipeline。每个子模块都是独立的 `nn.Module`，可单独访问（如 `pipe.vae`、`pipe.transformer`）。

---

## 2. Image Reference Pipeline 关键入口

### 2.1 `diffusers.StableDiffusion3Pipeline`

- **API 路径**：`diffusers.StableDiffusion3Pipeline`
- **架构**：MMDiT（Multimodal DiT）+ rectified flow。text encoder 支持 CLIP-L + CLIP-G + T5-XXL 三路编码。T5 最重，SD3-Medium 可省略 T5。
- **关键参数**：
  - `num_inference_steps`：默认 28。蒸馏变体可用 4。
  - `guidance_scale`：推荐 4.0~7.0。
  - `height` / `width`：需能被 VAE 的 `sample_size` 整除（通常 64 的倍数）。
- **典型调用**：
  ```python
  pipe = StableDiffusion3Pipeline.from_pretrained(
      "stabilityai/stable-diffusion-3.5-medium",
      torch_dtype=torch.float16,
  )
  pipe.enable_model_cpu_offload()
  image = pipe("prompt", num_inference_steps=28, guidance_scale=4.5).images[0]
  ```
- **12GB 判断**：SD3-Medium（no-T5）在 fp16 + CPU offload 下约 8GB VRAM，安全可跑。SD3-Large 需 ~12GB，在 12GB 卡上是极限操作。

### 2.2 `diffusers.FluxPipeline`

- **API 路径**：`diffusers.FluxPipeline`
- **架构**：FLUX single-stream DiT（非 MMDiT 双流）。支持 multi-text-encoder（CLIP + T5）。schnell 变体是蒸馏版，4 步即可出图。
- **关键参数**：
  - `num_inference_steps`：schnell 用 4，dev 用 28~50。
  - `guidance_scale`：schnell 推荐 0.0（无 CFG），dev 用 3.5~7.0。
  - `height` / `width`：需是 16 的倍数（与 VAE 配置相关）。
- **典型调用**（schnell）：
  ```python
  pipe = FluxPipeline.from_pretrained(
      "black-forest-labs/FLUX.1-schnell",
      torch_dtype=torch.float16,
  )
  pipe.enable_sequential_cpu_offload()  # FLUX 较大，用 sequential
  image = pipe("prompt", num_inference_steps=4, guidance_scale=0.0).images[0]
  ```
- **12GB 判断**：schnell 在 fp16 + sequential offload 下约 10GB，可行但紧张。dev 在大分辨率下可能 OOM。

### 2.3 `diffusers.SanaPipeline`

- **API 路径**：`diffusers.SanaPipeline`
- **架构**：Sana efficient DiT。text encoder 使用 Gemma-2B（一个小 LLM）。VAE 压缩比高达 32×，latent 尺寸远小于 SD3/FLUX。
- **关键参数**：
  - `num_inference_steps`：推荐 14~20。
  - `guidance_scale`：推荐 4.0~5.0。
  - `height` / `width`：支持 1024×1024 甚至更高，得益于 32× 高压缩 VAE。
- **12GB 判断**：Sana 是 12GB 下最友好的图像模型。1024×1024 仅需 <6GB VRAM（fp16，无 CPU offload 也能跑）。**T14 文生图参考推理的首选模型**。

---

## 3. Video Reference Pipeline 关键入口

### 3.1 `diffusers.LTXVideoPipeline`

- **API 路径**：`diffusers.LTXVideoPipeline`
- **架构**：LTX-Video DiT + T5-XXL text encoder。2B 参数，distilled for few-step inference。关键卖点是 **实时/准实时视频生成**：4~8 步即可输出可用的短视频。
- **关键参数**：
  - `num_inference_steps`：推荐 4~8（蒸馏模型，步数极少）。
  - `num_frames`：默认 121（约 5 秒 @ 24fps）。降级时可减至 16。
  - `height` / `width`：默认 480×720。降级至 256×256。
  - `guidance_scale`：蒸馏模型通常用 0.0~3.0。
- **典型调用**：
  ```python
  pipe = LTXVideoPipeline.from_pretrained(
      "Lightricks/LTX-Video",
      torch_dtype=torch.float16,
  )
  pipe.enable_model_cpu_offload()
  video = pipe(
      "一只猫在草地上奔跑",
      num_inference_steps=8,
      num_frames=16,
      height=256, width=256,
  ).frames[0]
  ```
- **12GB 判断**：2B params + few-step + 小分辨率 → ~6-8GB VRAM。RTX 4060 8GB 上实测可跑 720×480@121f。是 T15 的首选视频模型。

### 3.2 `diffusers.CogVideoXPipeline`

- **API 路径**：`diffusers.CogVideoXPipeline`
- **架构**：CogVideoX expert transformer + T5-XXL text encoder。2B 参数（CogVideoX-2B）。temporal attention 使用 **causal mask**（当前帧只能看过去的帧），这是视频模型中非常独特的设计。
- **关键参数**：
  - `num_inference_steps`：默认 50。建议降级至 30（12GB）。
  - `num_frames`：默认 49。降级至 16 或 25。
  - `height` / `width`：默认 480×720。降级至 256×256。
  - `guidance_scale`：推荐 6.0（比图像模型高，CogVideoX 的 optimal scale 较大）。
  - `use_dynamic_cfg`：CogVideoX 有动态 CFG 选项，可在去噪后期自动降低 guidance scale。
- **典型调用**：
  ```python
  pipe = CogVideoXPipeline.from_pretrained(
      "THUDM/CogVideoX-2b",
      torch_dtype=torch.float16,
  )
  pipe.enable_model_cpu_offload()
  pipe.vae.enable_tiling()
  video = pipe(
      "一只猫在草地上奔跑",
      num_inference_steps=30,
      num_frames=16,
      height=256, width=256,
      guidance_scale=6.0,
  ).frames[0]
  ```
- **12GB 判断**：官方 min VRAM 为 4GB。49f@480p 约 9GB，安全可跑。**需要注意**：CogVideoX 的 tokenizer 和 text encoder 是 T5-XXL（~9GB 权重），CPU offload 是必须的。

### 3.3 `diffusers.WanPipeline`

- **API 路径**：`diffusers.WanPipeline`（Wan 2.1 系列）
- **架构**：Wan DiT + Wan 3D VAE。1.3B 参数是最小版本（Wan2.1-T2V-1.3B）。3D VAE 直接在 `(B, C, T, H, W)` latent 上编码/解码，非逐帧处理。
- **关键参数**：
  - `num_inference_steps`：默认 50。建议降级到 30（12GB）。
  - `num_frames`：默认 81（~5s @ 16fps）。降级至 16。
  - `height` / `width`：默认 480×832。降级至 256×256。
  - `guidance_scale`：推荐 5.0。
- **典型调用**：
  ```python
  pipe = WanPipeline.from_pretrained(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
      torch_dtype=torch.float16,
  )
  pipe.enable_model_cpu_offload()
  video = pipe(
      "一只猫在草地上奔跑",
      num_inference_steps=30,
      num_frames=16,
      height=256, width=256,
  ).frames[0]
  ```
- **12GB 判断**：1.3B params + 3D VAE，480p 时 81 帧约 8GB。在 12GB 下极限可跑但需将所有优化开关全开。

---

## 4. Image 和 Video Pipeline 的共同 API 模式

### 4.1 统一的 `__call__` 参数

所有 diffusers pipeline 的 `__call__` 方法共享以下参数子集：

```python
output = pipe(
    prompt,                        # str | List[str]
    negative_prompt=None,          # 默认 ""（空字符串 prompt）
    num_inference_steps=...,       # 推理步数
    guidance_scale=...,            # CFG scale
    height=..., width=...,         # 像素分辨率
    num_images_per_prompt=1,       # 每 prompt 生成的图片数
    generator=None,                # torch.Generator（seed 控制）
    output_type="pil",             # "pil" | "pt" | "np" | "latent"
    return_dict=True,              # 返回 dict 还是 ImagePipelineOutput
)
```

### 4.2 视频 pipeline 的额外参数

```python
output = video_pipe(
    prompt,
    num_frames=...,                # ★ 视频独有
    num_videos_per_prompt=1,       # 类似 num_images_per_prompt
    # 其他参数同上
)
```

### 4.3 共同的内存优化方法

```python
pipe.enable_model_cpu_offload()           # 推荐首选：模块级 GPU/CPU 切换
pipe.enable_sequential_cpu_offload()      # 备选：子层粒度，更慢更省
pipe.enable_vae_tiling()                  # VAE 分 tile decode
pipe.enable_vae_slicing()                 # VAE 分 slice decode
pipe.enable_attention_slicing()           # （部分 pipeline）注意力分片
pipe.to("cuda")                           # 若不 offload，直接全量到 GPU
```

---

## 5. 与 `diffusion_engine` 的对比

### 5.1 我们的 pipeline 应该长什么样

`diffusion_engine/core/pipeline.py` 的接口应模仿 diffusers 的设计但不复制实现：

```python
class DiffusionPipeline:
    """最小限度 diffusion pipeline skeleton。"""

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda") -> "DiffusionPipeline":
        """加载预训练模型权重。注意：这不是 diffusers 的 from_pretrained，
        是我们自己的加载逻辑。"""
        ...

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 28,
        guidance_scale: float = 4.5,
        height: int = 1024,
        width: int = 1024,
        num_frames: int | None = None,  # None = image, int = video
        seed: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """主推理入口。返回像素空间的图像或视频 tensor。"""
        ...

    def enable_cpu_offload(self) -> None:
        """将模型模块按需在 CPU/GPU 间切换。"""
        ...

    def enable_vae_tiling(self) -> None:
        """启用 VAE 分 tile decode。"""
        ...
```

### 5.2 我们复制什么，不复制什么

| 方面 | diffusers | diffusion_engine |
|------|----------|-----------------|
| **API 签名风格** | 复制其清晰性 | `prompt`, `num_inference_steps`, `guidance_scale`, `height`, `width`, `num_frames` 参数保留 |
| **内部实现** | 不复制 | 自己的 scheduler、attention、CFG 合并、latent 管理 |
| **from_pretrained** | 不复制 | 我们的加载逻辑是自定义的（可能只加载 .safetensors 权重） |
| **CPU offload 机制** | 不复制具体逻辑 | 只实现简单版本（将整个模块 .to("cuda") / .to("cpu")） |
| **VAE decoder** | 不复制 diffusers 的 AutoencoderKL | 使用 diffusers 的 VAE 模块（`pipe.vae`）——因为 VAE 是预训练的，我们没必要自己重写 |

### 5.3 明确的使用边界（T13/T15 执行者必读）

**diffusers pipeline 的角色**：
- ✅ 验证模型能在我们的硬件上跑通（VRAM、latency 实测）
- ✅ 作为 reference 对照我们自己引擎的输出质量
- ✅ 记录峰值 VRAM 和端到端延迟，为优化实验提供 baseline
- ❌ **不是我们的成果**。不要在 final report 中说"我们实现了 SD3 pipeline"——我们实现的是自己的 `diffusion_engine/`，用 diffusers 做对照。

**在 T13/T14/T15 中的使用方式**：
```
1. 用 diffusers pipeline 跑一遍 → 确认模型能跑，记录 VRAM + latency
2. 用我们的 diffusion_engine/ 跑一遍（加载相同权重）→ 对比输出
3. 如果我们的引擎因为 toy 简化产生质量差异 → 在 T18 报告中标注 toy 局限
```

---

## 6. 给 T13/T15 执行者的明确指令

### 6.1 T13（图像 reference 脚手架）

- 准备 `experiments/reference_image_inference/infer_reference_image.py` 脚本，使用 `diffusers.StableDiffusion3Pipeline` 或 `diffusers.SanaPipeline`。
- **必须先跑 Sana**（<6GB，最友好），再尝试 SD3-Medium，FLUX schnell 作备选。
- 输出格式：PNG 文件 + JSON manifest（记录模型名、prompt、steps、CFG、latency、peak VRAM）。
- 不要试图把 diffusers 的输出包装成 `diffusion_engine` 的输出。

### 6.2 T15（视频 reference 脚手架）

- 准备 `experiments/reference_video_inference/infer_reference_video.py` 脚本，使用 `diffusers.LTXVideoPipeline` 优先。
- **必须先跑 LTX-Video**（2B + few-step + 已知 8GB 能跑），再尝试 CogVideoX-2B。
- 按 `09_视频latent和spacetime_patch.md` 第 6 节的降级路径执行。
- 输出格式：MP4 文件 + 完整字段记录（包括失败情况）。
- **3 次 OOM 即记录 blocker**，不要无限重试。

### 6.3 必须记录的字段（无论成功或失败）

对每一次推理尝试：
1. 模型名 + HF repo id
2. 分辨率 + 帧数（视频）
3. `num_inference_steps` + `guidance_scale`
4. `dtype` + CPU offload 模式
5. 峰值 VRAM（`torch.cuda.max_memory_allocated()` 或 `nvidia-smi`）
6. 推理耗时（wall clock，不含模型下载）
7. 输出文件路径
8. Blocker 描述（如有）

---

## 7. 本页结论

diffusers 为所有现代扩散模型提供了统一的 pipeline API：`from_pretrained()` 加载权重，`__call__()` 执行推理，`enable_*_offload()` 管理显存。图像和视频 pipeline 共享 `prompt` / `num_inference_steps` / `guidance_scale` / `height` / `width` 参数，视频 pipeline 额外多了 `num_frames`。我们的 `diffusion_engine/` 模仿其接口设计但不复制其内部实现——diffusers 是我们的 reference 和 profiling 工具，不是我们的成果。T13/T14 文生图和 T15 文生视频的 reference 实验均应以 diffusers pipeline 作为基线，并在结果文件中明确标注"使用 diffusers 作为 reference"。
