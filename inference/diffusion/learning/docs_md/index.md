# diffusion-inference 知识库

> 一份面向现代扩散模型推理的中文静态知识库。覆盖 rectified flow、DiT/MMDiT、文生图、文生视频、系统优化的完整技术栈。13 个页面，零外部依赖，`file://` 协议直接打开。

## 项目概览

### 01 任务总览
项目目标、路线选择、最终成果概览。为什么不从 DDPM/U-Net 学起。**T18 新建**

### 02 旧引擎审计
minivLLM 14 模块逐一审计。复用决策 C（完全不适合扩散推理）。

### 03 推理最小背景
Diffusion 推理数据流 9 要点。从 U-Net 到 DiT/MMDiT 的转向。

## 技术细节

### 04 Rectified Flow & Flow Matching
线性路径、矢量场、ODE 积分。与 score-based 的本质区别。

### 05 Diffusion Transformer 架构
DiT 核心组件：patchify、AdaLN-Zero、full attention、joint attention。

### 06 SD3 MMDiT
多模态 DiT：image + text token 联合处理。双流注意力设计。

### 07 FLUX 与现代文生图推理
FLUX 架构、few-step distillation、资源档位分析。

### 08 Sana 高效高分辨率生成
Sana 的 DC-AE、linear DiT、高分辨率效率。对中低显存更友好模型。

## 视频生成

### 09 Sora-Style 视频生成架构
Spacetime patch、3D VAE、视频 latent 结构。架构范式对比。

### 10 Wan / Hunyuan / CogVideoX / LTX 视频模型
开源视频模型横向对比：架构、显存、授权。资源档位排序。

## 优化与总结

### 11 扩散推理系统优化
6 项技术 + 实验数据。与 LLM KV cache 的根本差异。Attention O(N²) 是真实瓶颈。**T18 新建**

### 12 DiffusionGemma 推理
Discrete diffusion、自纠错机制、Encoder-Denoiser、vLLM 推理部署。

### 13 最终成果说明
全部产出汇总、运行命令、已知限制、下一步计划。**T18 新建**

## 快速入口

如果你是第一次来，推荐阅读顺序：**01 任务总览** → **03 推理背景** → **04 Rectified Flow** → **05 DiT 架构** → **11 系统优化** → **12 DiffusionGemma** → **13 成果说明**。

如果你想运行代码，查看 [13 最终成果说明](13_最终成果说明.md) 中的"如何运行"一节，或阅读顶层 `README.md`。
