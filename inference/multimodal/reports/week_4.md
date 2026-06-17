# Week 4 — Wave 3 进展报告

> 日期: 2026-06-07
> 关联任务: Task 8 — inputs_embeds 路径正式接入

---

## Task 8: inputs_embeds 路径接入

### 修改摘要

**引擎改动（仅 `minivLLM/minivllm/model/qwen3.py`）：**

1. `Qwen3Model.forward` 新增可选参数 `inputs_embeds: torch.Tensor | None = None`
2. 双输入冲突检测：若 `input_ids is not None` 且 `inputs_embeds is not None`，抛 `ValueError("...cannot accept both input_ids and inputs_embeds...")`
3. 双空检测：若两者均为 None，抛 `ValueError("...requires at least one of input_ids or inputs_embeds...")`
4. `inputs_embeds` 分支：跳过 `embed_tokens`，直接以 `inputs_embeds` 作为 `hidden_states` 初始值
5. `Qwen3.forward` 同样透传 `inputs_embeds` 参数

**未修改文件**：`attention.py`、`activation.py`、`norm.py`、`rope.py`、`kv_cache.py`、`config.py`、`context.py`、`linear.py`、`embedding.py`、`validate_model.py`

### 测试结果

| 测试 | 结果 |
|------|------|
| `validate_model.py --full` | ✅ ALL CHECKS PASSED |
| `validate_model.py --compare-hf --full` | ✅ verdict=IDENTICAL, max\|diff\|=8.0e-5, cos_sim≈1.0 |
| `run_minimal_vlm.py --mode text_parity` | ✅ PASS, max\|diff\|=0.00e+00 |
| `run_minimal_vlm.py --mode invalid_dual_input` | ✅ PASS, ValueError caught with keyword match |

### 状态

- inputs_embeds 路径已正式接入 minivLLM 引擎
- text_parity 通过（input_ids == embed_tokens(input_ids) 路径）
- invalid_dual_input 正确拒绝
- HF parity 无回归
- Task 9（视觉 encoder 接入）的前置条件已满足

---

## Task 10: VLM Reference 矩阵（追加）

> 本段为 Wave 4 / Task 14 追加内容。原始 Week 4 报告写入于 Task 8 完成时。

### 目的

在 inputs_embeds 路径接入 minivLLM 之后，用 HF transformers 原生的 `model.generate()` 对 4 个 VLM 候选模型做 reference 对照推理，建立主路径 → 稳定 fallback → 先进对照 → 轻量对照的降级链。

### 4 个候选模型

| Model Key | Model ID | 角色 |
|-----------|----------|------|
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct` | 主路径 |
| `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | 稳定 fallback |
| `internvl3_5_4b` | `OpenGVLab/InternVL3_5-4B` | 先进对照 |
| `smolvlm2_2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 轻量对照 |

### 实验结果

执行环境：macOS (Apple Silicon MPS), transformers 5.8.0, torch 2.11.0, CUDA=False。

**4 个模型全部加载失败**。核心阻塞原因是 `device_map="auto"` 强制依赖 `accelerate` 包，而 `minivLLM/.venv` 未安装该包。InternVL3.5-4B 额外遇到 `InternVLChatConfig` 不被 `AutoModelForImageTextToText` 识别的问题。

**降级路径生效**：降级 smoke test 使用 `Qwen/Qwen3-0.6B` tokenizer-only 模式，tokenizer 加载正确，11 token round-trip 解码正确。

详细结果见完整 `reports/week_6.md`。

### 状态

- inputs_embeds 路径已就绪，为 visual token embedding 拼接进 LLM 提供了正确的输入通道
- 4 个 VLM reference 模型因环境依赖（缺 `accelerate`）在 macOS MPS 下全部失败，降级 smoke 通过
- 整体进度：纯文本路径 HF-IDENTICAL，inputs_embeds 路径 text_parity 通过，PagedKV correctness-first 通过，下一步需解决 visual tower 真实权重加载（Task 9）
