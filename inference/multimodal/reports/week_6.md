# Week 6 — Wave 4 进展报告

> 日期: 2026-06-07
> 关联任务: Task 10 — VLM Reference 矩阵

---

## Task 10: VLM Reference 矩阵（中等显存配置 约束下的小型 VLM reference）

### 目的

在 此前按受限显存假设做规划时，对 4 个 VLM 候选模型进行 reference 对照推理，建立主路径 → 稳定 fallback → 先进对照 → 轻量对照的逐级降级链。

### 实现

**脚本**：`experiments/vlm_minimal_demo/run_qwen_vl_reference.py`

4 个模型候选（硬编码，与 plan 完全一致）：

| Model Key | Model ID | 角色 |
|-----------|----------|------|
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct` | 主路径 |
| `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | 稳定 fallback |
| `internvl3_5_4b` | `OpenGVLab/InternVL3_5-4B` | 先进对照 |
| `smolvlm2_2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 轻量对照 |

**Gate 逻辑**：
- 成功路径：至少 1 个模型跑通 → 保存输出与显存记录。
- 降级路径：4 个模型全失败 → 保存 4 份失败日志 + `processor_only_smoke.json`（tokenizer-only）。

### 运行结果

**执行环境**：macOS (Apple Silicon, MPS), transformers 5.8.0, torch 2.11.0, CUDA=False

**主路径状态：降级（degraded）** — 4 个模型全部加载失败。

| 模型 | 状态 | 失败原因 |
|------|------|----------|
| Qwen3-VL-4B-Instruct | ❌ | `ValueError`: `device_map="auto"` 需要 `accelerate` 包 |
| Qwen2.5-VL-3B-Instruct | ❌ | `ValueError`: `device_map="auto"` 需要 `accelerate` 包 |
| InternVL3.5-4B | ❌ | `ValueError`: 配置类 `InternVLChatConfig` 不被 `AutoModelForImageTextToText` 识别 |
| SmolVLM2-2.2B-Instruct | ❌ | `ValueError`: `device_map="auto"` 需要 `accelerate` 包 |

**降级 smoke：✅ 通过**
- Model: `Qwen/Qwen3-0.6B`（tokenizer-only，不加载 VLM 权重）
- Tokenizer: `Qwen2Tokenizer`, vocab=151643
- Smoke test: 11 tokens, round-trip 解码正确

### 降级原因分析

1. **核心阻塞**：`minivLLM/.venv` 未安装 `accelerate` 包。`from_pretrained(device_map="auto")` 强制要求 `accelerate`，3 个 Qwen/SmolVLM 模型因此失败。
2. **InternVL 额外问题**：`InternVL3.5-4B` 的配置文件类 `InternVLChatConfig` 不在 `AutoModelForImageTextToText` 的已知模型类型列表中。该模型可能需要特定的加载器或更新的 transformers 版本。
3. **无 CUDA**：当前环境为 macOS + MPS，无 NVIDIA GPU。即使 `accelerate` 安装后，模型仍需下载权重并在 CPU/MPS 上运行，可能进一步面临内存或性能问题。

### 产物

**结果文件**（`experiments/vlm_minimal_demo/results/`）：
- `qwen3_vl_4b.fail.json`
- `qwen2_5_vl_3b.fail.json`
- `internvl3_5_4b.fail.json`
- `smolvlm2_2b.fail.json`
- `processor_only_smoke.json`

**证据文件**（`.omo/evidence/`）：
- `task-10-primary-reference.txt` — 主路径状态（degraded）
- `task-10-reference-success.txt` — 成功证据（降级路径生效）

**文档**：
- `experiments/vlm_minimal_demo/README.md` — 新增 "Reference 对照" 段

### 修复建议（非本任务范围）

```bash
# 安装 accelerate 以启用 device_map="auto"
pip install accelerate

# InternVL 可能需要特定加载器，或等待 transformers 更新
# 替代方案：使用 InternVL 专用加载器
```

### 状态

- [x] `run_qwen_vl_reference.py` 实现完毕（4 模型 cascade + 降级 smoke）
- [x] 4 个模型全部尝试（不因主模型失败而跳过 fallback）
- [x] 失败原因详细记录（error_type + error_message + traceback + GPU 信息）
- [x] 降级路径生效（processor-only smoke 通过）
- [x] Gate 结果写入 evidence
- [x] README 文档更新
- [ ] 等待 `accelerate` 安装后重新验证（留待 Wave 5/Task 12）
