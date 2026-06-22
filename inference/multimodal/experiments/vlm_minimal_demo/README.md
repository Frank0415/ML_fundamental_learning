# vlm_minimal_demo — VLM 最小可运行 Demo

本目录包含多模态推理的最小验证实验。

---

## 1. 输入嵌入路径验证（Task 8 / 9）

### run_minimal_vlm.py

验证 minivLLM 引擎的 `inputs_embeds` 路径与 `input_ids` 路径输出一致，并确认双输入冲突检测正确。

**运行：**

```bash
# text_parity: 验证 input_ids 与 inputs_embeds 路径输出一致
minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_minimal_vlm.py --mode text_parity

# invalid_dual_input: 验证双输入冲突检测
minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_minimal_vlm.py --mode invalid_dual_input
```

结果写入 `experiments/vlm_minimal_demo/results/text_parity.json` 与 `invalid_dual_input.json`。

**状态**：✅ 通过（Week 4 / Task 8）。

---

## 2. Reference 对照（Task 10）

### run_qwen_vl_reference.py

在 此前按受限显存假设做规划时，对 4 个 VLM 候选模型按主路径 → 稳定 fallback → 先进对照 → 轻量对照顺序依次尝试推理，记录显存、输出、失败原因。4 个模型全失败时自动执行降级 smoke（tokenizer-only）。

**4 个候选模型（硬编码，不扩展）**：

| Key | Model ID | 角色 |
|-----|----------|------|
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct` | 主路径 |
| `qwen2_5_vl_3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | 稳定 fallback |
| `internvl3_5_4b` | `OpenGVLab/InternVL3_5-4B` | 先进对照 |
| `smolvlm2_2b` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 轻量对照 |

**可复现命令**：

```bash
# 尝试全部 4 个模型（推荐）
minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_qwen_vl_reference.py \
  --image experiments/vlm_minimal_demo/sample_images/demo.jpg \
  --prompt "请描述这张图片。" \
  --prompt "图片里有什么文字？" \
  --max-new-tokens 64

# 仅尝试主模型
minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_qwen_vl_reference.py \
  --image experiments/vlm_minimal_demo/sample_images/demo.jpg \
  --model-key qwen3_vl_4b

# 仅尝试指定 fallback
minivLLM/.venv/bin/python experiments/vlm_minimal_demo/run_qwen_vl_reference.py \
  --model-key qwen2_5_vl_3b \
  --image experiments/vlm_minimal_demo/sample_images/demo.jpg
```

**结果文件**（写入 `experiments/vlm_minimal_demo/results/`）：

- `qwen3_vl_4b.{ok,fail}.json`
- `qwen2_5_vl_3b.{ok,fail}.json`
- `internvl3_5_4b.{ok,fail}.json`
- `smolvlm2_2b.{ok,fail}.json`
- `processor_only_smoke.json`（仅当 4 个模型全失败时生成）

**Gate 逻辑**：

- 成功路径：至少 1 个模型跑通 → 保存输出与显存记录。
- 降级路径：4 个模型全失败 → 保存 4 份失败日志 + tokenizer-only smoke 结果 → 退出码 0（smoke 通过时）。

**已知限制**：

- **网络依赖**：模型首次加载需从 Hugging Face Hub 下载权重（数 GB），网络不可达时所有模型均失败，仅降级 smoke 通过。
- **显存限制**：在无 CUDA 设备（如 macOS）上，4B 参数模型加载到 CPU 可能因内存不足失败；降级 smoke 仅需 tokenizer 配置，保证可运行。
- **GPU 需求**：4 个模型均需 ~4-8 GB GPU 显存（bf16），推荐 ≥NVIDIA GPU。CPU 推理在技术上可行但极慢且可能 OOM。
- **模型版本**：不锁定 transformers 版本，使用当前环境中已安装的版本（`transformers>=5.0`）。
- **图片格式**：仅测试过 JPEG/PNG；其他格式未验证。
- **不保存权重**：模型权重不写入仓库，由 transformers 缓存管理。
