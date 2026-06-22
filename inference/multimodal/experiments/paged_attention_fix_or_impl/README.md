# Paged Attention：固化最小设计与验证

## 项目目的

在 minivLLM 推理框架中，为 `kvcache_block_size=16`（见 `minivLLM/minivllm/config.py`）的配置固化一套 **最小可用的 Paged KV Cache 设计与验证流程**。目标不是实现一个工业级 scheduler，而是确保：

1. **基础正确性**：paged attention 输出的 logits 与 contiguous KV 参考实现的 logits 在 fp32 精度下逐 token 对齐（`allclose(atol=1e-5, rtol=1e-5)`）。
2. **显存安全**：在 中低显存 GPU 上，单个推理请求的 KV 分配不会溢出；碎片化可控。
3. **接口清晰**：`BlockManager` / `BlockTable` / `RequestState` 的职责边界明确，能被后续 scheduler 或 prefiller 直接消费。

## 最小接受标准（Minimum Acceptance Criteria）

| 编号 | 验收项 | 判定方式 | 备注 |
|------|--------|----------|------|
| MAC-1 | contiguous KV 与 paged KV 的 logits 对齐 | `allclose(atol=1e-5, rtol=1e-5)` | 单个 prefill + N 步 decode，随机 token 序列 |
| MAC-2 | `wasted_slots` ≤ `block_size`（每个请求最多浪费一个 block） | 统计指标输出 | 碎片化可接受上限 |
| MAC-3 | `allocated_blocks` + `free_blocks` + `reserved_system_blocks` = `total_blocks` | 统计指标一致性检查 | 无泄漏 |
| MAC-4 | 至少 1 个 VLM 模型在 中低显存 GPU 上完成 prefill + decode 循环（≤ 512 tokens） | end-to-end 运行 | 优先 Qwen3-VL-4B-Instruct |
| MAC-5 | `gather_kv_for_attention()` 返回的 kv 块能被 `ref_attn()` 逐 token 对齐 | 单元测试 | 不依赖完整 forward |

## 运行命令（占位）

```bash
# 环境激活
conda activate minivllm

# 单元测试：BlockManager 分配/释放
cd multimodal/experiments/paged_attention_fix_or_impl
python -m pytest tests/ -v -k "block"

# 对齐验证：contiguous vs paged logits
python tests/test_kv_alignment.py --model hf_model_id --n-steps 8

# 显存压力：在中档显存卡的上限 + 碎片化统计
python tests/test_memory_budget.py --gpu-memory-gb 12 --model hf_model_id

# 完整设计文档
cat design.md
```

## Task 6：最小 paged KV 路径运行方式

```bash
cd /Users/franksair/Documents/learning_ML/inference

PYTHONPATH=multimodal/minivLLM multimodal/minivLLM/.venv/bin/python \
  multimodal/experiments/paged_attention_fix_or_impl/tests/run_paged_kv_checks.py \
  --block-sizes 16 32

PYTHONPATH=multimodal/minivLLM multimodal/minivLLM/.venv/bin/python \
  multimodal/experiments/paged_attention_fix_or_impl/benchmarks/compare_contiguous_vs_paged.py
```

输出文件：

- `experiments/paged_attention_fix_or_impl/results/paged_kv_checks.txt`
- `experiments/paged_attention_fix_or_impl/results/contiguous_vs_paged.txt`
- `.omo/evidence/task-6-paged-tests.txt`
- `.omo/evidence/task-6-paged-compare.txt`

当前实现保持 correctness-first：paged cache 在 attention 前通过 `gather_kv_for_attention()` 拼回 contiguous K/V，然后复用 Task 5 已验证的 `Qwen3Model.forward(..., kv_cache, is_prefill)` decode 路径；不改 `minivLLM/`，不写 CUDA/Triton kernel。

## 目录结构

```
paged_attention_fix_or_impl/
├── README.md          # 本文件
├── design.md          # 接口设计、指标、量化门槛、非目标
├── tests/             # 单元测试与对齐测试
├── benchmarks/        # 对齐结果、显存剖面
└── results/           # 实验输出
```

## 依赖

- `minivLLM/minivllm/config.py` 提供的 `kvcache_block_size=16`、`num_kvcache_blocks=-1`（自动）、`gpu_memory_utilization=0.9`
- `torch` ≥ 2.0，CUDA ≥ 11.8
- `transformers`（仅用于加载 HF 模型权重与 config）
- 本机约 NVIDIA GPU

## 非目标（明确排除）

本实验 **不做** 以下事情：

- 不做 scheduler / continuous batching
- 不做 prefix sharing / cascade attention
- 不写定制 CUDA kernel（使用 PyTorch 原生算子即可）
- 不做训练、不做 LoRA、不做量化感知调优
- 不做分布式 / tensor parallel（单卡即可）
