# 多模态 KV Cache 管理实验

## 实验目标

本实验量化评估三种 cache key 策略在多模态推理场景下的 false hit 风险。

### 核心问题

纯文本 LLM 常用 **text-only prefix cache**，以文本 token 序列的 hash 作为 cache key, 相同文本前缀的请求可复用 KV cache。

**但在多模态推理中**, 请求的视觉输入 (图像) 可能不同而文本前缀相同。此时 text-only cache 会错误地判定"命中", 导致模型基于不匹配的图像 KV 做解码，即 **false hit**。

### 三种策略

| 策略 | 简称 | Hash 内容 |
|------|------|-----------|
| A | text_only | 仅 `hash(text_token_ids)` |
| B | text+img_hash | `hash(text_token_ids, sha256(image_bytes))` |
| C | full_multimodal | `hash(model, config, text, image, size, grid, layout, order...)` |

详见 [`cache_key_design.md`](./cache_key_design.md)。

### 七类测试场景

1. **same_text_same_image** - 相同文本 + 相同图片 → 应命中
2. **same_text_different_image** - 相同文本 + 不同图片 → **关键**: A false_hit, B/C miss
3. **same_image_different_question** - 相同图片 + 不同文本 → 应 miss
4. **same_image_different_resize** - 同一图片不同分辨率 → 应 miss
5. **multi_image_same_order** - 多图相同顺序 → 应命中
6. **multi_image_different_order** - 多图不同顺序 → 应 miss
7. **same_video_different_frame_sampling** - 视频帧采样差异 (说明占位)

## 如何运行

### 环境要求

- Python 3.9+ (无需 PyTorch/GPU/transformers)
- 系统 `hashlib`, `json` (均标准库)

### 运行全部 benchmark

```bash
cd inference/multimodal/experiments/mm_kv_cache_management/

# 确保参考图片存在
ls ../vlm_minimal_demo/sample_images/demo.jpg

# 逐个运行 (全部独立可执行)
python3 benchmark_same_text_same_image.py
python3 benchmark_same_text_different_image.py
python3 benchmark_same_image_different_question.py
python3 benchmark_same_image_different_resize.py
python3 benchmark_multi_image_order.py

# 或一次性运行全部
for f in benchmark_*.py; do echo "=== $f ===" && python3 "$f"; done
```

### 输出

- **JSON 结果**: `results/<case_name>_results.json`
- **HTML 报告**: `results/<case_name>_report.html`
- **终端摘要**: 每个 benchmark 运行后打印关键指标

### 验收标准 (硬性)

- `same_text_different_image` 中 **策略 A: `false_hit_count > 0`**
- 同一 case 中 **策略 B/C: `false_hit_count == 0`**

## 目录结构

```
mm_kv_cache_management/
├── README.md                              ← 本文件
├── cache_key_design.md                    ← 三策略详细定义
├── mm_cache_simulator.py                  ← 共享模拟器
├── benchmark_same_text_same_image.py      ← Case 1
├── benchmark_same_text_different_image.py ← Case 2 (关键)
├── benchmark_same_image_different_question.py ← Case 3
├── benchmark_same_image_different_resize.py   ← Case 4
├── benchmark_multi_image_order.py         ← Case 5+6+7
└── results/
    ├── *_results.json
    └── *_report.html
```
