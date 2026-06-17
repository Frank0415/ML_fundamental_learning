# 多模态 KV Cache Key 策略设计文档

## 概述

本文件定义三种 cache key 策略的详细 hash 逻辑、匹配规则与 false hit 判定标准。

## 通用数据结构

每次多模态推理请求使用以下字段描述:

| 字段 | 类型 | 说明 |
|------|------|------|
| `text_token_ids` | `list[int]` | 文本 token ID 序列 (含 `<image>` 占位符) |
| `image_bytes_list` | `list[bytes]` | 每张图片的原始字节 |
| `model_id` | `str` | 模型标识 (如 `qwen3-vl-4b`) |
| `tokenizer_config_hash` | `str` | tokenizer 配置版本 hash |
| `processor_config_hash` | `str` | 图像处理器配置版本 hash |
| `original_sizes` | `list[(H,W)]` | 原始图片尺寸 |
| `resized_sizes` | `list[(H,W)]` | 模型 resize 后的尺寸 |
| `patch_grids` | `list[(ph,pw)]` | 视觉 patch 网格 |
| `num_visual_tokens_list` | `list[int]` | 每张图产生的 visual token 数 |
| `placeholder_layout` | `str` | 文本中占位符的布局模式 |
| `multi_image_order` | `str` | 多图的有序标识 (如 `"image_0,image_1"`) |
| `video_frame_sampling_meta` | `str` | 视频帧采样元信息 (非视频时为 `"none"`) |

---

## 策略 A: `text_only_cache_key`

### Hash 构成

```
key = SHA-256(json(text_token_ids))
```

### 匹配规则

- key 完全相等 → 判定为 **命中 (hit)**。
- key 不等 → **未命中 (miss)**。

### False Hit 定义

当请求的 **语义图像指纹** (所有 `image_bytes` 的 SHA-256 有序拼接) 与 cache 条目不同,
但 text_token_ids 相同导致 key 匹配 → **false hit**。

### 示例

| Cache 条目 | 新请求 | key 匹配? | 语义图像匹配? | 判定 |
|------------|--------|-----------|--------------|------|
| text=[1,2,3], img=[A] | text=[1,2,3], img=[A] | ✅ | ✅ | true_hit |
| text=[1,2,3], img=[A] | text=[1,2,3], img=[B] | ✅ | ❌ | **false_hit** |
| text=[1,2,3], img=[A] | text=[4,5,6], img=[A] | ❌ | - | safe_miss |

**关键**: 策略 A 最危险——相同 text 但不同 image 会产生 false hit, 导致 KV cache 语义错配。

---

## 策略 B: `text_plus_image_hash_cache_key`

### Hash 构成

```
key = SHA-256(
    json(text_token_ids)
    || SHA-256(image_bytes[0])
    || SHA-256(image_bytes[1])
    || ...
)
```

其中 `||` 表示字节级拼接, `SHA-256(image_bytes)` 指示每张图的完整字节 hash。

### 匹配规则

- key 完全相等 → 命中。
- key 不等 → 未命中。

### False Hit 可能性

策略 B 将图像字节的 SHA-256 纳入 key。若两张图的 `SHA-256(image_bytes)` 不同 (即图不同), key 必不同 → 不会 false hit。

唯一的 false hit 风险是 **SHA-256 碰撞** (概率 ≈ 2⁻²⁵⁶, 可忽略)。在本次实验中, 策略 B 的 `false_hit_count` 始终为 0。

---

## 策略 C: `full_multimodal_cache_key`

### Hash 构成

```
key = SHA-256(
    model_id
    || tokenizer_config_hash
    || processor_config_hash
    || json(text_token_ids)
    || SHA-256(image_bytes[0])
    || SHA-256(image_bytes[1])
    || ...
    || json(original_sizes)
    || json(resized_sizes)
    || json(patch_grids)
    || json(num_visual_tokens_list)
    || placeholder_layout
    || multi_image_order
    || video_frame_sampling_meta
)
```

### 匹配规则

- key 完全相等 → 命中。
- key 不等 → 未命中。

### 策略 C 的额外保护

除文本 + 图像字节外, 策略 C 额外纳入:

1. **模型标识** (`model_id`): 不同模型间的 KV cache 不可复用。
2. **配置版本** (`tokenizer_config_hash`, `processor_config_hash`): tokenizer/processor 升级后 cache 自动失效。
3. **图像处理参数** (`original_sizes`, `resized_sizes`, `patch_grids`): 同一张图不同分辨率 → 不同 visual token 数 → 不应复用。
4. **视觉 token 数** (`num_visual_tokens_list`): 直接反映序列长度差异。
5. **多图顺序** (`multi_image_order`): `[imgA, imgB]` ≠ `[imgB, imgA]`。
6. **视频帧采样** (`video_frame_sampling_meta`): 不同采样率/帧索引的 cache 不应复用。

### False Hit 可能性

策略 C 在策略 B 基础上增加了 7 项元数据, false hit 概率进一步降低 (理论碰撞概率 < 2⁻²⁵⁶)。
在本次实验中, 策略 C 的 `false_hit_count` 始终为 0。

---

## 三策略对比总结

| 维度 | 策略 A (text-only) | 策略 B (text+img) | 策略 C (full) |
|------|-------------------|-------------------|---------------|
| Hash 输入 | 仅 text_token_ids | text + image SHA-256 | 全量元数据 |
| "同文不同图" 判定 | ❌ false hit | ✅ safe miss | ✅ safe miss |
| "同图不同尺寸" 判定 | ❌ false hit | ❌ false hit | ✅ safe miss |
| "同图不同顺序" 判定 | ❌ false hit | ❌ false hit | ✅ safe miss |
| 碰撞风险 | 高 (语义错配) | 极低 (SHA-256) | 极低 (SHA-256) |
| Cache 命中率 | 高 (不准确) | 中 | 低 (最严格) |

**核心结论**: text-only prefix cache 策略 (A) 对多模态推理是**不安全**的——相同文本前缀但不同图像输入时, A 会错误地复用 KV cache, 导致模型输出基于错误的 visual context。
