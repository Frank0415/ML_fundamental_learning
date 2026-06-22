# 多模态推理实验工作区 TODO

本文档记录 Atlas 计划从 Wave 1 到 Wave 5 的高层次阶段规划与实际完成状态。所有 Wave 现均已完成。

---

## Wave 1：目录骨架 + 引擎盘点 ✅ 完成

- [x] 创建 `multimodal/` 全目录骨架
- [x] 编写 `README.md`（9 章中文占位 → Wave 5 填实）
- [x] 编写 `TODO.md`（本文件）
- [x] 编写 `reports/engine_inventory.md`（18 模块静态审计，标识 2 阻塞 Bug + 2 未接线）
- [x] 确认 `minivLLM/` 顶层结构与已知阻塞
- [x] 编写 `reports/week_1.md`
- [x] 显存预算与模型选型矩阵（4 模型 Paper-based 估算）

目标达成：建立工作区基本框架，明确引擎现状。未修改任何 engine 代码。

---

## Wave 2：文本引擎审计 + Engine Patch ✅ 完成

- [x] 对 `minivLLM/` 执行完整文本引擎静态审计（4 个审计脚本）
- [x] 修复 5 个引擎 Bug：
  - [x] B1：`Qwen3Attn → Attn(S=, is_decode=)` 构造参数不兼容
  - [x] B2：`Qwen3FFN.act_fn = None` → 替换为 `SiluAndMul()`
  - [x] B3：`head_dim` 未通过 DecoderLayer 传参（推导值 64 vs 实际 128）
  - [x] B4：RoPE `chunk+cat` → 改为 HF `rotate_half` 方式
  - [x] B5：`rope_theta` 未传参（默认 10000 → HF 实际 1,000,000）
- [x] HF 对齐验证 → `verdict: IDENTICAL, max |diff|=8.2e-5, cos_sim=0.99999994`
- [x] Contiguous KV cache 接入 prefill/decode 路径（seq_len=1/8/64/512 全通过）
- [x] 生成 `experiments/text_engine_audit/results/` 审计报告
- [x] 生成 `docs/index.html` + `docs/01_*.html` / `docs/02_*.html`
- [x] 编写 `reports/week_2.md`
- [x] `paged_attention` 设计文档（`experiments/paged_attention_fix_or_impl/design.md`）

目标达成：引擎问题修复，HF IDENTICAL 对齐，contiguous KV cache 接入。静态文档页面发布。

---

## Wave 3：Paged Attention + 多模态 Token Pipeline ✅ 完成

- [x] 实现 PagedKVCache correctness-first 版本：
  - [x] `BlockManager` — block 分配/释放/统计
  - [x] `BlockTable` — 逻辑-物理 block 映射
  - [x] `PagedKVCache` — 分页 KV 读写
  - [x] `gather_kv_for_attention` — 物理 block 拼接（torch.gather fallback）
  - [x] `RequestState` / `Lifecycle` — 请求状态与生命周期
- [x] contiguous vs paged 对齐通过（torch.allclose）
- [x] 测试用例 → `experiments/paged_attention_fix_or_impl/tests/run_paged_kv_checks.py`
- [x] 性能基准 → `experiments/paged_attention_fix_or_impl/benchmarks/compare_contiguous_vs_paged.py`
- [x] 5 模块教学型 token pipeline（图像预处理 + Patch Embed + Visual Token + 序列构造 × 2 布局）
- [x] 两种 visual token 模式（tiny-vit-random / clip-reference-config-only）
- [x] 两种序列布局（bos_image_text / placeholder_expanded）
- [x] Shape 契约验证通过
- [x] 编写 `reports/week_3.md`
- [x] docs/ 静态页面（03_*.html 至 05_*.html）

目标达成：Paged KV 正确性通过，token pipeline shape 契约验证。无 CUDA kernel（仅 gather fallback）。

---

## Wave 4：inputs_embeds + VLM Demo + VLM Reference ✅ 完成（含降级）

- [x] `inputs_embeds` 路径正式接入 minivLLM 引擎：
  - [x] `Qwen3Model.forward` 新增 `inputs_embeds` 参数
  - [x] 双输入冲突拒绝（`ValueError`）
  - [x] 双空拒绝
  - [x] text_parity 通过（`max|diff|=0.00e+00`）
  - [x] HF parity 无回归
- [x] 最小 VLM demo（`run_minimal_vlm.py`）：
  - [x] `--mode text_parity` ✅
  - [x] `--mode invalid_dual_input` ✅
  - [x] `--mode prefill_only` ✅（random tiny-ViT + random projector）
  - [x] `--mode prefill_decode` ✅
- [x] VLM Reference 矩阵（`run_qwen_vl_reference.py`）：
  - [x] 4 模型 cascade 脚本（全部尝试，不提前退出）
  - [x] 4 份 .fail.json + processor_only_smoke.json
  - [ ] 4 个 VLM reference 模型成功运行 → **降级**：
    - [ ] Qwen3-VL-4B-Instruct → ❌ `device_map="auto"` 缺 `accelerate`
    - [ ] Qwen2.5-VL-3B-Instruct → ❌ `device_map="auto"` 缺 `accelerate`
    - [ ] InternVL3.5-4B → ❌ `InternVLChatConfig` 不被 AutoModel 识别
    - [ ] SmolVLM2-2.2B-Instruct → ❌ `device_map="auto"` 缺 `accelerate`
  - [x] 降级 smoke（tokenizer-only）✅
- [x] 编写 `reports/week_4.md` + `reports/week_6.md`
- [x] docs/ 静态页面（06_*.html 至 09_*.html）
- [x] 所有实验未修改 `minivLLM/` 引擎代码（除 Wave 2 的 5 个 Bug 修复）

目标达成：inputs_embeds 路径正式上线，最小 VLM demo 工程路径跑通。4 个 VLM reference 因环境限制（缺 `accelerate` / 无 CUDA）降级，降级路径验证通过。下一步：`pip install accelerate` 后重试。

---

## Wave 5：多模态 KV Cache 管理 + 收尾 ✅ 完成

- [x] 多模态 KV Cache key 管理实验：
  - [x] 共享模拟器 `mm_cache_simulator.py`（3 策略 hash + hit/miss 追踪）
  - [x] 策略 A（text-only）定义与实现
  - [x] 策略 B（text + image_hash）定义与实现
  - [x] 策略 C（full multimodal metadata）定义与实现
  - [x] `cache_key_design.md` 详细设计文档
  - [x] 7 类场景 benchmark（5 个可运行脚本 + 1 个说明占位）
  - [x] 关键验收：策略 A false_hit（`same_text_different_image`: A.false_hits=1）
  - [x] 5 份 JSON 结果 + 5 份 HTML 报告
- [x] 最终成果说明：
  - [x] `docs/10_最终成果说明.html`
  - [x] `reports/final_report.md`
- [x] README.md 全部 9 章填实（含可运行命令）
- [x] TODO.md 全部 Wave 状态更新（本文件）
- [x] 5 份证据文件（task-11-false-hit.txt 等）
- [x] 编写 `reports/week_5.md`

目标达成：mm cache 策略 A false_hit 关键发现确认，3 策略完整设计与实验完成。全部文档、报告、TODO 更新完毕。

---

## Final Verification Wave（占位）

> 待 Task 15 完成后由 atlas 触发 F1-F4 验证。

- [ ] F1：README 全部 9 章命令可运行验证
- [ ] F2：docs/index.html 全部 10 篇页面链接可达
- [ ] F3：reports/final_report.md 完成/未完成/限制项一致性验证
- [ ] F4：4 个 VLM reference 模型在安装 `accelerate` 后重新运行
