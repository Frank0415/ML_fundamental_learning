# text_engine_audit — 文本引擎静态审计实验

## 目的

在不修改 `minivLLM/` 的前提下，对引擎的 attention、KV cache、paged attention 三个关键模块进行静态审计，输出 JSON 结果报告。

## 脚本

| 脚本 | 审计目标 | 输出 |
|------|---------|------|
| `audit_attention.py` | `Attn` 类 + `Qwen3Attn` 接口匹配 + `Qwen3FFN` act_fn 接线 | `results/attention.json` |
| `audit_kv_cache.py` | `KVCache` 类 + 全仓库交叉引用 + Context 脚手架 | `results/kv_cache.json` |
| `audit_paged_attention.py` | 全仓库关键词扫描 + paged attention 实现状态判定 | `results/paged_attention.json` |

## 运行

```bash
cd multimodal/experiments/text_engine_audit
python audit_attention.py
python audit_kv_cache.py
python audit_paged_attention.py
```

## 约束

- 不 `import minivLLM`（避免触发构造错误）
- 不修改 `minivLLM/` 下任何文件
- 只做静态文件读取 + 正则/AST 分析
- Python 3.10+，无第三方依赖（仅 stdlib）

## 结果

结果写入 `results/` 目录，每个脚本输出一个 JSON 文件。关键结论：

- `attention.py` 的 `Attn` 类本身正确，但 `Qwen3Attn.__init__` 向其传输了不存在的参数 → **构建时 TypeError**
- `Qwen3FFN.act_fn = None` → **前向传播时 TypeError**
- `KVCache` 为 contiguous buffer，**无 forward 引用**，是 dead code
- `Context.set_context()` **从未被调用**，脚手架字段休眠
- **Paged attention: 未实现。** 仅有 `Context.block_tables` 和 `Config.kvcache_block_size` 占位字段，无 block 分配器、无 paged kernel、无调度器
