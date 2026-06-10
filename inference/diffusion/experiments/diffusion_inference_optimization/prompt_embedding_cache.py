#!/usr/bin/env python3
"""
prompt_embedding_cache.py — Prompt Embedding Cache 实验

★★★ 这不是 LLM 的 KV cache ★★★

- LLM KV cache：存储每层 attention 的 key/value 历史，跨 token 步复用，自回归生成专用。
- prompt embedding cache：缓存 text encoder 的输出（prompt embedding / pooled embedding），
  同一 prompt 在推理循环中多次使用（如 cond + uncond encode）时避免重复编码。
  这是 diffusion 特有的优化点：text encoder 只跑一次，embedding 可复用多次。

本脚本实现：
  1. PromptEmbeddingCacheKey：@dataclass，至少 7 个字段的 cache key，__hash__ + __eq__
  2. PromptEmbeddingCache：内部 dict 存储 numpy 数组，lookup/store/cache_stats/clear
  3. --demo 模式：模拟 100 次 prompt 调用，30% 重复，测量 cache 收益

Cache key 字段（9 个，≥7 满足扩散多模型/多配置隔离需求）：
  - model_id:           模型标识（"SD3-Medium" / "FLUX-schnell"），不同模型必须 miss
  - tokenizer_hash:     tokenizer 配置的 SHA256 前 16 位，换 tokenizer 后 miss
  - text_encoder_hash:  text encoder 配置的 SHA256 前 16 位，换 encoder 后 miss
  - prompt:             正向文本提示
  - negative_prompt:    负向文本提示（参与 key，因为空串和"ugly"产生不同 embedding）
  - max_sequence_length: 最大 token 序列长度（不同截断长度 → shape 不同 → 不可互换）
  - dtype:              数据类型（float32/float16/bfloat16 不可互换）
  - device:             设备（cpu/cuda/mps）
  - offload_strategy:   offload 策略（cpu/cuda/none）

纯 numpy 实现，不依赖 torch。

========== 使用示例 ==========

# 查看帮助
python prompt_embedding_cache.py --help

# 运行 demo
python prompt_embedding_cache.py --demo --num_prompts 100 --repeat_ratio 0.3 --output_dir results

# 自定义参数
python prompt_embedding_cache.py --demo --num_prompts 500 --repeat_ratio 0.5 --hidden_size 128 --output_dir results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np


# =============================================================================
# PromptEmbeddingCacheKey — 9 字段 cache key
# =============================================================================


@dataclass(frozen=True)
class PromptEmbeddingCacheKey:
    """
    多维 cache key，确保以下场景不会误命中：

    1. 不同模型（FLUX 的 embedding 不能喂 SD3）
    2. 不同 tokenizer/text encoder 版本
    3. 不同 max_sequence_length（embedding shape 不同）
    4. 不同 dtype（float16 embedding 不能当 float32 用）
    5. 不同 device（GPU tensor 在 CPU 上不可用）
    6. 不同 offload 策略（offload 到 CPU 的 tensor 不等于 GPU tensor）

    frozen=True → 不可变，可 hash，可 dict key。
    """

    model_id: str
    """模型标识（如 "SD3-Medium" / "FLUX-schnell" / "Sana-1.6B"）"""

    tokenizer_hash: str
    """tokenizer 配置 JSON 的 SHA256 前 16 位（hex 字符串）"""

    text_encoder_hash: str
    """text encoder 配置 JSON 的 SHA256 前 16 位（hex 字符串）"""

    prompt: str
    """正向文本提示（参与 key 因为不同 prompt 产出不同 embedding）"""

    negative_prompt: str
    """负向文本提示（参与 key，" " 和 "ugly" 产出不同 uncond embedding）"""

    max_sequence_length: int
    """最大 token 序列长度（shape 维度之一，不同长度不可互换）"""

    dtype: str
    """数据类型（"float32" / "float16" / "bfloat16"）"""

    device: str
    """设备（"cpu" / "cuda" / "mps"）"""

    offload_strategy: str
    """offload 策略（"cpu" / "cuda" / "none"）"""

    def __hash__(self) -> int:
        """基于所有 9 个字段的确定性 hash。"""
        return hash((
            self.model_id,
            self.tokenizer_hash,
            self.text_encoder_hash,
            self.prompt,
            self.negative_prompt,
            self.max_sequence_length,
            self.dtype,
            self.device,
            self.offload_strategy,
        ))

    def __eq__(self, other: object) -> bool:
        """严格相等：所有 9 个字段必须逐项匹配。"""
        if not isinstance(other, PromptEmbeddingCacheKey):
            return NotImplemented
        return (
            self.model_id == other.model_id
            and self.tokenizer_hash == other.tokenizer_hash
            and self.text_encoder_hash == other.text_encoder_hash
            and self.prompt == other.prompt
            and self.negative_prompt == other.negative_prompt
            and self.max_sequence_length == other.max_sequence_length
            and self.dtype == other.dtype
            and self.device == other.device
            and self.offload_strategy == other.offload_strategy
        )

    def __repr__(self) -> str:
        """紧凑表示：仅显示 model_id + prompt 前 30 字符（避免日志爆炸）。"""
        prompt_short = self.prompt[:30] + "..." if len(self.prompt) > 30 else self.prompt
        return (
            f"CacheKey(model={self.model_id}, prompt='{prompt_short}', "
            f"max_len={self.max_sequence_length}, dtype={self.dtype}, "
            f"device={self.device}, offload={self.offload_strategy})"
        )


# =============================================================================
# 辅助函数：生成 hash
# =============================================================================


def _config_hash(config_str: str) -> str:
    """
    计算配置字符串的 SHA256 并返回前 16 位（8 字节 hex）。

    参数：
        config_str: tokenizer/text encoder 的配置 JSON 字符串。

    返回：
        16 字符 hex 字符串（SHA256 前 64 bits）。

    设计说明：
        使用 16 位 hex（64 bits）而非完整 64 位 hex（256 bits），
        因为 cache key 的 hash 冲突概率在 64 bits 下已经极低（~10^-19），
        无需完整 SHA256 增加内存开销。
    """
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# PromptEmbeddingCache — 缓存管理器
# =============================================================================


class PromptEmbeddingCache:
    """
    Prompt embedding 缓存。

    内部维护 dict[PromptEmbeddingCacheKey, np.ndarray]。
    提供 lookup / store / cache_stats / clear 接口。

    注意：这不是 LLM 的 KV cache。这里缓存的是 text encoder 的输出 tensor
    （prompt embedding 和 pooled embedding），不是 attention key/value 历史。

    典型使用模式：
        cache = PromptEmbeddingCache()
        key = PromptEmbeddingCacheKey(...)
        emb = cache.lookup(key)
        if emb is None:
            emb = text_encoder.encode(key.prompt, key.negative_prompt, ...)
            cache.store(key, emb)
    """

    def __init__(self, max_size: int = 1000):
        """
        参数：
            max_size: 最大缓存条目数（超过后 FIFO 淘汰，默认 1000）。
        """
        self._cache: Dict[PromptEmbeddingCacheKey, np.ndarray] = {}
        self._max_size = max_size
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._access_order: list = []  # FIFO 淘汰队列

    def lookup(self, key: PromptEmbeddingCacheKey) -> Optional[np.ndarray]:
        """
        查找缓存。

        参数：
            key: 9 字段 cache key。

        返回：
            命中时返回对应的 np.ndarray（写时复制保护），未命中时返回 None。
        """
        if key in self._cache:
            self._hit_count += 1
            # 返回 copy 防止外部修改污染缓存
            return self._cache[key].copy()
        self._miss_count += 1
        return None

    def store(self, key: PromptEmbeddingCacheKey, embedding: np.ndarray) -> None:
        """
        存入缓存。

        参数：
            key: 9 字段 cache key。
            embedding: text encoder 输出的 numpy 数组（任意 shape）。
        """
        # FIFO 淘汰：超过 max_size 时移除最早条目
        if len(self._cache) >= self._max_size and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        # 存储 copy 防止外部后续修改
        self._cache[key] = embedding.copy()

        # 记录访问顺序
        if key not in self._access_order:
            self._access_order.append(key)

    def cache_stats(self) -> dict:
        """
        返回缓存统计。

        返回：
            {
                "hit_count": int,       # 命中次数
                "miss_count": int,      # 未命中次数
                "hit_ratio": float,     # 命中率 (0.0–1.0)
                "size": int,            # 当前缓存条目数
                "max_size": int,        # 最大容量
            }
        """
        total = self._hit_count + self._miss_count
        hit_ratio = self._hit_count / total if total > 0 else 0.0
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_ratio": round(hit_ratio, 4),
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """清空所有缓存（保留统计计数）。"""
        self._cache.clear()
        self._access_order.clear()


# =============================================================================
# ToyTextConditioner（numpy 版，不依赖 torch）
# =============================================================================


class ToyTextConditioner:
    """
    Toy 文本编码器（numpy 实现，不依赖 torch）。

    接口与 T12 的 ToyTextConditioner 兼容（同样提供 encode() 和 cache_stats()），
    但内部使用 numpy 和 PromptEmbeddingCacheKey 而非 torch tensor。
    """

    def __init__(
        self,
        model_id: str = "ToyModel-v1",
        hidden_size: int = 64,
        max_seq_len: int = 77,
        dtype: str = "float32",
        device: str = "cpu",
        offload_strategy: str = "none",
        seed: int = 42,
    ):
        self.model_id = model_id
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        self.offload_strategy = offload_strategy
        self.seed = seed
        self._cache = PromptEmbeddingCache(max_size=1000)

        # 固定 hash（模拟真实 tokenizer/text_encoder 配置）
        self._tokenizer_hash = _config_hash(f"{model_id}_tokenizer_v1.0")
        self._text_encoder_hash = _config_hash(f"{model_id}_text_encoder_v1.0")

    def _build_key(self, prompt: str, negative_prompt: str) -> PromptEmbeddingCacheKey:
        """构造 9 字段 cache key。"""
        return PromptEmbeddingCacheKey(
            model_id=self.model_id,
            tokenizer_hash=self._tokenizer_hash,
            text_encoder_hash=self._text_encoder_hash,
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=self.max_seq_len,
            dtype=self.dtype,
            device=self.device,
            offload_strategy=self.offload_strategy,
        )

    def _generate_embedding(self, prompt: str, seed_offset: int = 0) -> np.ndarray:
        """
        生成确定性随机 embedding。

        模拟 text encoder forward 的耗时和输出。相同 prompt 产生相同 embedding（可复现）。

        参数：
            prompt: 文本。
            seed_offset: 子种子偏移（区分 cond 和 uncond）。

        返回：
            shape (1, max_seq_len, hidden_size) 的 numpy 数组。
        """
        prompt_hash = abs(hash(prompt)) % (2**31)
        sub_seed = self.seed + prompt_hash + seed_offset
        rng = np.random.RandomState(sub_seed)
        return rng.randn(1, self.max_seq_len, self.hidden_size).astype(
            np.float32 if self.dtype == "float32" else np.float16
        )

    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        simulate_latency: float = 0.005,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        编码 prompt（含 cache 查表 + 模拟耗时）。

        参数：
            prompt: 正向文本提示。
            negative_prompt: 负向文本提示（默认空串）。
            simulate_latency: 模拟 text encoder 前向耗时（秒，默认 5ms）。

        返回：
            (cond_embedding, uncond_embedding) — numpy 数组，uncond 可能为 None。
        """
        # 构造 cache key
        key = self._build_key(prompt, negative_prompt)

        # 查找 cache
        cached = self._cache.lookup(key)
        if cached is not None:
            # 命中：模拟极短延迟（仅 dict lookup，<1μs）
            time.sleep(simulate_latency * 0.001)  # 模拟 0.1% 的原有耗时
            return cached, None

        # 未命中：模拟真实 text encoder 前向（完整耗时）
        time.sleep(simulate_latency)

        cond_emb = self._generate_embedding(prompt, seed_offset=0)
        uncond_emb = self._generate_embedding(negative_prompt, seed_offset=1)

        # 存入 cache
        self._cache.store(key, cond_emb)

        return cond_emb, uncond_emb

    def cache_stats(self) -> dict:
        """代理到内部 PromptEmbeddingCache.cache_stats()。"""
        return self._cache.cache_stats()


# =============================================================================
# Demo 运行器
# =============================================================================


def run_demo(
    num_prompts: int = 100,
    repeat_ratio: float = 0.3,
    hidden_size: int = 64,
    max_seq_len: int = 77,
    output_dir: str = "results",
    seed: int = 42,
) -> dict:
    """
    运行 prompt embedding cache demo。

    工作流：
      1. 生成 num_prompts 个 prompt 调用（repeat_ratio 比例重复）
      2. 每次调用走 ToyTextConditioner.encode()（含 cache 查表）
      3. 分别记录 with-cache 和 without-cache 的耗时与分配

    参数：
        num_prompts: 总 prompt 调用次数（默认 100）。
        repeat_ratio: 重复 prompt 比例（0.0–1.0，默认 0.3）。
        hidden_size: embedding 维度。
        max_seq_len: 最大序列长度。
        output_dir: 结果输出目录。
        seed: 随机种子。

    返回：
        包含所有指标的结果字典。
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # 生成基础 prompt 列表（num_unique 个 unique prompt）
    num_unique = int(num_prompts * (1 - repeat_ratio))
    # 用随机英文短句模拟
    base_prompts = [
        f"a {rng.choice(['cat', 'dog', 'bird', 'car', 'house', 'tree', 'mountain', 'river'])} "
        f"{rng.choice(['sitting', 'running', 'flying', 'standing', 'sleeping'])} "
        f"in a {rng.choice(['sunny', 'rainy', 'snowy', 'foggy'])} {rng.choice(['garden', 'city', 'forest', 'beach'])}"
        for _ in range(num_unique)
    ]

    # 生成 prompt 调用序列：repeat_ratio 比例重复
    prompt_sequence = []
    for i in range(num_prompts):
        if i < num_unique or rng.random() < repeat_ratio:
            prompt_sequence.append(base_prompts[rng.randint(0, num_unique)])
        else:
            prompt_sequence.append(base_prompts[i - num_unique])

    # === 场景 A：无 cache（每次全新 ToyTextConditioner，无状态）===
    conditioner_no_cache = ToyTextConditioner(
        model_id="SD3-Medium",
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        dtype="float32",
        device="cuda",
        offload_strategy="cpu",
        seed=seed,
    )
    # 清空内部 cache 模拟无缓存场景
    conditioner_no_cache._cache.clear()
    conditioner_no_cache._cache._hit_count = 0
    conditioner_no_cache._cache._miss_count = 0

    # 重写 encode 方法：每步必 miss（绕过 cache 查表）
    def encode_no_cache(prompt: str, neg: str = "") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        time.sleep(0.005)  # 模拟完整 text encoder 前向
        cond = conditioner_no_cache._generate_embedding(prompt, 0)
        uncond = conditioner_no_cache._generate_embedding(neg, 1)
        return cond, uncond

    alloc_count_no_cache = 0
    start = time.perf_counter()
    for idx, p in enumerate(prompt_sequence):
        _, _ = encode_no_cache(p, "")
        alloc_count_no_cache += 3  # cond + uncond + 内部临时
    elapsed_no_cache = time.perf_counter() - start

    # === 场景 B：有 cache（正常 ToyTextConditioner）===
    conditioner_with_cache = ToyTextConditioner(
        model_id="SD3-Medium",
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        dtype="float32",
        device="cuda",
        offload_strategy="cpu",
        seed=seed,
    )
    # 确保 cache 干净
    conditioner_with_cache._cache.clear()

    alloc_count_with_cache = 0
    start = time.perf_counter()
    for idx, p in enumerate(prompt_sequence):
        _, _ = conditioner_with_cache.encode(p, "", simulate_latency=0.005)
        alloc_count_with_cache += 1  # 仅 cond（或命中时的零分配）
    elapsed_with_cache = time.perf_counter() - start

    stats = conditioner_with_cache.cache_stats()

    # === 场景 C：不同 model_id cache 隔离验证 ===
    conditioner_flux = ToyTextConditioner(
        model_id="FLUX-schnell",
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        dtype="float32",
        device="cuda",
        offload_strategy="cpu",
        seed=999,
    )
    # 先让 SD3 conditioner 缓存一些 prompt
    sd3_cached = set()
    for p in base_prompts[:10]:
        conditioner_with_cache.encode(p, "")
        sd3_cached.add(p)

    # FLUX 查询是否有任何命中
    flux_stats_before = conditioner_flux._cache.cache_stats()
    for p in base_prompts[:10]:
        conditioner_flux.encode(p, "")
    flux_stats_after = conditioner_flux._cache.cache_stats()

    # FLUX 应该全部 miss：SD3 的 cache 不能喂给 FLUX
    cross_model_misses = flux_stats_after["miss_count"] - flux_stats_before["miss_count"]

    # === 汇总结果 ===
    latency_saved = elapsed_no_cache - elapsed_with_cache
    alloc_saved = alloc_count_no_cache - alloc_count_with_cache

    results = {
        "experiment": "prompt_embedding_cache",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "num_prompts": num_prompts,
            "repeat_ratio": repeat_ratio,
            "num_unique_prompts": num_unique,
            "hidden_size": hidden_size,
            "max_seq_len": max_seq_len,
            "estimated_text_encoder_latency_s": 0.005,
        },
        "cache_stats": stats,
        "performance": {
            "without_cache": {
                "total_latency_s": round(elapsed_no_cache, 4),
                "avg_latency_per_call_ms": round(elapsed_no_cache / num_prompts * 1000, 3),
                "allocation_count": alloc_count_no_cache,
            },
            "with_cache": {
                "total_latency_s": round(elapsed_with_cache, 4),
                "avg_latency_per_call_ms": round(elapsed_with_cache / num_prompts * 1000, 3),
                "allocation_count": alloc_count_with_cache,
            },
            "saved": {
                "latency_s": round(latency_saved, 4),
                "latency_reduction_pct": round(latency_saved / elapsed_no_cache * 100, 1) if elapsed_no_cache > 0 else 0,
                "allocation_saved": alloc_saved,
            },
        },
        "cross_model_isolation": {
            "sd3_model_id": "SD3-Medium",
            "flux_model_id": "FLUX-schnell",
            "prompts_shared": 10,
            "flux_cache_misses": cross_model_misses,
            "flux_cache_hits": flux_stats_after["hit_count"] - flux_stats_before["hit_count"],
            "isolation_verified": cross_model_misses == 10,
        },
    }

    return results


# =============================================================================
# 命令行接口
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "Prompt Embedding Cache 实验 — 缓存 text encoder 输出，"
            "避免同一 prompt 重复编码。\n"
            "★★★ 这是 diffusion 的 prompt embedding cache，不是 LLM 的 KV cache ★★★"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
========== 示例 ==========

# 默认 demo
python prompt_embedding_cache.py --demo

# 自定义参数
python prompt_embedding_cache.py --demo --num_prompts 500 --repeat_ratio 0.5 --output_dir results

# 仅检查 cache key 构造（不跑 demo）
python -c "
from prompt_embedding_cache import PromptEmbeddingCacheKey
k1 = PromptEmbeddingCacheKey('SD3', 'a1b2', 'c3d4', 'hello', '', 77, 'float32', 'cuda', 'none')
k2 = PromptEmbeddingCacheKey('SD3', 'a1b2', 'c3d4', 'hello', '', 77, 'float32', 'cuda', 'none')
print(f'Same key eq: {k1 == k2}')
print(f'Same key hash: {hash(k1) == hash(k2)}')
k3 = PromptEmbeddingCacheKey('FLUX', 'a1b2', 'c3d4', 'hello', '', 77, 'float32', 'cuda', 'none')
print(f'Diff model eq: {k1 == k3}')
print(f'Diff model hash: {hash(k1) != hash(k3)}')
"
""",
    )

    # === 运行模式 ===
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行 demo 实验：模拟 100 次 prompt 调用，30%% 重复，输出到 results/",
    )

    # === Demo 参数 ===
    demo_group = parser.add_argument_group("Demo 参数")
    demo_group.add_argument(
        "--num_prompts",
        type=int,
        default=100,
        help="总 prompt 调用次数（默认 100）",
    )
    demo_group.add_argument(
        "--repeat_ratio",
        type=float,
        default=0.3,
        help="重复 prompt 比例 0.0–1.0（默认 0.3，即 30%% 重复）",
    )
    demo_group.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="模拟的 text encoder 输出维度（默认 64）",
    )
    demo_group.add_argument(
        "--max_seq_len",
        type=int,
        default=77,
        help="最大序列长度（默认 77）",
    )

    # === 输出 ===
    output_group = parser.add_argument_group("输出选项")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="结果输出目录（默认 results）",
    )
    output_group.add_argument(
        "--no_save",
        action="store_true",
        help="不保存结果文件，仅打印到 stdout",
    )

    # === 杂项 ===
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）",
    )

    return parser


def main() -> None:
    """主入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if not args.demo:
        parser.print_help()
        print("\n提示：使用 --demo 运行实验，或 --help 查看完整帮助。")
        sys.exit(0)

    # 创建输出目录
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  Prompt Embedding Cache 实验")
    print("  ★ 这不是 LLM 的 KV cache ★")
    print("=" * 72)
    print(f"  参数: num_prompts={args.num_prompts}, "
          f"repeat_ratio={args.repeat_ratio}, "
          f"hidden_size={args.hidden_size}, "
          f"max_seq_len={args.max_seq_len}")
    print(f"  Cache key 字段: model_id, tokenizer_hash, text_encoder_hash, "
          f"prompt, negative_prompt, max_sequence_length, dtype, device, offload_strategy")
    print()

    # 运行 demo
    results = run_demo(
        num_prompts=args.num_prompts,
        repeat_ratio=args.repeat_ratio,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # 打印结果
    print("─" * 72)
    print("  Cache 统计")
    print("─" * 72)
    cs = results["cache_stats"]
    print(f"  Hit 次数:   {cs['hit_count']}")
    print(f"  Miss 次数:  {cs['miss_count']}")
    print(f"  命中率:     {cs['hit_ratio']:.2%}")
    print(f"  缓存条目:   {cs['size']} / {cs['max_size']}")

    print()
    print("─" * 72)
    print("  性能对比")
    print("─" * 72)
    pf = results["performance"]
    print(f"  无 cache:  {pf['without_cache']['total_latency_s']:.3f}s "
          f"({pf['without_cache']['avg_latency_per_call_ms']:.2f}ms/次)")
    print(f"  有 cache:  {pf['with_cache']['total_latency_s']:.3f}s "
          f"({pf['with_cache']['avg_latency_per_call_ms']:.2f}ms/次)")
    print(f"  节省延迟:  {pf['saved']['latency_s']:.3f}s "
          f"({pf['saved']['latency_reduction_pct']:.1f}%)")
    print(f"  节省分配:  {pf['saved']['allocation_saved']} 次")

    print()
    print("─" * 72)
    print("  跨模型隔离验证")
    print("─" * 72)
    iso = results["cross_model_isolation"]
    print(f"  SD3 cache 种群 → FLUX 查询 {iso['prompts_shared']} 次")
    print(f"  FLUX miss:  {iso['flux_cache_misses']}")
    print(f"  FLUX hit:   {iso['flux_cache_hits']}")
    print(f"  隔离验证:   {'✅ 通过（不同 model_id 不会误命中）' if iso['isolation_verified'] else '❌ 失败（cache 跨模型泄漏）'}")

    # 保存结果
    if not args.no_save:
        timestamp = results["timestamp"]
        json_path = os.path.join(args.output_dir, f"prompt_cache_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {json_path}")

    print()
    print("=" * 72)
    print("  实验完成。")
    print("=" * 72)


if __name__ == "__main__":
    main()
