"""
多模态 KV Cache 管理实验 - 共享 cache 模拟器。

本模块提供:
1. 三种 cache key 策略 (A/B/C) 的 hash 函数;
2. MultimodalRequest 数据类;
3. CacheSimulator 模拟器, 支持 hit/false-hit/miss 判定与统计。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 真实图片读取 (用于 hash 比对)
# ---------------------------------------------------------------------------
_SAMPLE_IMG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "vlm_minimal_demo", "sample_images", "demo.jpg",
)


def _read_sample_bytes() -> bytes:
    """读取 demo.jpg 的真实字节, 若不存在则报错。"""
    p = os.path.abspath(_SAMPLE_IMG)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"参考图片不存在: {p}")
    with open(p, "rb") as f:
        return f.read()


def _make_synthetic_bytes(seed: int, length: int = 1024) -> bytes:
    """用确定性 seed 生成合成字节 (模拟不同图片)。"""
    import random
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(length))


# ---------------------------------------------------------------------------
# 请求数据结构
# ---------------------------------------------------------------------------
@dataclass
class MultimodalRequest:
    """模拟一次多模态推理请求。"""
    text_token_ids: list[int]
    image_bytes_list: list[bytes]           # 每张图的原始字节
    model_id: str = "qwen3-vl-4b"
    tokenizer_config_hash: str = "tok_v1"
    processor_config_hash: str = "proc_v1"
    original_sizes: list[tuple[int, int]] = field(default_factory=list)
    resized_sizes: list[tuple[int, int]] = field(default_factory=list)
    patch_grids: list[tuple[int, int]] = field(default_factory=list)
    num_visual_tokens_list: list[int] = field(default_factory=list)
    placeholder_layout: str = "default"
    multi_image_order: str = "image_0,image_1"
    video_frame_sampling_meta: str = "none"

    def __post_init__(self) -> None:
        n = len(self.image_bytes_list)
        # 自动填充缺失的列表元数据
        if not self.original_sizes:
            self.original_sizes = [(512, 512)] * n
        if not self.resized_sizes:
            self.resized_sizes = [(336, 336)] * n
        if not self.patch_grids:
            self.patch_grids = [(12, 12)] * n
        if not self.num_visual_tokens_list:
            self.num_visual_tokens_list = [256] * n
        # 自动推断 multi_image_order
        if self.multi_image_order == "default" and n >= 1:
            self.multi_image_order = ",".join(f"image_{i}" for i in range(n))

    @property
    def image_hashes(self) -> list[str]:
        """返回每张图片的 SHA-256 hex 摘要。"""
        return [hashlib.sha256(b).hexdigest() for b in self.image_bytes_list]

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.text_token_ids, sort_keys=True).encode()
        ).hexdigest()


# ---------------------------------------------------------------------------
# 三种 cache key 策略
# ---------------------------------------------------------------------------
def strategy_a_text_only(text_token_ids: list[int]) -> str:
    """策略 A: 仅 hash 文本 token IDs。"""
    payload = json.dumps(text_token_ids, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strategy_b_text_plus_image_hash(
    text_token_ids: list[int], image_bytes_list: list[bytes]
) -> str:
    """策略 B: hash 文本 token IDs + 每张图片的 SHA-256。"""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(text_token_ids, sort_keys=True).encode("utf-8"))
    for img_bytes in image_bytes_list:
        hasher.update(hashlib.sha256(img_bytes).digest())
    return hasher.hexdigest()


def strategy_c_full_multimodal(req: MultimodalRequest) -> str:
    """策略 C: 全量多模态 hash (model/config/text/images/layout/order...)。"""
    hasher = hashlib.sha256()
    # 模型与配置
    hasher.update(req.model_id.encode("utf-8"))
    hasher.update(req.tokenizer_config_hash.encode("utf-8"))
    hasher.update(req.processor_config_hash.encode("utf-8"))
    # 文本
    hasher.update(json.dumps(req.text_token_ids, sort_keys=True).encode("utf-8"))
    # 图片字节 (先 hash 再纳入, 与 B 一致)
    for img_bytes in req.image_bytes_list:
        hasher.update(hashlib.sha256(img_bytes).digest())
    # 图像处理元数据
    for key in ("original_sizes", "resized_sizes", "patch_grids",
                "num_visual_tokens_list"):
        hasher.update(
            json.dumps(getattr(req, key), sort_keys=True).encode("utf-8")
        )
    # 布局与顺序
    hasher.update(req.placeholder_layout.encode("utf-8"))
    hasher.update(req.multi_image_order.encode("utf-8"))
    hasher.update(req.video_frame_sampling_meta.encode("utf-8"))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Cache 条目
# ---------------------------------------------------------------------------
@dataclass
class CacheEntry:
    key: str
    strategy: str          # "A" | "B" | "C"
    request: MultimodalRequest
    prefill_tokens: int    # 估算的 prefill token 数
    kv_blocks_used: int    # 估算的 KV block 数
    memory_bytes: int      # 估算的显存占用


# ---------------------------------------------------------------------------
# Cache 模拟器
# ---------------------------------------------------------------------------
class CacheSimulator:
    """按某一种策略维护的 cache store, 支持 hit/false-hit/miss 判定。"""

    def __init__(self, strategy: str, block_size: int = 16,
                 bytes_per_token: int = 256):
        if strategy not in ("A", "B", "C"):
            raise ValueError("strategy must be A/B/C")
        self.strategy = strategy
        self.block_size = block_size
        self.bytes_per_token = bytes_per_token   # 每个 token 的 KV 字节估算
        self._store: dict[str, CacheEntry] = {}
        # 统计
        self.hits = 0
        self.false_hits = 0
        self.safe_misses = 0
        self.prefill_tokens_saved = 0
        self.kv_blocks_reused = 0
        self.memory_saved = 0

    # ---- key 计算 ----
    def _compute_key(self, req: MultimodalRequest) -> str:
        if self.strategy == "A":
            return strategy_a_text_only(req.text_token_ids)
        elif self.strategy == "B":
            return strategy_b_text_plus_image_hash(
                req.text_token_ids, req.image_bytes_list
            )
        else:  # C
            return strategy_c_full_multimodal(req)

    # ---- 语义级 image hash (用于 false-hit 检测) ----
    @staticmethod
    def _image_fingerprint(req: MultimodalRequest) -> str:
        """返回请求的语义级图像指纹 (所有图片 hash 的有序拼接)。"""
        parts = [hashlib.sha256(b).hexdigest() for b in req.image_bytes_list]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()

    # ---- 插入 ----
    def insert(self, req: MultimodalRequest,
               prefill_tokens: int | None = None,
               kv_blocks_used: int | None = None) -> str:
        """将请求插入 cache, 返回 key。"""
        key = self._compute_key(req)
        pt = prefill_tokens if prefill_tokens is not None else len(req.text_token_ids)
        kv = kv_blocks_used if kv_blocks_used is not None else max(1, pt // self.block_size)
        mem = pt * self.bytes_per_token
        self._store[key] = CacheEntry(
            key=key, strategy=self.strategy, request=req,
            prefill_tokens=pt, kv_blocks_used=kv, memory_bytes=mem,
        )
        return key

    # ---- 查询 (核心) ----
    def query(self, req: MultimodalRequest) -> dict[str, Any]:
        """
        查询 cache。返回 hit 判定:
          - "true_hit":  key 匹配且语义 image 匹配
          - "false_hit": key 匹配但语义 image 不匹配 (仅策略 A)
          - "safe_miss": key 不匹配

        同时更新统计计数器。
        """
        key = self._compute_key(req)
        entry = self._store.get(key)
        req_fp = self._image_fingerprint(req)

        if entry is None:
            self.safe_misses += 1
            return {"verdict": "safe_miss", "key": key, "cached_key": None}

        cached_fp = self._image_fingerprint(entry.request)
        if cached_fp == req_fp:
            self.hits += 1
            self.prefill_tokens_saved += entry.prefill_tokens
            self.kv_blocks_reused += entry.kv_blocks_used
            self.memory_saved += entry.memory_bytes
            return {
                "verdict": "true_hit",
                "key": key,
                "cached_key": entry.key,
                "prefill_tokens": entry.prefill_tokens,
                "kv_blocks": entry.kv_blocks_used,
                "memory_bytes": entry.memory_bytes,
            }
        else:
            self.false_hits += 1
            return {"verdict": "false_hit", "key": key, "cached_key": entry.key}

    # ---- 统计 ----
    def stats(self) -> dict[str, Any]:
        total = self.hits + self.false_hits + self.safe_misses
        return {
            "strategy": self.strategy,
            "cache_size": len(self._store),
            "total_queries": total,
            "true_hits": self.hits,
            "false_hits": self.false_hits,
            "safe_misses": self.safe_misses,
            "cache_hit_rate": round(self.hits / total, 4) if total else 0.0,
            "false_hit_rate": round(self.false_hits / total, 4) if total else 0.0,
            "prefill_tokens_saved": self.prefill_tokens_saved,
            "kv_blocks_reused": self.kv_blocks_reused,
            "memory_saved_bytes": self.memory_saved,
        }
