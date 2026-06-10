"""
text_conditioning.py — 文本条件编码与 Prompt Embedding Cache

提供：
- PromptEmbeddings: @dataclass 持有 cond/uncond embeddings
- TextConditioner: Protocol 接口定义
- ToyTextConditioner: toy 实现（随机 embedding，不调真 text encoder）
- HFCachedTextConditioner: 接口预留（T13 用 diffusers 实现）

关键说明：
  本模块的 cache 不是 LLM KV cache。
  - LLM KV cache：存储每层 attention 的 key/value 历史，跨 token 步复用
  - prompt embedding cache：缓存 text encoder 的输出（pooled + seq embedding），
    同一 prompt 在推理循环中多次使用（如 cond + uncond encode）时避免重复编码
  这是 diffusion 特有的优化点：text encoder 只跑一次，embedding 可复用多次。

CFG（Classifier-Free Guidance）相关：
  - CFG 需要 cond 和 uncond 两个 embedding
  - 典型 text encoder（如 CLIP-L 或 T5）可能占用 1–3 GB 显存
  - 缓存机制将"每步 encode"降为"首次 encode + 后续查表"

未使用 minivLLM 任何代码。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PromptEmbeddings:
    """
    持有条件文本（cond）和无条件文本（uncond）的 embedding tensor。

    注意：dtype 和 device 显式记录，用于 cache key 生成和显存预算。
    本结构不包含 LLM 的 KV cache——那是 attention 层的 key/value 历史缓存，
    而这里只是 text encoder 的输出 embedding。
    """

    cond: Optional[torch.Tensor] = None
    """(B, max_seq_len, hidden_size) 条件 text embedding（如 prompt 的编码）。"""

    uncond: Optional[torch.Tensor] = None
    """(B, max_seq_len, hidden_size) 无条件 text embedding（如空字符串的编码）。"""

    @property
    def device(self) -> Optional[torch.device]:
        """获取 embedding 所在设备（从 cond 推导）。"""
        if self.cond is not None:
            return self.cond.device
        if self.uncond is not None:
            return self.uncond.device
        return None

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """获取 embedding 数据类型（从 cond 推导）。"""
        if self.cond is not None:
            return self.cond.dtype
        if self.uncond is not None:
            return self.uncond.dtype
        return None

    def to(self, *args, **kwargs) -> "PromptEmbeddings":
        """将 cond 和 uncond 同时移动到指定设备/数据类型。"""
        cond = self.cond.to(*args, **kwargs) if self.cond is not None else None
        uncond = self.uncond.to(*args, **kwargs) if self.uncond is not None else None
        return PromptEmbeddings(cond=cond, uncond=uncond)


# ──────────────────────────────────────────────────────────────────────────────
# 接口
# ──────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class TextConditioner(Protocol):
    """
    文本条件编码器接口（Protocol，非 ABC）。

    所有实现（toy / HF / 自定义）需支持：
        encode(prompt, negative_prompt, max_seq_len) -> PromptEmbeddings

    注意：
    - 不要求继承特定基类
    - 返回的 PromptEmbeddings 应包含 cond 和 uncond 的 torch tensor
    - 典型实现会包含 prompt embedding cache
    """

    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        max_seq_len: Optional[int] = None,
    ) -> PromptEmbeddings:
        """
        将文本 prompt 编码为 embedding。

        参数：
            prompt: 正向文本提示（如 "a cat sitting on a chair"）。
            negative_prompt: 负向文本提示（通常为空字符串 ""）。
            max_seq_len: 最大 token 序列长度（None 时使用编码器默认值）。

        返回：
            PromptEmbeddings 包含 cond 和 uncond tensor。
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Toy 实现
# ──────────────────────────────────────────────────────────────────────────────


class ToyTextConditioner:
    """
    Toy 文本编码器：不调真实 text encoder，直接生成随机 embedding。

    用途：与 TinyDiT 联调 pipeline，验证 shape 和 cache 逻辑。
    不做任何真实 NLP 编码——seed 受控的随机 tensor 来模拟 text encoder 输出。

    Cache 设计：
        - 内部维护 dict[str, PromptEmbeddings]
        - cache key 包含：prompt + negative_prompt + max_seq_len + hidden_size + dtype + device
        - 首次 encode 是 miss（生成随机 embedding），后续相同 key 是 hit（直接返回）
        - expose cache_stats() 返回 hit/miss 计数
    """

    def __init__(
        self,
        hidden_size: int = 64,
        max_seq_len: int = 16,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        参数：
            hidden_size: 输出 embedding 维度（需与 TinyDiT.text_dim 一致，默认 64）。
            max_seq_len: 最大序列长度（默认 16，与 TinyDiT.max_text_len 一致）。
            seed: 基础随机种子（确定性 embedding 生成）。
            device: 设备。
            dtype: 数据类型。
        """
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.device = device
        self.dtype = dtype

        # == Prompt embedding cache ==
        # 注意：这不是 LLM KV cache。这里是 text encoder 输出缓存。
        self._cache: dict[str, PromptEmbeddings] = {}
        self._hits: int = 0
        self._misses: int = 0

    def _cache_key(
        self,
        prompt: str,
        negative_prompt: str,
        max_seq_len: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: str,
    ) -> str:
        """
        生成 cache key。

        key 包含 6 个字段：prompt、negative_prompt、max_seq_len、hidden_size、dtype、device。
        使用 SHA256 hash 避免 key 过长，同时保证确定性。
        """
        raw = f"{prompt}|{negative_prompt}|{max_seq_len}|{hidden_size}|{dtype}|{device}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _generate_embedding(
        self, prompt: str, max_seq_len: int
    ) -> torch.Tensor:
        """
        生成确定性随机 embedding。

        使用 seed + hash(prompt) 作为子种子，确保：
        - 相同 prompt → 相同 embedding（可复现）
        - 不同 prompt → 不同 embedding（模拟真实 text encoder 行为）

        参数：
            prompt: 文本提示。
            max_seq_len: 序列长度。

        返回：
            (1, max_seq_len, hidden_size) 的随机 tensor。
        """
        # 子种子：base seed + hash(prompt)
        prompt_hash = abs(hash(prompt)) % (2**31)
        sub_seed = self.seed + prompt_hash
        generator = torch.Generator(device=self.device).manual_seed(sub_seed)

        return torch.randn(
            1,
            max_seq_len,
            self.hidden_size,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        max_seq_len: Optional[int] = None,
    ) -> PromptEmbeddings:
        """
        编码 prompt → PromptEmbeddings（含 cache 查表）。

        参数：
            prompt: 正向文本提示。
            negative_prompt: 负向文本提示（默认空字符串）。
            max_seq_len: 最大序列长度（None 时用初始化默认值）。

        返回：
            PromptEmbeddings（cond + uncond）。
        """
        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        # 生成 cache key
        key = self._cache_key(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_seq_len=max_seq_len,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )

        # 检查 cache
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        # Cache miss — 生成随机 embedding
        self._misses += 1

        cond_emb = self._generate_embedding(prompt, max_seq_len)
        uncond_emb = (
            self._generate_embedding(negative_prompt, max_seq_len)
            if negative_prompt is not None
            else None
        )

        embeddings = PromptEmbeddings(cond=cond_emb, uncond=uncond_emb)
        self._cache[key] = embeddings
        return embeddings

    def cache_stats(self) -> dict:
        """
        返回 prompt embedding cache 统计。

        返回：
            {"hits": int, "misses": int, "size": int} — cache 命中/未命中/当前条目数
        """
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }


# ──────────────────────────────────────────────────────────────────────────────
# HF 接口预留
# ──────────────────────────────────────────────────────────────────────────────


class HFCachedTextConditioner:
    """
    HuggingFace Text Encoder 包装器（接口预留 — T13 实现）。

    T13 将使用 diffusers 的 text encoder（CLIP-L / CLIP-G / T5 / Gemma），
    本类提供完整接口定义和注释，但不实现具体逻辑。

    设计要点：
    - encode(prompt, negative_prompt, max_seq_len) → PromptEmbeddings
    - 内部维护 prompt embedding cache（与 ToyTextConditioner 同样接口）
    - 负责 tokenize → 调用 text encoder → 缓存结果
    - 注意：diffusion 的 text encoder 可能输出两种 embedding：
      - pooled_output：(B, D_pool)  — 全局条件注入
      - last_hidden_state：(B, L, D_seq) — 序列条件注入（cross-attention）
    - 本类在 PromptEmbeddings.cond 中需要同时存储两者或分开字段

    不使用 minivLLM 任何代码。
    不依赖 diffusers 在 T12 阶段导入（仅在 T13 中实例化）。
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        subfolder: str = "text_encoder",
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        """
        参数：
            model_id: HuggingFace 模型 ID（如 stabilityai/stable-diffusion-3.5-medium）。
            subfolder: 子文件夹名（text_encoder 或 text_encoder_2）。
            device: 推理设备。
            dtype: 推理数据类型。
        """
        self.model_id = model_id
        self.subfolder = subfolder
        self.device = device
        self.dtype = dtype

        # == Prompt embedding cache ==
        # 同 ToyTextConditioner 接口，key 设计一致
        self._cache: dict[str, PromptEmbeddings] = {}
        self._hits: int = 0
        self._misses: int = 0

        # 实际 text encoder 和 tokenizer 在 T13 中加载
        self._text_encoder = None  # T13: 加载 diffusers CLIP/T5
        self._tokenizer = None     # T13: 加载对应 tokenizer

    def encode(
        self,
        prompt: str,
        negative_prompt: str = "",
        max_seq_len: Optional[int] = None,
    ) -> PromptEmbeddings:
        """
        编码 prompt → PromptEmbeddings（T13 实现）。

        T13 实现步骤：
        1. _cache_key(prompt, negative_prompt, max_seq_len, ...)
        2. cache 命中 → 直接返回
        3. cache 未命中 → tokenize → text_encoder.forward → 缓存 → 返回

        当前阶段：抛出 NotImplementedError。
        """
        raise NotImplementedError(
            "HFCachedTextConditioner 在 T13 中实现。"
            "当前阶段请使用 ToyTextConditioner 进行 pipeline 联调。"
        )

    def cache_stats(self) -> dict:
        """返回 cache 统计（T13 实现后可用）。"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
        }
