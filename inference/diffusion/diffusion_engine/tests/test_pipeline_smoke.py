"""
test_pipeline_smoke.py — DiffusionPipeline 集成 smoke test（T12）

测试:
- 完整 4 步推理 → 输出 image shape = (1, 3, H, W)
- seed=0 vs seed=1 产生不同输出
- sequential vs batched CFG 数值近似（容差 1e-4）
- PromptEmbeddings cache 第二次调用是 hit
- LatentBufferManager.get/swap/reset

环境要求: torch>=2.7
若 torch 未安装，所有测试自动 skip。
"""

import pytest

# ── 安全导入 ──────────────────────────────────────────────────────────
try:
    import torch
    HAVE_TORCH = True
    TORCH_REASON = ""
except ImportError as e:
    HAVE_TORCH = False
    TORCH_REASON = f"torch 未安装 ({e}) — 测试跳过。请安装 torch>=2.7 后运行。"

try:
    from diffusion_engine.core.dit import TinyDiT
    from diffusion_engine.core.scheduler import RectifiedFlowScheduler
    from diffusion_engine.core.text_conditioning import (
        PromptEmbeddings,
        TextConditioner,
        ToyTextConditioner,
    )
    from diffusion_engine.core.memory_manager import (
        CFGMode,
        LatentBufferManager,
        EmbeddingCache,
        MemoryStats,
    )
    from diffusion_engine.core.vae_stub import ToyVAE
    from diffusion_engine.core.pipeline import DiffusionPipeline
    HAVE_ENGINE = True
except ImportError as e:
    HAVE_ENGINE = False

# 统一 skip 条件
requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason=TORCH_REASON)
requires_engine = pytest.mark.skipif(
    not HAVE_ENGINE,
    reason="diffusion_engine.core T12 模块导入失败（torch 可能未安装）",
)


# ══════════════════════════════════════════════════════════════════════════════
# 测试夹具
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def toy_pipeline():
    """
    构建完整的 toy pipeline 夹具:
    TinyDiT (hidden=64, depth=2) + RectifiedFlowScheduler +
    ToyTextConditioner + ToyVAE
    """
    if not HAVE_TORCH or not HAVE_ENGINE:
        pytest.skip("torch 或 engine 模块不可用")

    dit = TinyDiT(
        in_channels=4,
        patch_size=2,
        hidden_size=64,
        depth=2,
        num_heads=4,
        text_dim=64,
        max_text_len=16,
    )
    scheduler = RectifiedFlowScheduler(num_steps=28)
    conditioner = ToyTextConditioner(
        hidden_size=64, max_seq_len=16, seed=42, device="cpu", dtype=torch.float32
    )
    vae = ToyVAE(latent_channels=4, img_channels=3, base_channels=16)
    return DiffusionPipeline(dit, scheduler, conditioner, vae)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 集成测试
# ══════════════════════════════════════════════════════════════════════════════


class TestPipelineSmoke:
    """DiffusionPipeline 完整 smoke test"""

    @requires_torch
    @requires_engine
    def test_full_4_step_inference_shape(self, toy_pipeline):
        """完整跑 4 步推理，断言输出 image shape = (1, 3, H, W)"""
        images = toy_pipeline.run(
            prompt="a cat",
            num_steps=4,
            cfg_scale=2.0,
            height=64,
            width=64,
            seed=0,
        )
        assert images.shape == (1, 3, 64, 64), f"期望 (1,3,64,64)，实际 {images.shape}"
        assert not torch.isnan(images).any(), "输出包含 NaN"
        assert not torch.isinf(images).any(), "输出包含 inf"

    @requires_torch
    @requires_engine
    def test_different_seeds_different_output(self, toy_pipeline):
        """seed=0 vs seed=1 产生不同输出"""
        img0 = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=0,
        )
        img1 = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=1,
        )
        # 不同 seed 应产生不同结果
        assert not torch.allclose(img0, img1), (
            "seed=0 和 seed=1 应产生不同输出"
        )

    @requires_torch
    @requires_engine
    def test_same_seed_deterministic(self, toy_pipeline):
        """相同 seed 应产生完全相同的输出（确定性）"""
        img_a = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=42,
        )
        img_b = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=42,
        )
        assert torch.allclose(img_a, img_b), "相同 seed 应产生完全相同的输出"

    @requires_torch
    @requires_engine
    def test_sequential_vs_batched_cfg(self, toy_pipeline):
        """sequential vs batched CFG 数值近似（容差 1e-4）"""
        img_seq = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=0,
            mode=CFGMode.SEQUENTIAL,
        )
        img_batch = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=0,
            mode=CFGMode.BATCHED,
        )
        # 两种 CFG 模式应数值近似（相同 seed 和 prompt 生成相同初始噪声和 embedding）
        assert torch.allclose(img_seq, img_batch, atol=1e-4, rtol=1e-4), (
            f"sequential 与 batched CFG 期望近似，最大差异: "
            f"{(img_seq - img_batch).abs().max().item()}"
        )

    @requires_torch
    @requires_engine
    def test_cfg_scale_one(self, toy_pipeline):
        """cfg_scale=1.0（无 CFG）不报错"""
        images = toy_pipeline.run(
            prompt="a cat", num_steps=4, cfg_scale=1.0,
            height=64, width=64, seed=0,
        )
        assert images.shape == (1, 3, 64, 64)
        assert not torch.isnan(images).any()

    @requires_torch
    @requires_engine
    def test_profile_run(self, toy_pipeline):
        """profile_run 返回预期字段"""
        result = toy_pipeline.profile_run(
            prompt="a cat", num_steps=4, cfg_scale=2.0,
            height=64, width=64, seed=0,
        )
        assert "image_shape" in result
        assert "memory_snapshot" in result
        assert "num_steps" in result
        assert "mode" in result
        assert result["image_shape"] == (1, 3, 64, 64)


# ══════════════════════════════════════════════════════════════════════════════
# PromptEmbeddings Cache 测试
# ══════════════════════════════════════════════════════════════════════════════


class TestPromptEmbeddingCache:
    """ToyTextConditioner 的 prompt embedding cache 测试"""

    @requires_torch
    @requires_engine
    def test_second_call_is_hit(self):
        """第二次调用相同 prompt 是 cache hit"""
        conditioner = ToyTextConditioner(
            hidden_size=64, max_seq_len=16, seed=42, device="cpu", dtype=torch.float32
        )

        # 第一次调用 — miss
        emb1 = conditioner.encode(prompt="a cat", negative_prompt="")
        stats1 = conditioner.cache_stats()
        assert stats1["misses"] == 1
        assert stats1["hits"] == 0
        assert stats1["size"] == 1

        # 第二次调用相同 prompt — hit
        emb2 = conditioner.encode(prompt="a cat", negative_prompt="")
        stats2 = conditioner.cache_stats()
        assert stats2["hits"] == 1
        assert stats2["misses"] == 1
        assert stats2["size"] == 1

        # 返回的 embedding 应相同（相同 key 的 cache 引用）
        assert emb1.cond is emb2.cond, (
            "第二次调用相同 prompt 应返回缓存引用"
        )

    @requires_torch
    @requires_engine
    def test_different_prompt_is_miss(self):
        """不同 prompt 是 cache miss"""
        conditioner = ToyTextConditioner(
            hidden_size=64, max_seq_len=16, seed=42, device="cpu", dtype=torch.float32
        )

        emb1 = conditioner.encode(prompt="a cat", negative_prompt="")
        emb2 = conditioner.encode(prompt="a dog", negative_prompt="")

        stats = conditioner.cache_stats()
        assert stats["misses"] == 2
        assert stats["size"] == 2
        assert emb1.cond is not emb2.cond, "不同 prompt 应产生不同缓存条目"

    @requires_torch
    @requires_engine
    def test_cache_key_includes_all_params(self):
        """cache key 包含所有关键参数 — 改变参数应导致 miss"""
        conditioner = ToyTextConditioner(
            hidden_size=64, max_seq_len=16, seed=42, device="cpu", dtype=torch.float32
        )

        # baseline
        emb = conditioner.encode(prompt="a cat", negative_prompt="", max_seq_len=16)
        stats_before = conditioner.cache_stats()

        # 改变 max_seq_len 应导致新 cache entry
        emb2 = conditioner.encode(prompt="a cat", negative_prompt="", max_seq_len=8)
        stats_after = conditioner.cache_stats()

        assert stats_after["size"] == stats_before["size"] + 1, (
            "改变 max_seq_len 应产生新 cache entry"
        )
        assert emb.cond is not emb2.cond, "不同 max_seq_len 的缓存应分开"

    @requires_torch
    @requires_engine
    def test_prompt_embeddings_to_method(self):
        """PromptEmbeddings.to() 方法正确移动设备"""
        if not HAVE_TORCH:
            return
        conditioner = ToyTextConditioner(
            hidden_size=64, max_seq_len=16, seed=42, device="cpu", dtype=torch.float32
        )
        emb = conditioner.encode(prompt="a cat")
        assert emb.device == torch.device("cpu")

        # to() 方法
        emb_cpu = emb.to("cpu")
        assert emb_cpu.device == torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# LatentBufferManager 测试
# ══════════════════════════════════════════════════════════════════════════════


class TestLatentBufferManager:
    """LatentBufferManager: get/swap/reset 接口测试"""

    @requires_torch
    @requires_engine
    def test_get_returns_tensor(self):
        """get 返回正确 shape 的 tensor"""
        shape = (1, 4, 8, 8)
        manager = LatentBufferManager(
            image_shape=shape, device="cpu", dtype=torch.float32, seed=0
        )
        x_t = manager.get("x_t")
        assert x_t.shape == shape, f"期望 {shape}，实际 {x_t.shape}"
        assert x_t.dtype == torch.float32

    @requires_torch
    @requires_engine
    def test_swap_ping_pong(self):
        """swap 实现 ping-pong：交换后引用互调"""
        shape = (1, 4, 8, 8)
        manager = LatentBufferManager(
            image_shape=shape, device="cpu", dtype=torch.float32, seed=0
        )

        # 记录原始数据
        x_t_before = manager.get("x_t").clone()
        x_next_before = manager.get("x_next").clone()

        # swap
        manager.swap("x_t", "x_next")

        # 验证数据交换
        assert torch.allclose(manager.get("x_t"), x_next_before), "swap 后 x_t 应为原 x_next"
        assert torch.allclose(manager.get("x_next"), x_t_before), "swap 后 x_next 应为原 x_t"

    @requires_torch
    @requires_engine
    def test_reset_reinitializes_buffers(self):
        """reset 用新噪声重新初始化"""
        shape = (1, 4, 8, 8)
        manager = LatentBufferManager(
            image_shape=shape, device="cpu", dtype=torch.float32, seed=0
        )

        noise_before = manager.get("noise").clone()

        # reset 重新初始化
        manager.reset()
        noise_after = manager.get("noise")

        # 相同 seed → 相同噪声
        assert torch.allclose(noise_before, noise_after), (
            "相同 seed 的 reset 应产生相同噪声"
        )

    @requires_torch
    @requires_engine
    def test_get_unknown_key_raises(self):
        """获取未知 buffer 应抛 KeyError"""
        shape = (1, 4, 8, 8)
        manager = LatentBufferManager(image_shape=shape, device="cpu", seed=0)
        with pytest.raises(KeyError):
            manager.get("nonexistent")

    @requires_torch
    @requires_engine
    def test_video_shape(self):
        """video_shape 参数应正确支持 5D latent"""
        video_shape = (1, 4, 16, 8, 8)  # (B, C, T, H, W)
        manager = LatentBufferManager(
            image_shape=(1, 4, 8, 8),   # 若 video_shape 为 None 则用 image_shape
            video_shape=video_shape,
            device="cpu",
            seed=0,
        )
        assert manager.get("x_t").shape == video_shape


# ══════════════════════════════════════════════════════════════════════════════
# MemoryStats 测试
# ══════════════════════════════════════════════════════════════════════════════


class TestMemoryStats:
    """MemoryStats: 显存统计接口"""

    @requires_torch
    @requires_engine
    def test_snapshot_has_required_keys(self):
        """snapshot 返回所有必需字段"""
        stats = MemoryStats()
        snap = stats.snapshot()
        required = {"peak_allocated", "peak_reserved", "current_allocated",
                     "allocation_count", "backend"}
        assert required.issubset(snap.keys()), f"缺少字段: {required - set(snap.keys())}"

    @requires_torch
    @requires_engine
    def test_backend_is_valid(self):
        """backend 字段为 'cuda' / 'mps' / 'cpu' 之一"""
        stats = MemoryStats()
        assert stats.snapshot()["backend"] in ("cuda", "mps", "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache 测试
# ══════════════════════════════════════════════════════════════════════════════


class TestEmbeddingCache:
    """EmbeddingCache: prompt embedding 缓存包装"""

    @requires_torch
    @requires_engine
    def test_put_and_get(self):
        """put → get 往返正确"""
        cache = EmbeddingCache()
        cond = torch.randn(1, 16, 64)
        uncond = torch.randn(1, 16, 64)
        emb = PromptEmbeddings(cond=cond, uncond=uncond)

        cache.put("test_key", emb)
        retrieved = cache.get("test_key")

        assert retrieved is not None
        assert retrieved.cond is cond  # 引用相同
        assert retrieved.uncond is uncond

    @requires_torch
    @requires_engine
    def test_contains(self):
        """contains 正确判断缓存存在性"""
        cache = EmbeddingCache()
        assert not cache.contains("missing")

        emb = PromptEmbeddings(cond=torch.randn(1, 16, 64))
        cache.put("exists", emb)
        assert cache.contains("exists")

    @requires_torch
    @requires_engine
    def test_clear(self):
        """clear 清空所有缓存"""
        cache = EmbeddingCache()
        emb = PromptEmbeddings(cond=torch.randn(1, 16, 64))
        cache.put("key1", emb)
        cache.put("key2", emb)
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("key1") is None


# ══════════════════════════════════════════════════════════════════════════════
# ToyVAE 测试
# ══════════════════════════════════════════════════════════════════════════════


class TestToyVAE:
    """ToyVAE: 编解码 shape 测试"""

    @requires_torch
    @requires_engine
    def test_encode_shape(self):
        """encode: (1, 3, H, W) → (1, 4, H/8, W/8)"""
        vae = ToyVAE()
        x = torch.randn(1, 3, 64, 64)
        z = vae.encode(x)
        assert z.shape == (1, 4, 8, 8), f"期望 (1,4,8,8)，实际 {z.shape}"

    @requires_torch
    @requires_engine
    def test_decode_shape(self):
        """decode: (1, 4, H/8, W/8) → (1, 3, H, W)"""
        vae = ToyVAE()
        z = torch.randn(1, 4, 8, 8)
        x = vae.decode(z)
        assert x.shape == (1, 3, 64, 64), f"期望 (1,3,64,64)，实际 {x.shape}"

    @requires_torch
    @requires_engine
    def test_roundtrip_shape(self):
        """encode → decode 往返 shape 匹配"""
        vae = ToyVAE()
        x = torch.randn(1, 3, 128, 64)
        z = vae.encode(x)
        x_recon = vae.decode(z)
        assert x_recon.shape == x.shape

    @requires_torch
    @requires_engine
    def test_latent_scale_factor(self):
        """latent_scale_factor 应为 0.18215"""
        vae = ToyVAE()
        assert abs(vae.latent_scale_factor - 0.18215) < 1e-6
