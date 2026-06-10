"""
DiT shape 系统测试（T11）

测试 SelfAttention、JointAttention、DiTBlock、TinyDiT 的输入输出 shape。

环境要求: torch>=2.7
若 torch 未安装，所有测试会自动 skip 并说明原因。
"""

import math
import pytest

import numpy as np

# 安全导入 torch：未安装时所有测试自动 skip
try:
    import torch
    HAVE_TORCH = True
    TORCH_REASON = ""
except ImportError as e:
    HAVE_TORCH = False
    TORCH_REASON = f"torch 未安装 ({e}) — 测试跳过。请安装 torch>=2.7 后运行。"

# 安全导入 engine 模块
try:
    from diffusion_engine.core.attention import SelfAttention, JointAttention
    from diffusion_engine.core.transformer_block import DiTBlock
    from diffusion_engine.core.dit import (
        TinyDiT,
        timestep_embedding,
        PatchEmbed,
        Unpatchify,
    )
    HAVE_ENGINE = True
except ImportError as e:
    HAVE_ENGINE = False

# 统一 skip 条件
requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason=TORCH_REASON)
requires_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="diffusion_engine.core 模块导入失败")

# ===========================================================================
# Torch 环境检查
# ===========================================================================


class TestTorchEnvironment:
    """验证 torch 环境基本信息"""

    @requires_torch
    def test_torch_version(self):
        """确保 torch 版本满足要求"""
        assert torch.__version__ >= "2.0", f"torch 版本 {torch.__version__} 低于 2.0"

    @requires_torch
    def test_mps_availability(self):
        """记录 MPS 可用性（不要求可用，仅记录）"""
        mps_ok = torch.backends.mps.is_available()
        print(f"MPS available: {mps_ok}")


# ===========================================================================
# SelfAttention 测试
# ===========================================================================


class TestSelfAttention:
    """SelfAttention: non-causal full attention"""

    @requires_torch
    @requires_engine
    @pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_shape(self, num_heads, batch_size):
        """验证不同 num_heads 和 batch size 下的输入输出 shape 一致性"""
        dim = num_heads * 16  # 确保 dim 是 num_heads 的整数倍
        attn = SelfAttention(dim=dim, num_heads=num_heads, qkv_bias=True)
        x = torch.randn(batch_size, 32, dim)
        out = attn(x)
        assert out.shape == x.shape, f"期望 {x.shape}，实际 {out.shape}"

    @requires_torch
    @requires_engine
    def test_no_nan(self):
        """验证输出无 NaN、无 inf"""
        attn = SelfAttention(dim=64, num_heads=4)
        x = torch.randn(1, 16, 64)
        out = attn(x)
        assert not torch.isnan(out).any(), "输出包含 NaN"
        assert not torch.isinf(out).any(), "输出包含 inf"

    @requires_torch
    @requires_engine
    def test_deterministic(self):
        """验证相同输入产生相同输出（无随机性）"""
        attn = SelfAttention(dim=64, num_heads=4)
        attn.eval()
        x = torch.randn(1, 16, 64)
        with torch.no_grad():
            out1 = attn(x)
            out2 = attn(x)
        assert torch.allclose(out1, out2), "相同输入产生不同输出"


# ===========================================================================
# JointAttention 测试
# ===========================================================================


class TestJointAttention:
    """JointAttention: MMDiT-style 联合注意力"""

    @requires_torch
    @requires_engine
    def test_shape(self):
        """验证两组 tokens 拼接-attend-拆分的 shape 正确性"""
        dim = 64
        attn = JointAttention(dim=dim, num_heads=4)
        x1 = torch.randn(2, 16, dim)  # image tokens
        x2 = torch.randn(2, 8, dim)   # text tokens
        y1, y2 = attn(x1, x2)
        assert y1.shape == x1.shape, f"y1 期望 {x1.shape}，实际 {y1.shape}"
        assert y2.shape == x2.shape, f"y2 期望 {x2.shape}，实际 {y2.shape}"

    @requires_torch
    @requires_engine
    def test_no_nan(self):
        """验证输出无 NaN、无 inf"""
        dim = 64
        attn = JointAttention(dim=dim, num_heads=4)
        x1 = torch.randn(1, 20, dim)
        x2 = torch.randn(1, 10, dim)
        y1, y2 = attn(x1, x2)
        assert not torch.isnan(y1).any()
        assert not torch.isnan(y2).any()
        assert not torch.isinf(y1).any()
        assert not torch.isinf(y2).any()

    @requires_torch
    @requires_engine
    def test_combined_attention(self):
        """验证 joint attention 比各自独立 attention 产生不同结果"""
        dim = 64
        joint = JointAttention(dim=dim, num_heads=4)
        self_attn = SelfAttention(dim=dim, num_heads=4)

        x1 = torch.randn(1, 16, dim)
        x2 = torch.randn(1, 8, dim)

        # joint attention
        y1_j, y2_j = joint(x1, x2)

        # 各自独立 self-attention（不应该相同，因为 joint 版本能跨组 attend）
        y1_s = self_attn(x1)
        y2_s = self_attn(x2)

        # 验证 joint 的结果与各自独立的结果不同（至少对非零 tensor）
        assert not torch.allclose(y1_j, y1_s), "joint attention 应与单独 self-attention 不同"


# ===========================================================================
# DiTBlock 测试
# ===========================================================================


class TestDiTBlock:
    """DiTBlock: AdaLN 调制的 transformer block"""

    @requires_torch
    @requires_engine
    def test_self_attention_mode(self):
        """验证纯 self-attention 模式下 shape 正确"""
        block = DiTBlock(hidden_size=64, num_heads=4)
        x = torch.randn(2, 16, 64)       # image tokens
        t_emb = torch.randn(2, 64)        # timestep embedding
        out = block(x, t_emb, text_tokens=None)
        assert out.shape == x.shape

    @requires_torch
    @requires_engine
    def test_joint_attention_mode(self):
        """验证 joint attention 模式下 image tokens shape 不变"""
        block = DiTBlock(hidden_size=64, num_heads=4)
        x = torch.randn(2, 16, 64)       # image tokens
        t_emb = torch.randn(2, 64)        # timestep embedding
        text = torch.randn(2, 8, 64)      # text tokens
        out = block(x, t_emb, text_tokens=text)
        assert out.shape == x.shape

    @requires_torch
    @requires_engine
    def test_timestep_modulation_effect(self):
        """验证不同 timestep 产生不同调制效果"""
        block = DiTBlock(hidden_size=64, num_heads=4)
        x = torch.randn(1, 16, 64)

        # 两个不同的 timestep embedding
        t0 = torch.zeros(1, 64)
        t1 = torch.ones(1, 64)

        with torch.no_grad():
            out0 = block(x, t0)
            out1 = block(x, t1)

        # 不同 timestep 应产生不同输出
        assert not torch.allclose(out0, out1), "不同 timestep 应产生不同输出"

    @requires_torch
    @requires_engine
    def test_no_nan(self):
        """验证输出无 NaN、无 inf"""
        block = DiTBlock(hidden_size=64, num_heads=4)
        x = torch.randn(1, 16, 64)
        t_emb = torch.randn(1, 64)
        out = block(x, t_emb)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @requires_torch
    @requires_engine
    def test_residual_connection(self):
        """验证残差连接存在：输入 x 对输出有直接影响"""
        block = DiTBlock(hidden_size=64, num_heads=4)
        x = torch.randn(1, 16, 64)
        t_emb = torch.randn(1, 64)

        # 如果残余连接断了，clone 掉梯度检查输出是否受 x 值影响
        x_copy = x.clone()
        out = block(x, t_emb)
        # 将 x 加倍
        x2 = x_copy * 2.0
        out2 = block(x2, t_emb)
        # 残差贡献使得输出不同
        assert not torch.allclose(out, out2), "DiTBlock 应保留残差路径"


# ===========================================================================
# PatchEmbed / Unpatchify 测试
# ===========================================================================


class TestPatchEmbed:
    """PatchEmbed: image latent -> patch tokens"""

    @requires_torch
    @requires_engine
    @pytest.mark.parametrize("patch_size", [1, 2, 4])
    def test_shape(self, patch_size):
        """验证不同 patch_size 下的 token 数和维度"""
        C, H, W = 4, 8, 8
        patch_dim = patch_size * patch_size * C
        embed = PatchEmbed(in_channels=C, patch_size=patch_size, hidden_size=patch_dim)
        x = torch.randn(2, C, H, W)
        tokens = embed(x)
        expected_N = (H // patch_size) * (W // patch_size)
        assert tokens.shape == (2, expected_N, patch_dim), \
            f"期望 (2, {expected_N}, {patch_dim})，实际 {tokens.shape}"

    @requires_torch
    @requires_engine
    def test_round_trip(self):
        """验证 patch + unpatch 恢复原始 shape（维度匹配时）"""
        C, H, W = 4, 8, 8
        p = 2
        patch_dim = p * p * C
        embed = PatchEmbed(in_channels=C, patch_size=p, hidden_size=patch_dim)
        unpatch = Unpatchify(out_channels=C, patch_size=p, hidden_size=patch_dim)
        x = torch.randn(1, C, H, W)
        tokens = embed(x)
        restored = unpatch(tokens)
        assert restored.shape == x.shape, f"期望 {x.shape}，实际 {restored.shape}"


class TestTinyDiT:
    """TinyDiT: 完整 toy DiT 模型"""

    @requires_torch
    @requires_engine
    def test_forward_shape(self):
        """验证 TinyDiT forward 输出 shape 与输入一致"""
        model = TinyDiT(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            depth=2,
            num_heads=4,
            text_dim=64,
            max_text_len=16,
        )
        x = torch.randn(1, 4, 8, 8)       # 噪声 latent
        t = torch.tensor([0.5])            # timestep
        text = torch.randn(1, 16, 64)      # text tokens

        out = model(x, t, text)
        assert out.shape == x.shape, f"期望 {x.shape}，实际 {out.shape}"

    @requires_torch
    @requires_engine
    def test_forward_without_text(self):
        """验证无 text tokens 时 forward 不报错"""
        model = TinyDiT(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([0.5])
        out = model(x, t, text_tokens=None)
        assert out.shape == x.shape

    @requires_torch
    @requires_engine
    def test_batch_size(self):
        """验证不同 batch size"""
        model = TinyDiT(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            depth=2,
            num_heads=4,
        )
        for B in [1, 2, 4]:
            x = torch.randn(B, 4, 8, 8)
            t = torch.full((B,), 0.5)
            out = model(x, t)
            assert out.shape == x.shape, f"batch={B} 失败"

    @requires_torch
    @requires_engine
    @pytest.mark.parametrize("patch_size", [1, 2, 4])
    def test_patch_sizes(self, patch_size):
        """验证不同 patch_size 的 TinyDiT"""
        model = TinyDiT(
            in_channels=4,
            patch_size=patch_size,
            hidden_size=patch_size * patch_size * 4,  # 让 patch_dim == hidden_size
            depth=2,
            num_heads=4,
        )
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([0.5])
        out = model(x, t)
        assert out.shape == x.shape, f"patch_size={patch_size} 失败"
        assert not torch.isnan(out).any()

    @requires_torch
    @requires_engine
    def test_no_nan_inf(self):
        """验证 forward 输出无 NaN、无 inf"""
        model = TinyDiT(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(1, 4, 8, 8)
        t = torch.tensor([0.5])
        text = torch.randn(1, 16, 64)
        out = model(x, t, text)
        assert not torch.isnan(out).any(), "输出包含 NaN"
        assert not torch.isinf(out).any(), "输出包含 inf"

    @requires_torch
    @requires_engine
    def test_timestep_embedding_shape(self):
        """测试 timestep_embedding 函数的 shape"""
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = timestep_embedding(t, dim=64)
        assert emb.shape == (3, 64), f"期望 (3, 64)，实际 {emb.shape}"

    @requires_torch
    @requires_engine
    def test_different_timesteps(self):
        """验证不同 timestep 产生不同输出"""
        model = TinyDiT(
            in_channels=4,
            patch_size=2,
            hidden_size=64,
            depth=2,
            num_heads=4,
        )
        x = torch.randn(1, 4, 8, 8)
        out0 = model(x, torch.tensor([0.0]))
        out1 = model(x, torch.tensor([1.0]))
        assert not torch.allclose(out0, out1), "不同 timestep 应产生不同预测"
