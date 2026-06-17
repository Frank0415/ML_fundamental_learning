"""
test_diffusion_gemma.py — DiffusionGemmaPipeline 与 EntropyBoundedSampler 的单元测试
"""

import pytest

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    from diffusion_engine.core.diffusion_gemma_pipeline import (
        EntropyBoundedSampler,
        DiffusionGemmaPipeline
    )
    HAVE_ENGINE = True
except ImportError:
    HAVE_ENGINE = False

requires_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch 未安装")
requires_engine = pytest.mark.skipif(not HAVE_ENGINE, reason="engine 模块导入失败")


class MockDenoiser(torch.nn.Module):
    """一个模拟去噪器，用于测试推理管线"""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x_embeddings: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        # x_embeddings: (1, seq_len, hidden_size)
        # kv_cache: (1, prompt_len, hidden_size)
        # 返回 logits: (1, seq_len, vocab_size)
        seq_len = x_embeddings.shape[1]
        
        # 产生确定性的 logits 用于测试
        # 让前面一些位置的第一个 token 具有极高的置信度，以便触发早停
        logits = torch.zeros(1, seq_len, self.vocab_size, device=x_embeddings.device, dtype=x_embeddings.dtype)
        # 前 20 个位置非常确定
        logits[:, :20, 0] = 50.0
        # 后面的位置有些噪音，但是也有一定的置信度
        logits[:, 20:, :] = torch.randn(1, seq_len - 20, self.vocab_size, device=x_embeddings.device, dtype=x_embeddings.dtype)
        return logits


class TestEntropyBoundedSampler:
    """熵界采样器测试"""

    @requires_torch
    @requires_engine
    def test_sampler_basic(self):
        vocab_size = 10
        sampler = EntropyBoundedSampler(vocab_size=vocab_size, entropy_bound=0.2, renoise_seed=42)

        # 构造概率：前几个位置概率非常确定，后几个位置比较均匀
        probs = torch.zeros(1, 5, vocab_size)
        # 位置 0: 100% 预测第 3 个 token
        probs[0, 0, 3] = 1.0
        # 位置 1: 95% 预测第 5 个 token, 5% 其他
        probs[0, 1, 5] = 0.95
        probs[0, 1, 0] = 0.05
        # 位置 2: 70% 预测第 1 个 token, 30% 第 2 个
        probs[0, 2, 1] = 0.70
        probs[0, 2, 2] = 0.30
        # 位置 3 & 4: 均匀分布 (最不确定)
        probs[0, 3, :] = 1.0 / vocab_size
        probs[0, 4, :] = 1.0 / vocab_size

        current_canvas = torch.zeros(1, 5, dtype=torch.long)

        next_canvas, entropies = sampler.sample(probs, current_canvas)

        assert next_canvas.shape == (1, 5)
        assert entropies.shape == (1, 5)

        # 熵的单调性：位置 0 的熵为 0，位置 1 熵低，位置 3/4 熵最高
        assert entropies[0, 0].item() == 0.0
        assert entropies[0, 1].item() < entropies[0, 2].item()
        assert entropies[0, 2].item() < entropies[0, 3].item()

        # 置信度极高的位置 0 和 1 应该被接受（即 predictions 填入对应位置）
        assert next_canvas[0, 0].item() == 3
        assert next_canvas[0, 1].item() == 5


class TestDiffusionGemmaPipeline:
    """DiffusionGemmaPipeline 推理流程测试"""

    @requires_torch
    @requires_engine
    def test_pipeline_generate_shape(self):
        vocab_size = 100
        hidden_size = 64
        canvas_length = 32  # 使用短画布加快测试

        # 创建嵌入矩阵
        embedding_matrix = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))

        # 实例化模拟去噪器
        denoiser = MockDenoiser(vocab_size)

        # 实例化管道
        pipeline = DiffusionGemmaPipeline(
            denoiser=denoiser,
            embedding_matrix=embedding_matrix,
            vocab_size=vocab_size,
            canvas_length=canvas_length
        )

        prompt_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
        
        # 运行生成
        max_new_tokens = 64
        total_tokens = pipeline.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            num_steps_per_canvas=5,
            seed=42
        )

        # 结果长度应该等于 prompt_len + max_new_tokens
        assert total_tokens.shape == (1, 3 + max_new_tokens)
        assert total_tokens[0, 0].item() == 10
        assert total_tokens[0, 1].item() == 20
        assert total_tokens[0, 2].item() == 30

    @requires_torch
    @requires_engine
    def test_adaptive_early_stopping(self):
        vocab_size = 10
        hidden_size = 32
        canvas_length = 8

        embedding_matrix = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))
        
        # 建立一个去噪器，它能提供极高的置信度
        class ConfidentDenoiser(torch.nn.Module):
            def forward(self, x, kv):
                # 产生极强的置信度使得所有位置预测都是 0，且熵极低
                logits = torch.zeros(1, canvas_length, vocab_size, device=x.device, dtype=x.dtype)
                logits[:, :, 0] = 100.0  # 极强的确定性
                return logits

        denoiser = ConfidentDenoiser()
        pipeline = DiffusionGemmaPipeline(
            denoiser=denoiser,
            embedding_matrix=embedding_matrix,
            vocab_size=vocab_size,
            canvas_length=canvas_length
        )

        # 运行去噪，设置步数为 20
        # 确认它会在前几步就触发自适应早停（即能正常运行并结束，且不抛异常）
        kv_cache = torch.randn(1, 4, hidden_size)
        canvas = pipeline.run_denoising_loop(
            kv_cache=kv_cache,
            num_steps=20,
            seed=123
        )

        assert canvas.shape == (1, canvas_length)
        # 预测全都是最确定的 token 0
        assert torch.all(canvas == 0)
