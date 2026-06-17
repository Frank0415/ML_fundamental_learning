"""
diffusion_gemma_pipeline.py — DiffusionGemma 的高效推理实现（离散文本扩散）

本模块提供 DiffusionGemma 离散文本扩散推理流程的核心实现：
- EntropyBoundedSampler: 熵界采样器，支持熵排序、熵界过滤与拒绝 token 再噪声化
- DiffusionGemmaPipeline: 完整的离散扩散推理管道，支持自适应早停、Self-Conditioning 以及分块自回归多画布拼接
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyBoundedSampler:
    """
    Entropy Bounded Sampler（熵界采样器）。
    
    用于离散文本扩散。每个去噪步中，采样器根据模型预测的概率分布：
    1. 计算每个位置的香农熵以评估置信度。
    2. 按置信度由高到低（熵由低到高）排序。
    3. 利用累加熵界规则（Entropy Bound）决定保留（接受）哪些 Token，重置（拒绝）哪些 Token。
    4. 对被拒绝的 Token 进行再噪声化（用词表中的均匀随机 Token 覆盖）。
    """

    def __init__(self, vocab_size: int, entropy_bound: float = 0.1, renoise_seed: Optional[int] = None):
        """
        参数:
            vocab_size: 词表大小。
            entropy_bound: 熵接纳上限阈值（通常为 0.1）。
            renoise_seed: 随机种子，控制再噪声化生成。
        """
        self.vocab_size = vocab_size
        self.entropy_bound = entropy_bound
        self._generator = torch.Generator()
        if renoise_seed is not None:
            self._generator.manual_seed(renoise_seed)

    def sample(
        self,
        probs: torch.Tensor,
        current_canvas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据概率预测执行熵排序、接纳判定和再噪声化。

        参数:
            probs: 模型输出的概率分布，shape (1, seq_len, vocab_size)。
            current_canvas: 当前步的 Token 画布，shape (1, seq_len)。

        返回:
            next_canvas: 更新后的 Token 画布，shape (1, seq_len)。
            entropies: 画布上各位置的熵值，shape (1, seq_len)。
        """
        device = probs.device
        seq_len = probs.shape[1]

        # 1. 计算每个位置的熵：H = -sum(p * log(p))
        # 避免概率为 0 导致 log 产生 NaN
        eps = 1e-9
        entropies = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # (1, seq_len)

        # 2. 预测概率最大的 Token
        predictions = torch.argmax(probs, dim=-1)  # (1, seq_len)

        # 3. 按熵值由小到大（置信度由高到低）排序
        sorted_entropies, sorted_indices = torch.sort(entropies, dim=-1)  # (1, seq_len)

        # 4. 熵界过滤 (Entropy Bounding Criteria)
        # 根据公式：在累加各位置的熵时，若累加和（减去当前接纳组内最大熵，或直接累加和）未超过限制则接纳。
        # 简化版实现：计算累计熵值累计和，当累计和小于 self.entropy_bound 时接纳 Token。
        cumsum_entropy = torch.cumsum(sorted_entropies, dim=-1)
        accepted_mask_sorted = cumsum_entropy <= self.entropy_bound
        
        # 至少接受最自信的那一个，防止极端情况下全部拒绝导致死循环
        accepted_mask_sorted[:, 0] = True

        # 将 sorted 的 mask 映射回原始的画布序列位置
        accepted_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        accepted_mask.scatter_(1, sorted_indices, accepted_mask_sorted)

        # 5. 更新画布：接受位置替换为预测值，拒绝位置进行再噪声化（填充词表随机数）
        # 从词表中均匀采样随机 Token 填入被拒绝位置
        random_tokens = torch.randint(
            0, self.vocab_size, (1, seq_len),
            generator=self._generator if self._generator.device == device.type else None,
            device=device
        )

        next_canvas = torch.where(accepted_mask, predictions, random_tokens)
        return next_canvas, entropies


class DiffusionGemmaPipeline:
    """
    DiffusionGemma 离散文本扩散推理管道。

    封装多画布采样（Block Autoregressive Diffusion）和自适应早停（Adaptive Stopping）流程。
    """

    def __init__(
        self,
        denoiser: nn.Module,
        embedding_matrix: nn.Parameter,
        vocab_size: int,
        canvas_length: int = 256,
        self_cond_proj: Optional[nn.Module] = None
    ):
        """
        参数:
            denoiser: 去噪模块。应接收 (x_embeddings, kv_cache) 并返回 logits。
            embedding_matrix: 模型 Token 嵌入表矩阵 (vocab_size, hidden_size)，用于 Self-Conditioning。
            vocab_size: 词表大小。
            canvas_length: 扩散画布大小，默认 256。
            self_cond_proj: Self-Conditioning 映射层 (nn.Linear)，若无则采用恒等映射。
        """
        self.denoiser = denoiser
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.canvas_length = canvas_length
        self.hidden_size = embedding_matrix.shape[1]
        
        if self_cond_proj is not None:
            self.self_cond_proj = self_cond_proj
        else:
            self.self_cond_proj = nn.Linear(self.hidden_size, self.hidden_size)
            # 初始化为接近 0 的小权重，防止初期扰乱嵌入
            nn.init.zeros_(self.self_cond_proj.weight)
            nn.init.zeros_(self.self_cond_proj.bias)

    @torch.no_grad()
    def run_denoising_loop(
        self,
        kv_cache: torch.Tensor,
        num_steps: int = 48,
        temp_start: float = 0.8,
        temp_end: float = 0.4,
        entropy_bound: float = 0.1,
        adaptive_stop_threshold: float = 0.005,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        单次画布 (256 Token) 的扩散去噪循环。

        参数:
            kv_cache: Prompt 编码后只读的 KV Cache 表示（例如输入 prompt 的 context 嵌入）。
            num_steps: 去噪步数（默认 48 步）。
            temp_start: Logits 初始采样温度（默认 0.8）。
            temp_end: Logits 终止采样温度（默认 0.4）。
            entropy_bound: 熵界参数（默认 0.1）。
            adaptive_stop_threshold: 自适应早停的平均熵阈值（默认 0.005）。
            seed: 随机种子。

        返回:
            denoised_canvas: 去噪完成的 Token 序列，shape (1, canvas_length)。
        """
        device = self.embedding_matrix.device
        dtype = self.embedding_matrix.dtype

        # 1. 熵界采样器实例化
        sampler = EntropyBoundedSampler(self.vocab_size, entropy_bound=entropy_bound, renoise_seed=seed)

        # 2. 画布随机初始化 (从词表中均匀随机生成)
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        current_canvas = torch.randint(
            0, self.vocab_size, (1, self.canvas_length),
            generator=generator, device=device
        )

        # 3. 初始化 Self-Conditioning 的上一轮概率分布记录 (全 0 或均匀分布)
        prev_probs = torch.zeros(1, self.canvas_length, self.vocab_size, device=device, dtype=dtype)
        last_predictions = current_canvas.clone()
        stable_count = 0

        # 4. 迭代去噪主循环
        for step in range(num_steps):
            # A. 计算当前步的 Logits 温度 T (线性衰减)
            t_frac = step / max(1, num_steps - 1)
            temp = temp_start - (temp_start - temp_end) * t_frac

            # B. 获取当前画布的嵌入表示
            # embedding_matrix shape: (vocab_size, hidden_size)
            canvas_embeddings = F.embedding(current_canvas, self.embedding_matrix)  # (1, canvas_len, hidden_size)

            # C. Self-Conditioning (自我记忆调节)
            # 通过上一轮的概率分布 prev_probs 与嵌入矩阵进行点积结合，投影后注入
            # prev_probs: (1, canvas_len, vocab_size) @ (vocab_size, hidden) -> (1, canvas_len, hidden)
            self_cond_vector = torch.matmul(prev_probs, self.embedding_matrix)
            self_cond_offset = self.self_cond_proj(self_cond_vector)
            
            # 融合嵌入
            x_input = canvas_embeddings + self_cond_offset

            # D. 前向传播模型以获得 Logits
            # 输入: 融合后的 Token 嵌入 + Prompt KV Cache 的表示
            logits = self.denoiser(x_input, kv_cache)  # (1, canvas_len, vocab_size)

            # E. 概率缩放及归一化 (Logits Temperature Scheduler)
            scaled_logits = logits / max(temp, 1e-4)
            probs = F.softmax(scaled_logits, dim=-1)

            # F. 熵界采样更新画布及噪声重置
            next_canvas, entropies = sampler.sample(probs, current_canvas)

            # G. 自适应早停判定 (Adaptive Stopping)
            # 条件一：全画布平均预测熵低于阈值 (高度确信)
            mean_entropy = torch.mean(entropies).item()
            is_confident = mean_entropy < adaptive_stop_threshold

            # 条件二：最高预测 Token 已经稳定 (连续 2 步未变)
            predictions = torch.argmax(probs, dim=-1)
            is_stable_step = torch.equal(predictions, last_predictions)
            if is_stable_step:
                stable_count += 1
            else:
                stable_count = 0

            # 记录历史用于下一迭代
            current_canvas = next_canvas
            prev_probs = probs
            last_predictions = predictions

            # 达到稳定状态（如连续 2 步不变）并且平均熵非常低，即可早停
            if is_confident and stable_count >= 1:
                # print(f"[DiffusionGemma] Early stopped at step {step + 1} (mean_entropy: {mean_entropy:.5f})")
                break

        return current_canvas

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 512,
        num_steps_per_canvas: int = 48,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        执行分块自回归多画布拼接生成。

        参数:
            prompt_ids: 输入 Prompt 的 Token IDs，shape (1, prompt_len)。
            max_new_tokens: 最大生成的新 Token 数量。
            num_steps_per_canvas: 每个画布迭代的步数。
            seed: 随机种子。

        返回:
            total_ids: 完整生成的 Token 序列（含 prompt），shape (1, prompt_len + generated_len)。
        """
        device = prompt_ids.device
        
        # 1. 模拟编码阶段产生 KV Cache (模拟机制)
        # 用 prompt_ids 计算静态的 prompt/KV Cache 激活向量
        # 在真实 DiffusionGemma 中此阶段使用 Causal Attention 进行 Prefill
        prompt_embeddings = F.embedding(prompt_ids, self.embedding_matrix)
        kv_cache = prompt_embeddings  # 作为 denoiser 的跨注意力上下文

        generated_tokens = []
        tokens_to_generate = max_new_tokens
        current_seed = seed

        # 2. 分块自回归循环 (Block Autoregressive Denoising)
        while tokens_to_generate > 0:
            # 每次运行去噪循环生成一画布 (256 Token)
            block_tokens = self.run_denoising_loop(
                kv_cache=kv_cache,
                num_steps=num_steps_per_canvas,
                seed=current_seed
            )

            # 追加新生成的块
            generated_tokens.append(block_tokens)
            tokens_to_generate -= self.canvas_length

            # 更新后续块的随机种子，保持随机性
            if current_seed is not None:
                current_seed += 1

            # 3. 自回归 KV Cache 扩展 (Incremental Prefill)
            # 新生成的 256 画布在去噪完成后，被追加回 KV Cache 中，供下一块去噪时参考
            new_block_embeddings = F.embedding(block_tokens, self.embedding_matrix)
            kv_cache = torch.cat([kv_cache, new_block_embeddings], dim=1)

            # 如果生成的块中包含终止符 (例如 EOS_TOKEN = 1)，可提前终止分块循环
            if 1 in block_tokens:
                break

        # 4. 组合结果并返回
        all_generated = torch.cat(generated_tokens, dim=1)
        total_ids = torch.cat([prompt_ids, all_generated[:, :max_new_tokens]], dim=1)
        return total_ids
