from __future__ import annotations

import torch
from torch import nn


class PagedKVCache(nn.Module):
    def __init__(
        self,
        total_blocks: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if min(total_blocks, num_layers, num_heads, head_dim, block_size) <= 0:
            raise ValueError("all PagedKVCache dimensions must be positive")
        self.total_blocks = total_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        shape = (total_blocks, num_layers, 2, block_size, num_heads, head_dim)
        self.register_buffer("kv_tensor", torch.empty(shape, dtype=dtype, device=device), persistent=False)

    def write_kv(
        self,
        block_id: int,
        layer: int,
        k_or_v: int,
        token_offset: int,
        data: torch.Tensor,
    ) -> None:
        self._check_indices(block_id, layer, k_or_v)
        data = self._normalize_data(data)
        end_offset = token_offset + data.shape[0]
        if token_offset < 0 or end_offset > self.block_size:
            raise IndexError("token write range exceeds block_size")
        expected_tail = (self.num_heads, self.head_dim)
        if tuple(data.shape[1:]) != expected_tail:
            raise ValueError(f"data tail must have shape {expected_tail}, got {tuple(data.shape[1:])}")
        self.kv_tensor[block_id, layer, k_or_v, token_offset:end_offset] = data.to(
            device=self.kv_tensor.device,
            dtype=self.kv_tensor.dtype,
        )

    def read_kv(self, block_id: int, layer: int, k_or_v: int, token_offset: int) -> torch.Tensor:
        self._check_indices(block_id, layer, k_or_v)
        if token_offset < 0 or token_offset >= self.block_size:
            raise IndexError("token_offset out of block range")
        return self.kv_tensor[block_id, layer, k_or_v, token_offset]

    def gather_kv_for_attention(
        self,
        physical_block_ids: list[int],
        layer: int,
        start_token: int,
        end_token: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_token < 0 or end_token < start_token:
            raise ValueError("invalid gather token range")
        if end_token == start_token:
            empty = self.kv_tensor.new_empty((0, self.num_heads, self.head_dim))
            return empty, empty.clone()
        if layer < 0 or layer >= self.num_layers:
            raise IndexError("layer out of range")

        block_ids = torch.tensor(physical_block_ids, device=self.kv_tensor.device, dtype=torch.long)
        if block_ids.numel() == 0:
            raise IndexError("cannot gather non-empty range from an empty block table")
        selected = torch.index_select(self.kv_tensor[:, layer], dim=0, index=block_ids)
        k_all = torch.cat([selected[:, 0, :, :, :].reshape(-1, self.num_heads, self.head_dim)], dim=0)
        v_all = torch.cat([selected[:, 1, :, :, :].reshape(-1, self.num_heads, self.head_dim)], dim=0)
        if end_token > k_all.shape[0]:
            raise IndexError("end_token exceeds physical block capacity")
        return k_all[start_token:end_token], v_all[start_token:end_token]

    def _check_indices(self, block_id: int, layer: int, k_or_v: int) -> None:
        if block_id < 0 or block_id >= self.total_blocks:
            raise IndexError("block_id out of range")
        if layer < 0 or layer >= self.num_layers:
            raise IndexError("layer out of range")
        if k_or_v not in (0, 1):
            raise IndexError("k_or_v must be 0 (K) or 1 (V)")

    @staticmethod
    def _normalize_data(data: torch.Tensor) -> torch.Tensor:
        if data.ndim == 2:
            return data.unsqueeze(0)
        if data.ndim != 3:
            raise ValueError("data must have shape [tokens, heads, head_dim] or [heads, head_dim]")
        return data
