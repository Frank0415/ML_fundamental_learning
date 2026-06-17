from __future__ import annotations


class BlockTable:
    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.physical_block_ids: list[int] = []
        self._num_tokens = 0

    def append_block(self, block_id: int) -> None:
        if block_id in self.physical_block_ids:
            raise ValueError(f"block {block_id} is already in this table")
        self.physical_block_ids.append(block_id)

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_blocks(self) -> int:
        return len(self.physical_block_ids)

    def set_num_tokens(self, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        capacity = self.num_blocks * self.block_size
        if num_tokens > capacity:
            raise ValueError(f"num_tokens={num_tokens} exceeds block capacity={capacity}")
        self._num_tokens = num_tokens

    def token_to_block_offset(self, token_idx: int) -> tuple[int, int]:
        if token_idx < 0 or token_idx >= self._num_tokens:
            raise IndexError("token_idx out of request token range")
        table_idx = token_idx // self.block_size
        offset_in_block = token_idx % self.block_size
        return self.physical_block_ids[table_idx], offset_in_block

    def clear(self) -> None:
        self.physical_block_ids.clear()
        self._num_tokens = 0
