from __future__ import annotations


class OutOfKVBlocks(RuntimeError):
    pass


class BlockManager:
    def __init__(self, total_blocks: int, block_size: int, reserved_system_blocks: int = 0) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if reserved_system_blocks < 0 or reserved_system_blocks >= total_blocks:
            raise ValueError("reserved_system_blocks must be in [0, total_blocks)")

        self.total_blocks = total_blocks
        self.block_size = block_size
        self.reserved_system_blocks = reserved_system_blocks
        self.free_blocks: set[int] = set(range(reserved_system_blocks, total_blocks))
        self.allocated_blocks: set[int] = set()
        self._block_used_tokens: dict[int, int] = {}

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise OutOfKVBlocks("no free KV blocks available")
        block_id = min(self.free_blocks)
        self.free_blocks.remove(block_id)
        self.allocated_blocks.add(block_id)
        self._block_used_tokens[block_id] = 0
        return block_id

    def free_blocks_of_request(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            if block_id not in self.allocated_blocks:
                raise ValueError(f"block {block_id} is not allocated")
        for block_id in block_ids:
            self.allocated_blocks.remove(block_id)
            self.free_blocks.add(block_id)
            self._block_used_tokens.pop(block_id, None)

    def available_blocks(self) -> int:
        return len(self.free_blocks)

    def set_block_used_tokens(self, block_id: int, used_tokens: int) -> None:
        if block_id not in self.allocated_blocks:
            raise ValueError(f"block {block_id} is not allocated")
        if used_tokens < 0 or used_tokens > self.block_size:
            raise ValueError("used_tokens must fit inside one block")
        self._block_used_tokens[block_id] = used_tokens

    def fragmentation_ratio(self) -> float:
        allocated_slots = len(self.allocated_blocks) * self.block_size
        if allocated_slots == 0:
            return 0.0
        return self._wasted_slots() / allocated_slots

    def stats(self) -> dict[str, int | float]:
        return {
            "allocated_blocks": len(self.allocated_blocks),
            "free_blocks": len(self.free_blocks),
            "total_blocks": self.total_blocks,
            "used_tokens": self._used_tokens(),
            "wasted_slots": self._wasted_slots(),
            "fragmentation_ratio": self.fragmentation_ratio(),
        }

    def _used_tokens(self) -> int:
        return sum(self._block_used_tokens.values())

    def _wasted_slots(self) -> int:
        return len(self.allocated_blocks) * self.block_size - self._used_tokens()
