from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .block_table import BlockTable
except ImportError:
    from block_table import BlockTable


@dataclass
class RequestState:
    request_id: str
    max_tokens: int
    block_size: int = 16
    block_table: BlockTable = field(init=False)
    seq_len: int = 0
    is_prefill_done: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.block_table = BlockTable(self.block_size)
