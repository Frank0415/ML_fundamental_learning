from __future__ import annotations

try:
    from .block_manager import BlockManager
    from .request_state import RequestState
except ImportError:
    from block_manager import BlockManager
    from request_state import RequestState


def prefill_allocate(block_mgr: BlockManager, request: RequestState, prompt_len: int) -> None:
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    if prompt_len > request.max_tokens:
        raise ValueError("prompt_len exceeds request.max_tokens")
    needed_blocks = (prompt_len + block_mgr.block_size - 1) // block_mgr.block_size
    for _ in range(needed_blocks):
        request.block_table.append_block(block_mgr.allocate_block())
    request.seq_len = prompt_len
    request.block_table.set_num_tokens(prompt_len)
    _sync_block_usage(block_mgr, request)
    request.is_prefill_done = True


def decode_append(block_mgr: BlockManager, request: RequestState) -> int | None:
    if not request.is_prefill_done:
        raise ValueError("prefill must complete before decode_append")
    if request.seq_len >= request.max_tokens:
        raise ValueError("request has reached max_tokens")

    new_block_id = None
    if request.seq_len == request.block_table.num_blocks * block_mgr.block_size:
        new_block_id = block_mgr.allocate_block()
        request.block_table.append_block(new_block_id)
    request.seq_len += 1
    request.block_table.set_num_tokens(request.seq_len)
    _sync_block_usage(block_mgr, request)
    return new_block_id


def free_request(block_mgr: BlockManager, request: RequestState) -> None:
    block_ids = list(request.block_table.physical_block_ids)
    if block_ids:
        block_mgr.free_blocks_of_request(block_ids)
    request.block_table.clear()
    request.seq_len = 0
    request.is_prefill_done = False


def _sync_block_usage(block_mgr: BlockManager, request: RequestState) -> None:
    remaining = request.seq_len
    for block_id in request.block_table.physical_block_ids:
        used = min(block_mgr.block_size, remaining)
        block_mgr.set_block_used_tokens(block_id, used)
        remaining -= used
