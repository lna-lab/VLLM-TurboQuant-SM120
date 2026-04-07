"""TrinityTurbo attention backend for vLLM.

Phase 2: Compressed uint8 KV cache (64 bytes/head instead of 128).
All layers use compressed format — sliding window included.

The three pieces that must agree ("三位一体"):
  1. cache_spec.real_page_size_bytes  → how much memory vLLM reserves per page
  2. get_kv_cache_shape()             → actual tensor dimensions
  3. do_kv_cache_update() / forward() → what gets written / read
"""

from __future__ import annotations

import logging
from typing import ClassVar

from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

from trinity_turbo.kernels.triton_compress import SLOT_BYTES

logger = logging.getLogger(__name__)


class TrinityTurboAttentionBackend(TritonAttentionBackend):
    """Layer-aware KV cache compression backend.

    Phase 2: last cache dimension is SLOT_BYTES (64) instead of head_size (128).
    The tensor dtype stays fp8/uint8 (1 byte per element), so each head uses
    64 bytes instead of 128 → 2× memory reduction per page.
    """

    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        from trinity_turbo.backend.attention_impl import TrinityTurboAttentionImpl
        return TrinityTurboAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Phase 2: compressed slot layout (64 bytes per head per token)
        # instead of head_size (128 bytes with fp8).
        return (num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES)
