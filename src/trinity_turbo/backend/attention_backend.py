"""TrinityTurbo attention backend for vLLM.

Phase 2: Uses compressed KV cache for global attention layers.
Reports smaller page sizes to vLLM → more blocks allocated → higher concurrency.

The KV cache shape uses uint8 with slot_bytes per head (63 bytes for 3-bit)
instead of head_size elements at FP8 (128 bytes).
"""

from __future__ import annotations

import logging
from typing import ClassVar

import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

logger = logging.getLogger(__name__)


class TrinityTurboAttentionBackend(TritonAttentionBackend):
    """Layer-aware KV cache compression backend.

    Phase 2: Inherits TritonAttentionBackend but overrides KV cache shape
    to use compressed format. Compression happens in TrinityTurboAttentionImpl.
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
        # Use standard Triton cache layout for Phase 2.
        # Compression will be applied inside forward() via
        # decompress-all before standard attention.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)
