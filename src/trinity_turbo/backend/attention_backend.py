"""TrinityTurbo attention backend for vLLM.

Registers as AttentionBackendEnum.CUSTOM. Provides layer-aware KV cache
compression: sliding window layers pass through, global layers get
TurboQuant compressed.

Phase 1 (MVP): Compress global layers, decompress before standard attention.
Phase 2: Fused paged decode kernel with in-tile decompression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.v1.attention.backend import AttentionBackend

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionImpl, AttentionMetadataBuilder


class TrinityTurboAttentionBackend(AttentionBackend):
    """Layer-aware KV cache compression backend for Trinity-Large."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_name() -> str:
        return "TRINITY_TURBO"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        from trinity_turbo.backend.attention_impl import TrinityTurboAttentionImpl
        return TrinityTurboAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        # Reuse the Triton attention metadata builder since our metadata
        # format is compatible (cu_seqlens, block_table, slot_mapping).
        from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Phase 1: Use standard layout, compression happens in forward().
        # This allows sliding window layers to use the same cache format.
        # The compressed format will be introduced in Phase 2 with custom
        # cache spec that returns different shapes per layer group.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_compute_capability(cls, capability: object) -> bool:
        return True  # SM120+ preferred but works on any GPU
