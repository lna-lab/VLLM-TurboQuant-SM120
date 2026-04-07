"""Custom KV cache spec that reports compressed page sizes.

By overriding real_page_size_bytes, we tell vLLM's memory allocator
that our pages use less memory per token. This causes vLLM to allocate
more blocks, enabling higher concurrent request counts.

For Trinity's global attention layers (15/60):
  FP8 baseline: 2 × block_size × num_kv_heads × 128 × 1 byte = 512 bytes/token/head
  TurboQuant 3-bit: 2 × block_size × num_kv_heads × 63 bytes = 252 bytes/token/head
  ~2x compression on global layers

Since 45/60 layers use sliding window (bounded at 4096 tokens),
the compression benefit is concentrated on the 15 global layers
that dominate memory at long contexts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec


class _CompressedPageMixin:
    """Shared real_page_size_bytes for compressed specs."""

    slot_bytes_per_head: int

    @property
    def real_page_size_bytes(self) -> int:
        # K + V, each using slot_bytes_per_head per KV head per token
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.slot_bytes_per_head
        )


@dataclass(frozen=True, kw_only=True)
class CompressedFullAttentionSpec(_CompressedPageMixin, FullAttentionSpec):
    """FullAttentionSpec with compressed page size."""

    slot_bytes_per_head: int = 64


@dataclass(frozen=True, kw_only=True)
class CompressedSlidingWindowSpec(_CompressedPageMixin, SlidingWindowSpec):
    """SlidingWindowSpec with compressed page size."""

    slot_bytes_per_head: int = 64
