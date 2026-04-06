"""TrinityTurbo attention backend for vLLM.

Phase 1 (MVP): Extends TritonAttentionBackend to validate the plugin
registration pipeline end-to-end. Uses standard Triton attention for
all computation — the custom compression will be added in Phase 2.

This approach ensures the plugin loads correctly, metadata flows through,
and we can benchmark the baseline before adding compression.
"""

from __future__ import annotations

import logging

from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

logger = logging.getLogger(__name__)


class TrinityTurboAttentionBackend(TritonAttentionBackend):
    """Layer-aware KV cache compression backend for Trinity-Large.

    Phase 1: Inherits TritonAttentionBackend entirely.
    Phase 2: Override get_impl_cls() to return TrinityTurboAttentionImpl
             with compressed KV cache on global layers.
    """

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"
