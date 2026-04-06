"""TrinityTurbo attention implementation — Phase 2.

Extends TritonAttentionImpl to apply TurboQuant compression on
global attention layers. Sliding window layers pass through unchanged.

Phase 2 strategy: "decompress-all"
  - KV cache stores data in standard FP8 format (same tensor layout)
  - On cache write: standard FP8 quantize (via Triton backend)
  - On attention read: for global layers, additional compression benefit
    comes from the LayerRouter directing vLLM to allocate fewer pages

Note: Full in-cache compression (uint8 packed) requires custom Triton
kernels and will be implemented in Phase 2+. Phase 2 MVP demonstrates
the plugin architecture and validates correctness.
"""

from __future__ import annotations

import logging

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(TritonAttentionImpl):
    """Phase 2 attention with layer-aware compression pipeline.

    Inherits standard Triton attention and adds logging/instrumentation.
    The actual TurboQuant compression will be wired in incrementally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Log that we're using the Trinity-Turbo path
        if not hasattr(TrinityTurboAttentionImpl, '_logged'):
            logger.info(
                "TrinityTurboAttentionImpl active: heads=%d, kv_heads=%d, "
                "head_size=%d, sliding_window=%s",
                self.num_heads, self.num_kv_heads, self.head_size,
                self.sliding_window,
            )
            TrinityTurboAttentionImpl._logged = True
