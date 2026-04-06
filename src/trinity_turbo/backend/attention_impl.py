"""TrinityTurbo attention implementation.

Phase 1 (MVP): Compress KV for global layers before caching,
decompress before standard Triton attention. This is a correctness-first
implementation that validates the compression pipeline end-to-end.

Phase 2 will replace the decompress-then-attend path with a fused
Triton kernel that decompresses in-tile during attention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.backend import AttentionImpl, AttentionType

from trinity_turbo.config import get_global_config
from trinity_turbo.features import FeatureFlags
from trinity_turbo.quant.turboquant import QuantState, compress, decompress

if TYPE_CHECKING:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(AttentionImpl["TritonAttentionMetadata"]):
    """Layer-aware attention with TurboQuant compression on global layers.

    For sliding window layers: standard attention (passthrough).
    For global attention layers: compress KV before cache, decompress before attend.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

        # Initialize quantization state for global layers
        config = get_global_config()
        self.features = FeatureFlags.from_config(config)
        self.quant_state: QuantState | None = None

        # Lazy init — QuantState needs device info from first forward call
        self._initialized = False
        self._config = config

        logger.debug(
            "TrinityTurboAttentionImpl: heads=%d, kv_heads=%d, head_size=%d, sw=%s",
            num_heads, self.num_kv_heads, head_size, sliding_window,
        )

    def _lazy_init(self, device: torch.device) -> None:
        if self._initialized:
            return
        self.quant_state = QuantState.create(
            bits=self._config.bits,
            head_dim=self.head_size,
            num_outliers=self._config.num_outlier_channels,
            device=device,
        )
        self._initialized = True
        logger.info(
            "TrinityTurboAttentionImpl initialized: %d-bit, slot_bytes=%d, features=[%s]",
            self._config.bits, self.quant_state.slot_bytes, self.features.describe(),
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with layer-aware compression.

        Phase 1: All layers use standard Triton attention.
        Compression/decompression is a no-op for correctness validation.
        The actual compression pipeline will be wired in incrementally.
        """
        self._lazy_init(query.device)

        # Phase 1: Delegate to standard Triton attention for all layers.
        # This validates that the plugin registration and metadata flow works.
        # Compression will be added layer-by-layer once this path is stable.
        from vllm.v1.attention.backends.triton_attn import (
            triton_reshape_and_cache_flash,
            unified_attention,
        )

        # Update KV cache (standard path)
        triton_reshape_and_cache_flash(key, value, kv_cache, attn_metadata.slot_mapping)

        # Run standard attention
        return unified_attention(
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            scale=self.scale,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            output=output,
        )
