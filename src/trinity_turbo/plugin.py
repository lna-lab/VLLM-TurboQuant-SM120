"""vLLM plugin entry point for trinity-turbo.

Registered via pyproject.toml [project.entry-points."vllm.general_plugins"].
Called by vLLM's load_general_plugins() during startup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_trinity_turbo() -> None:
    """Entry point called by vLLM's plugin loader.

    Registers TrinityTurboAttentionBackend as CUSTOM backend.
    """
    from trinity_turbo.config import TrinityTurboConfig, set_global_config
    from trinity_turbo.features import FeatureFlags

    config = TrinityTurboConfig.from_env()
    if not config.enabled:
        logger.info("trinity-turbo: disabled via TRINITY_TURBO_ENABLED=0")
        return

    config.validate()
    set_global_config(config)

    flags = FeatureFlags.from_config(config)
    logger.info("trinity-turbo: registering backend [%s]", flags.describe())

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    # register_backend expects a dotted string path, not a class object
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "trinity_turbo.backend.attention_backend.TrinityTurboAttentionBackend",
    )

    # Monkey-patch Attention.get_kv_cache_spec to return compressed spec
    # for global attention layers (non-sliding-window)
    _patch_kv_cache_spec(config)

    logger.info("trinity-turbo: registered as AttentionBackendEnum.CUSTOM")


def _patch_kv_cache_spec(config: "TrinityTurboConfig") -> None:
    """Patch Attention.get_kv_cache_spec to report compressed page sizes
    for global attention layers.

    This causes vLLM's memory allocator to allocate more blocks for
    global attention layers, enabling higher concurrent request counts.
    """
    from trinity_turbo.backend.cache_spec import CompressedFullAttentionSpec
    from trinity_turbo.quant.turboquant import QuantState

    from vllm.model_executor.layers.attention.attention import Attention

    # Calculate slot_bytes for the configured bit width
    raw_slot_bytes = QuantState.create(
        bits=config.bits,
        head_dim=128,  # Trinity's head_dim
        num_outliers=config.num_outlier_channels,
        device="cpu",
    ).slot_bytes
    # Pad to power of 2 for vLLM page alignment
    slot_bytes = 1
    while slot_bytes < raw_slot_bytes:
        slot_bytes <<= 1

    original_get_spec = Attention.get_kv_cache_spec

    def patched_get_kv_cache_spec(self, vllm_config):
        spec = original_get_spec(self, vllm_config)

        # Only compress FullAttentionSpec (global layers)
        # SlidingWindowSpec (local layers) pass through unchanged
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        if isinstance(spec, FullAttentionSpec) and not isinstance(spec, CompressedFullAttentionSpec):
            compressed = CompressedFullAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                dtype=spec.dtype,
                sliding_window=spec.sliding_window,
                slot_bytes_per_head=slot_bytes,
            )
            logger.debug(
                "trinity-turbo: compressed spec for layer: %d bytes/page → %d bytes/page (%.1fx)",
                spec.real_page_size_bytes,
                compressed.real_page_size_bytes,
                spec.real_page_size_bytes / compressed.real_page_size_bytes,
            )
            return compressed
        return spec

    Attention.get_kv_cache_spec = patched_get_kv_cache_spec
    logger.info(
        "trinity-turbo: patched get_kv_cache_spec (slot_bytes=%d, %.1fx vs FP8)",
        slot_bytes, 128 / slot_bytes,
    )
