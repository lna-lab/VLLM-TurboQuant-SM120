"""vLLM plugin entry point for trinity-turbo.

Registered via pyproject.toml [project.entry-points."vllm.general_plugins"].
Called by vLLM's load_general_plugins() during startup.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_trinity_turbo() -> None:
    """Entry point called by vLLM's plugin loader."""
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

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "trinity_turbo.backend.attention_backend.TrinityTurboAttentionBackend",
    )

    _patch_kv_cache_spec(config)
    logger.info("trinity-turbo: registered as AttentionBackendEnum.CUSTOM")


def _patch_kv_cache_spec(config: "TrinityTurboConfig") -> None:
    """Patch Attention.get_kv_cache_spec to report head_size=SLOT_BYTES.

    This makes real_page_size_bytes agree with get_kv_cache_shape's last dim.
    Uses STANDARD vLLM spec classes (no custom subclasses) so the KV cache
    manager recognizes them without KeyError.
    """
    from trinity_turbo.kernels.triton_compress import SLOT_BYTES

    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    original_get_spec = Attention.get_kv_cache_spec

    def patched_get_kv_cache_spec(self, vllm_config):
        spec = original_get_spec(self, vllm_config)

        # Replace head_size with SLOT_BYTES so that
        # real_page_size_bytes = 2 × bs × heads × SLOT_BYTES × dtype_size
        # matches get_kv_cache_shape's last dim exactly.

        if isinstance(spec, SlidingWindowSpec):
            if spec.head_size != SLOT_BYTES:
                spec = SlidingWindowSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=SLOT_BYTES,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                )
            return spec

        if isinstance(spec, FullAttentionSpec):
            if spec.head_size != SLOT_BYTES:
                spec = FullAttentionSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=SLOT_BYTES,
                    dtype=spec.dtype,
                    sliding_window=spec.sliding_window,
                )
            return spec

        return spec

    Attention.get_kv_cache_spec = patched_get_kv_cache_spec
    logger.info(
        "trinity-turbo: patched get_kv_cache_spec (head_size→%d, %.1fx vs FP8)",
        SLOT_BYTES, 128 / SLOT_BYTES,
    )
