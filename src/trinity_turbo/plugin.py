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

    logger.info("trinity-turbo: registered as AttentionBackendEnum.CUSTOM")
