"""Feature flags for incremental enablement of compression techniques.

Each technique can be independently toggled. The system validates
that dependencies are satisfied (e.g., crosslayer requires compression).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from trinity_turbo.config import TrinityTurboConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureFlags:
    compression_enabled: bool
    eviction_enabled: bool
    crosslayer_enabled: bool
    crosslayer_lite: bool
    persistence_enabled: bool
    cutile_enabled: bool
    bits: int

    @classmethod
    def from_config(cls, config: TrinityTurboConfig) -> FeatureFlags:
        cutile_available = False
        if config.kernel_backend == "cutile":
            try:
                import cuda.tile  # noqa: F401
                import torch

                cap = torch.cuda.get_device_capability()
                cutile_available = cap >= (12, 0)
                if not cutile_available:
                    logger.warning(
                        "cuTile requested but GPU SM%d%d0 < SM120, falling back to Triton",
                        cap[0],
                        cap[1],
                    )
            except ImportError:
                logger.warning("cuTile requested but cuda-tile not installed, falling back to Triton")

        flags = cls(
            compression_enabled=config.enabled,
            eviction_enabled=config.eviction_enabled and config.enabled,
            crosslayer_enabled=config.crosslayer_mode != "off" and config.enabled,
            crosslayer_lite=config.crosslayer_mode == "lite",
            persistence_enabled=config.persistence_enabled and config.enabled,
            cutile_enabled=cutile_available,
            bits=config.bits,
        )
        flags._validate()
        return flags

    def _validate(self) -> None:
        if self.crosslayer_enabled and not self.compression_enabled:
            raise ValueError("Cross-layer reconstruction requires compression")
        if self.eviction_enabled and not self.compression_enabled:
            raise ValueError("Eviction requires compression")

    def describe(self) -> str:
        parts: list[str] = []
        if self.compression_enabled:
            parts.append(f"TurboQuant {self.bits}-bit")
        if self.eviction_enabled:
            parts.append("Gate-eviction")
        if self.crosslayer_enabled:
            mode = "Lite" if self.crosslayer_lite else "Learned"
            parts.append(f"CrossLayer-{mode}")
        if self.persistence_enabled:
            parts.append("Disk-persist")
        if self.cutile_enabled:
            parts.append("cuTile-SM120")
        return " + ".join(parts) if parts else "Disabled"
