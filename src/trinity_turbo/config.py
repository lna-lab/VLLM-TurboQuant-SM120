"""Configuration for trinity-turbo plugin.

All settings are read from environment variables with TRINITY_TURBO_ prefix.
Defaults are tuned for Trinity-Large-Thinking-W4A16 on 4x RTX PRO 6000 Blackwell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

_global_config: TrinityTurboConfig | None = None


@dataclass
class TrinityTurboConfig:
    # Master switch
    enabled: bool = True

    # TurboQuant settings
    bits: int = 3  # 2, 3, or 4
    num_outlier_channels: int = 8

    # Eviction settings (Phase 2)
    eviction_enabled: bool = False
    eviction_keep_ratio: float = 0.30
    num_sink_tokens: int = 16

    # Cross-layer reconstruction (Phase 2)
    crosslayer_mode: str = "off"  # off, lite, learned

    # Disk persistence (Phase 3)
    persistence_enabled: bool = False
    persistence_dir: str = "/tmp/trinity-turbo-cache"

    # Kernel backend
    kernel_backend: str = "triton"  # triton, cutile

    @classmethod
    def from_env(cls) -> TrinityTurboConfig:
        """Parse all TRINITY_TURBO_* environment variables."""
        kwargs: dict = {}
        for f in fields(cls):
            env_key = f"TRINITY_TURBO_{f.name.upper()}"
            val = os.environ.get(env_key)
            if val is not None:
                if f.type == "bool" or f.type is bool:
                    kwargs[f.name] = val.lower() in ("1", "true", "yes")
                elif f.type == "int" or f.type is int:
                    kwargs[f.name] = int(val)
                elif f.type == "float" or f.type is float:
                    kwargs[f.name] = float(val)
                else:
                    kwargs[f.name] = val
        return cls(**kwargs)

    def validate(self) -> None:
        if self.bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {self.bits}")
        if self.crosslayer_mode not in ("off", "lite", "learned"):
            raise ValueError(f"crosslayer_mode must be off/lite/learned, got {self.crosslayer_mode}")
        if self.kernel_backend not in ("triton", "cutile"):
            raise ValueError(f"kernel_backend must be triton/cutile, got {self.kernel_backend}")
        if not 0.0 < self.eviction_keep_ratio <= 1.0:
            raise ValueError(f"eviction_keep_ratio must be in (0, 1], got {self.eviction_keep_ratio}")


def set_global_config(config: TrinityTurboConfig) -> None:
    global _global_config
    _global_config = config


def get_global_config() -> TrinityTurboConfig:
    global _global_config
    if _global_config is None:
        _global_config = TrinityTurboConfig.from_env()
    return _global_config
