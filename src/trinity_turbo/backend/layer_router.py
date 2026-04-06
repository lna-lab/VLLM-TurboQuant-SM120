"""Per-layer compression strategy routing for Trinity's hybrid attention.

Trinity-Large has 60 layers:
  - 45 sliding window layers (every 1st-3rd in each group of 4): bounded at 4096 tokens
  - 15 global attention layers (every 4th layer): unbounded, these need compression

The LayerRouter classifies each layer and assigns a strategy.
"""

from __future__ import annotations

import logging
from enum import Enum

from trinity_turbo.features import FeatureFlags

logger = logging.getLogger(__name__)


class LayerStrategy(Enum):
    PASSTHROUGH = "passthrough"  # Sliding window: no compression needed
    COMPRESS = "compress"  # Global: TurboQuant compress
    COMPRESS_EVICT = "compress_evict"  # Global: TurboQuant + gate eviction
    ANCHOR = "anchor"  # Global: anchor layer for cross-layer reconstruction
    RECONSTRUCT = "reconstruct"  # Global: reconstruct from anchors


class LayerRouter:
    """Decides per-layer strategy based on model config and feature flags."""

    def __init__(
        self,
        layer_types: list[str],
        num_layers: int,
        features: FeatureFlags,
    ) -> None:
        self.num_layers = num_layers
        self.strategies: dict[int, LayerStrategy] = {}
        self.anchor_layers: tuple[int, int] = (-1, -1)

        # Identify global attention layer indices
        global_layers = [i for i in range(num_layers) if layer_types[i] != "sliding_attention"]
        sliding_layers = [i for i in range(num_layers) if layer_types[i] == "sliding_attention"]

        logger.info(
            "LayerRouter: %d sliding window layers, %d global attention layers",
            len(sliding_layers),
            len(global_layers),
        )

        for i in range(num_layers):
            if layer_types[i] == "sliding_attention":
                self.strategies[i] = LayerStrategy.PASSTHROUGH
            elif features.crosslayer_enabled and len(global_layers) >= 3:
                # Pick 2 anchors: first and middle global layer
                if i == global_layers[0]:
                    self.strategies[i] = LayerStrategy.ANCHOR
                    self.anchor_layers = (i, self.anchor_layers[1])
                elif i == global_layers[len(global_layers) // 2]:
                    self.strategies[i] = LayerStrategy.ANCHOR
                    self.anchor_layers = (self.anchor_layers[0], i)
                else:
                    self.strategies[i] = LayerStrategy.RECONSTRUCT
            elif features.eviction_enabled:
                self.strategies[i] = LayerStrategy.COMPRESS_EVICT
            else:
                self.strategies[i] = LayerStrategy.COMPRESS

        # Log strategy summary
        counts: dict[str, int] = {}
        for s in self.strategies.values():
            counts[s.value] = counts.get(s.value, 0) + 1
        logger.info("LayerRouter strategies: %s", counts)
        if self.anchor_layers[0] >= 0:
            logger.info("LayerRouter anchors: bottom=%d, middle=%d", *self.anchor_layers)

    def get_strategy(self, layer_idx: int) -> LayerStrategy:
        return self.strategies[layer_idx]

    def needs_kv_storage(self, layer_idx: int) -> bool:
        """Whether this layer stores KV in compressed cache."""
        return self.strategies[layer_idx] in (
            LayerStrategy.COMPRESS,
            LayerStrategy.COMPRESS_EVICT,
            LayerStrategy.ANCHOR,
        )

    def needs_reconstruction(self, layer_idx: int) -> bool:
        return self.strategies[layer_idx] == LayerStrategy.RECONSTRUCT

    def is_passthrough(self, layer_idx: int) -> bool:
        return self.strategies[layer_idx] == LayerStrategy.PASSTHROUGH

    @property
    def global_layer_indices(self) -> list[int]:
        return [i for i, s in self.strategies.items() if s != LayerStrategy.PASSTHROUGH]

    @property
    def compressed_layer_count(self) -> int:
        return sum(1 for s in self.strategies.values() if s in (
            LayerStrategy.COMPRESS, LayerStrategy.COMPRESS_EVICT, LayerStrategy.ANCHOR,
        ))
