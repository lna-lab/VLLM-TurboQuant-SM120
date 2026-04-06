"""Tests for LayerRouter with Trinity's 45+15 layer configuration."""

from trinity_turbo.backend.layer_router import LayerRouter, LayerStrategy
from trinity_turbo.features import FeatureFlags
from trinity_turbo.config import TrinityTurboConfig


def _trinity_layer_types() -> list[str]:
    """Generate Trinity's 60-layer type pattern: 3:1 sliding:global."""
    types = []
    for i in range(60):
        if (i + 1) % 4 == 0:  # Every 4th layer is global
            types.append("attention")
        else:
            types.append("sliding_attention")
    return types


def test_basic_compress():
    """With compression only, global layers should be COMPRESS."""
    config = TrinityTurboConfig(enabled=True, eviction_enabled=False, crosslayer_mode="off")
    features = FeatureFlags.from_config(config)
    layer_types = _trinity_layer_types()

    router = LayerRouter(layer_types, 60, features)

    sliding_count = sum(1 for i in range(60) if router.is_passthrough(i))
    compress_count = sum(1 for i in range(60) if router.get_strategy(i) == LayerStrategy.COMPRESS)

    assert sliding_count == 45
    assert compress_count == 15


def test_eviction_mode():
    """With eviction enabled, global layers should be COMPRESS_EVICT."""
    config = TrinityTurboConfig(enabled=True, eviction_enabled=True, crosslayer_mode="off")
    features = FeatureFlags.from_config(config)
    layer_types = _trinity_layer_types()

    router = LayerRouter(layer_types, 60, features)

    evict_count = sum(1 for i in range(60) if router.get_strategy(i) == LayerStrategy.COMPRESS_EVICT)
    assert evict_count == 15


def test_crosslayer_lite():
    """With crosslayer lite, should have 2 anchors and 13 reconstruct layers."""
    config = TrinityTurboConfig(enabled=True, crosslayer_mode="lite")
    features = FeatureFlags.from_config(config)
    layer_types = _trinity_layer_types()

    router = LayerRouter(layer_types, 60, features)

    anchor_count = sum(1 for i in range(60) if router.get_strategy(i) == LayerStrategy.ANCHOR)
    reconstruct_count = sum(1 for i in range(60) if router.get_strategy(i) == LayerStrategy.RECONSTRUCT)

    assert anchor_count == 2
    assert reconstruct_count == 13
    assert router.anchor_layers[0] >= 0
    assert router.anchor_layers[1] >= 0
    assert router.anchor_layers[0] != router.anchor_layers[1]


def test_passthrough_layers_are_sliding():
    """Every passthrough layer should be a sliding_attention layer."""
    config = TrinityTurboConfig(enabled=True)
    features = FeatureFlags.from_config(config)
    layer_types = _trinity_layer_types()

    router = LayerRouter(layer_types, 60, features)

    for i in range(60):
        if router.is_passthrough(i):
            assert layer_types[i] == "sliding_attention"


def test_compressed_layer_count():
    config = TrinityTurboConfig(enabled=True)
    features = FeatureFlags.from_config(config)
    layer_types = _trinity_layer_types()

    router = LayerRouter(layer_types, 60, features)
    assert router.compressed_layer_count == 15
    assert len(router.global_layer_indices) == 15
