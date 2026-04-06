"""Tests for TrinityTurboConfig."""

import os

import pytest

from trinity_turbo.config import TrinityTurboConfig


def test_default_config():
    config = TrinityTurboConfig()
    assert config.enabled is True
    assert config.bits == 3
    assert config.num_outlier_channels == 8
    assert config.eviction_enabled is False
    assert config.crosslayer_mode == "off"


def test_from_env(monkeypatch):
    monkeypatch.setenv("TRINITY_TURBO_ENABLED", "1")
    monkeypatch.setenv("TRINITY_TURBO_BITS", "4")
    monkeypatch.setenv("TRINITY_TURBO_EVICTION_ENABLED", "true")
    monkeypatch.setenv("TRINITY_TURBO_EVICTION_KEEP_RATIO", "0.25")
    monkeypatch.setenv("TRINITY_TURBO_CROSSLAYER_MODE", "lite")

    config = TrinityTurboConfig.from_env()
    assert config.enabled is True
    assert config.bits == 4
    assert config.eviction_enabled is True
    assert config.eviction_keep_ratio == 0.25
    assert config.crosslayer_mode == "lite"


def test_validate_invalid_bits():
    config = TrinityTurboConfig(bits=5)
    with pytest.raises(ValueError, match="bits must be 2, 3, or 4"):
        config.validate()


def test_validate_invalid_crosslayer():
    config = TrinityTurboConfig(crosslayer_mode="invalid")
    with pytest.raises(ValueError, match="crosslayer_mode"):
        config.validate()


def test_validate_invalid_keep_ratio():
    config = TrinityTurboConfig(eviction_keep_ratio=0.0)
    with pytest.raises(ValueError, match="eviction_keep_ratio"):
        config.validate()
