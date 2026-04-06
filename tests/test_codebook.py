"""Tests for Lloyd-Max codebook generation."""

import torch

from trinity_turbo.quant.codebook import compute_lloyd_max_codebook, get_codebook_tensors


def test_codebook_num_levels():
    for bits in (2, 3, 4):
        num_levels = 1 << bits
        boundaries, centroids = compute_lloyd_max_codebook(num_levels)
        assert len(boundaries) == num_levels - 1
        assert len(centroids) == num_levels


def test_codebook_sorted():
    """Boundaries and centroids must be monotonically increasing."""
    for bits in (2, 3, 4):
        num_levels = 1 << bits
        boundaries, centroids = compute_lloyd_max_codebook(num_levels)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i + 1]
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1]


def test_codebook_symmetry():
    """Codebook for Gaussian N(0,1) should be approximately symmetric."""
    for bits in (2, 3, 4):
        num_levels = 1 << bits
        _, centroids = compute_lloyd_max_codebook(num_levels)
        # Sum of centroids should be approximately 0
        assert abs(sum(centroids)) < 0.01, f"Centroids not symmetric for {bits}-bit"


def test_codebook_caching():
    """Calling twice should return the same object (LRU cached)."""
    a = compute_lloyd_max_codebook(8)
    b = compute_lloyd_max_codebook(8)
    assert a is b


def test_get_codebook_tensors(device):
    boundaries, centroids = get_codebook_tensors(3, device)
    assert boundaries.shape == (7,)  # 2^3 - 1
    assert centroids.shape == (8,)  # 2^3
    assert boundaries.dtype == torch.float32
    assert centroids.device == device
