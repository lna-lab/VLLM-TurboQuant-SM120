"""Lloyd-Max codebook generation for TurboQuant.

For high-dimensional vectors (d >= 64), the random rotation makes coordinates
approximately Gaussian N(0, 1/d). We use this to compute optimal Lloyd-Max
quantization boundaries and centroids analytically.

Reference: TurboQuant (ICLR 2026, arXiv 2504.19874), Section 3.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def compute_lloyd_max_codebook(
    num_levels: int,
    num_iterations: int = 50,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute Lloyd-Max quantizer for N(0, 1) distribution.

    Returns (boundaries, centroids) where:
      - boundaries: (num_levels - 1,) partition points
      - centroids: (num_levels,) reconstruction values

    Uses the iterative Lloyd-Max algorithm with Gaussian conditional expectations.
    Results are cached since they only depend on num_levels.
    """
    # Initialize centroids uniformly in [-3, 3]
    centroids = [
        -3.0 + (6.0 * (i + 0.5) / num_levels)
        for i in range(num_levels)
    ]

    for _ in range(num_iterations):
        # Update boundaries as midpoints of adjacent centroids
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0
            for i in range(num_levels - 1)
        ]

        # Update centroids as conditional expectations E[X | a < X < b]
        # For Gaussian: E[X | a < X < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a))
        new_centroids = []
        all_bounds = [float("-inf")] + boundaries + [float("inf")]
        for i in range(num_levels):
            a, b = all_bounds[i], all_bounds[i + 1]
            phi_a = _gaussian_pdf(a)
            phi_b = _gaussian_pdf(b)
            cdf_a = _gaussian_cdf(a)
            cdf_b = _gaussian_cdf(b)
            denom = cdf_b - cdf_a
            if denom < 1e-15:
                new_centroids.append((a + b) / 2.0 if math.isfinite(a) and math.isfinite(b) else centroids[i])
            else:
                new_centroids.append((phi_a - phi_b) / denom)
        centroids = new_centroids

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0
        for i in range(num_levels - 1)
    ]

    return tuple(boundaries), tuple(centroids)


def get_codebook_tensors(
    bits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get boundaries and centroids as GPU tensors.

    Args:
        bits: Number of quantization bits (2, 3, or 4).
        device: Target device.

    Returns:
        (boundaries, centroids) as float32 tensors on device.
        boundaries shape: (2^bits - 1,)
        centroids shape: (2^bits,)
    """
    num_levels = 1 << bits
    boundaries, centroids = compute_lloyd_max_codebook(num_levels)
    return (
        torch.tensor(boundaries, dtype=torch.float32, device=device),
        torch.tensor(centroids, dtype=torch.float32, device=device),
    )


def _gaussian_pdf(x: float) -> float:
    """Standard Gaussian PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gaussian_cdf(x: float) -> float:
    """Standard Gaussian CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
