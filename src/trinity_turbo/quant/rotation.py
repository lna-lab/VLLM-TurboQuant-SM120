"""Random rotation via Walsh-Hadamard transform with random sign flips.

TurboQuant applies a random orthogonal rotation to KV vectors before quantization.
This concentrates coordinate distributions toward Gaussian, enabling optimal
scalar quantization per coordinate.

We use the Walsh-Hadamard Transform (WHT) with random sign flips instead of a
full random orthogonal matrix. WHT is O(d log d) vs O(d^2) for full QR, and
achieves equivalent distributional properties for quantization purposes.

Key optimization from TurboQuant paper: We do NOT inverse-rotate cached KV.
Instead, Q is pre-rotated by the same transform before attention. This saves
O(cache_len * d) work per attention call.

Reference: TurboQuant (ICLR 2026), Section 3.1.
"""

from __future__ import annotations

import torch


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def generate_sign_flips(dim: int, seed: int = 42, device: torch.device | str | None = None) -> torch.Tensor:
    """Generate deterministic random sign flips for WHT.

    Generates sign flips for the next power-of-2 dimension >= dim.
    The apply_rotation function handles padding/truncation.

    Args:
        dim: Vector dimension.
        seed: Random seed for reproducibility.
        device: Target device.

    Returns:
        Sign flip tensor of shape (padded_dim,) with values +1/-1 as float32,
        where padded_dim = next_power_of_2(dim).
    """
    padded_dim = _next_power_of_2(dim)
    # Always generate on CPU (deterministic), then move to target device.
    # Avoids "Expected 'cuda' device type for generator" under vLLM CUDA context.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    signs = torch.randint(0, 2, (padded_dim,), generator=gen, dtype=torch.float32, device="cpu")
    signs = signs * 2.0 - 1.0  # {0, 1} -> {-1, +1}
    if device is not None:
        signs = signs.to(device)
    return signs


def apply_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Apply Walsh-Hadamard rotation with sign flips: f(x) = H @ diag(signs) @ x / sqrt(d).

    Fast O(d log d) rotation that approximates random orthogonal transformation.
    Automatically pads to next power of 2 and truncates back.

    Args:
        x: Input tensor of shape (..., dim).
        sign_flips: Sign flip tensor of shape (padded_dim,).

    Returns:
        Rotated tensor of same shape as input.
    """
    *batch_shape, dim = x.shape
    padded_dim = sign_flips.shape[0]

    if dim < padded_dim:
        y = torch.nn.functional.pad(x, (0, padded_dim - dim))
    else:
        y = x

    # f(x) = H @ diag(signs) @ x / sqrt(d)
    y = y * sign_flips
    y = _fast_walsh_hadamard(y)

    if dim < padded_dim:
        y = y[..., :dim]

    return y


def apply_inverse_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Apply inverse rotation: f^{-1}(y) = diag(signs) @ H @ y / sqrt(d).

    Since H is symmetric and H/sqrt(d) is orthogonal,
    and diag(signs) is its own inverse:
    f^{-1} = diag(signs)^{-1} @ (H/sqrt(d))^{-1} = diag(signs) @ H / sqrt(d).

    Args:
        x: Rotated tensor of shape (..., dim).
        sign_flips: Same sign flip tensor used in apply_rotation.

    Returns:
        Original-space tensor of same shape.
    """
    *batch_shape, dim = x.shape
    padded_dim = sign_flips.shape[0]

    if dim < padded_dim:
        y = torch.nn.functional.pad(x, (0, padded_dim - dim))
    else:
        y = x

    # f^{-1}(y) = diag(signs) @ H @ y / sqrt(d)
    y = _fast_walsh_hadamard(y)
    y = y * sign_flips

    if dim < padded_dim:
        y = y[..., :dim]

    return y


def _fast_walsh_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform on the last dimension.

    O(d log d) butterfly operations. Normalizes by 1/sqrt(d).

    Args:
        x: Input tensor, last dimension must be power of 2.

    Returns:
        WHT of x along last dimension.
    """
    *batch_shape, d = x.shape
    assert d > 0 and (d & (d - 1)) == 0, f"WHT requires power-of-2 dimension, got {d}"

    x = x.contiguous().clone()
    h = 1
    while h < d:
        # Reshape for butterfly: groups of 2h elements
        x = x.view(*batch_shape, -1, 2 * h)
        left = x[..., :h].clone()
        right = x[..., h:].clone()
        x[..., :h] = left + right
        x[..., h:] = left - right
        h *= 2

    x = x.view(*batch_shape, d)

    # Normalize
    x = x / (d ** 0.5)
    return x
