"""Allocation-free Walsh-Hadamard Transform with pre-allocated buffer.

Standard PyTorch WHT: 21 allocations per call (3 clone × 7 steps).
This version: 0 allocations during execution (buffer allocated once at init).

In CUDA graph replay, each allocation = 1 cudaMalloc that was fixed at capture.
Zero-alloc means the graph has fewer nodes → faster replay.
"""

from __future__ import annotations

import torch

# Global pre-allocated buffer for WHT temp storage
_wht_buf: torch.Tensor | None = None


def _ensure_buf(shape, device, dtype):
    global _wht_buf
    numel = 1
    for s in shape:
        numel *= s
    if _wht_buf is None or _wht_buf.numel() < numel or _wht_buf.device != device:
        _wht_buf = torch.empty(numel, dtype=dtype, device=device)
    return _wht_buf[:numel].view(shape)


def fast_walsh_hadamard_inplace(x: torch.Tensor) -> torch.Tensor:
    """WHT with 1 pre-allocated temp buffer (zero per-call allocations).

    Uses double-buffer technique: x and buf alternate as source/dest.
    """
    *batch, d = x.shape
    assert d > 0 and (d & (d - 1)) == 0

    buf = _ensure_buf(x.shape, x.device, x.dtype)

    src = x
    dst = buf

    h = 1
    while h < d:
        src_view = src.view(*batch, -1, 2, h)
        dst_view = dst.view(*batch, -1, 2, h)

        left = src_view[..., 0, :]
        right = src_view[..., 1, :]

        # Write sum/diff to dst (src is read-only this step)
        torch.add(left, right, out=dst_view[..., 0, :])
        torch.sub(left, right, out=dst_view[..., 1, :])

        # Swap for next step
        src, dst = dst, src
        h *= 2

    # Result is in src. If src != x, copy back.
    if src.data_ptr() != x.data_ptr():
        x.copy_(src)

    x.div_(d ** 0.5)
    return x


def apply_rotation_fast(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Apply WHT rotation: f(x) = WHT(diag(signs) * x) / sqrt(d).

    Minimal allocations: 1 pad (if needed) + 0 per WHT step.
    """
    *batch, dim = x.shape
    padded_dim = sign_flips.shape[0]

    if dim < padded_dim:
        y = torch.nn.functional.pad(x, (0, padded_dim - dim))
    else:
        y = x.clone()

    y.mul_(sign_flips)
    fast_walsh_hadamard_inplace(y)

    if dim < padded_dim:
        return y[..., :dim]
    return y


def apply_inverse_rotation_fast(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Apply inverse WHT rotation: f^-1(x) = diag(signs) * WHT(x) / sqrt(d)."""
    *batch, dim = x.shape
    padded_dim = sign_flips.shape[0]

    if dim < padded_dim:
        y = torch.nn.functional.pad(x, (0, padded_dim - dim))
    else:
        y = x.clone()

    fast_walsh_hadamard_inplace(y)
    y.mul_(sign_flips)

    if dim < padded_dim:
        return y[..., :dim]
    return y
