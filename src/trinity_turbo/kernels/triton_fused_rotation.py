"""Triton fused WHT rotation for Q pre-rotate and output inverse-rotate.

Replaces PyTorch _fast_walsh_hadamard (7 butterfly steps × clone × add/sub)
with a single Triton kernel per (token, head) pair using XOR partner butterfly.

forward rotation:  f(x) = WHT(diag(signs) * x) / sqrt(d)
inverse rotation:  f^-1(x) = diag(signs) * WHT(x) / sqrt(d)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rotation_kernel(
    x_ptr,             # [..., normal_dim] float32 (in-place)
    scratch_ptr,       # [..., PADDED_DIM] float32
    sign_flips_ptr,    # [PADDED_DIM] float32
    stride_x: tl.int64,
    stride_scratch: tl.int64,
    NORMAL_DIM: tl.constexpr,
    PADDED_DIM: tl.constexpr,
    IS_INVERSE: tl.constexpr,       # True for inverse rotation
):
    """WHT rotation via XOR-partner butterfly in global memory."""
    pid = tl.program_id(0)
    x_base = pid * stride_x
    scratch_base = pid * stride_scratch

    nd = tl.arange(0, PADDED_DIM)
    nd_valid = nd < NORMAL_DIM

    # Load input, pad to PADDED_DIM
    val = tl.load(x_ptr + x_base + nd, mask=nd_valid, other=0.0)
    signs = tl.load(sign_flips_ptr + nd)

    # Forward: multiply signs BEFORE WHT
    if not IS_INVERSE:
        val = val * signs

    # Write to scratch for butterfly
    tl.store(scratch_ptr + scratch_base + nd, val, mask=nd < PADDED_DIM)

    # 7-step butterfly (PADDED_DIM=128)
    # Step 0
    p0 = nd ^ 1
    a0 = tl.load(scratch_ptr + scratch_base + nd)
    b0 = tl.load(scratch_ptr + scratch_base + p0, mask=p0 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 1) == 0, a0 + b0, b0 - a0))
    # Step 1
    p1 = nd ^ 2
    a1 = tl.load(scratch_ptr + scratch_base + nd)
    b1 = tl.load(scratch_ptr + scratch_base + p1, mask=p1 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 2) == 0, a1 + b1, b1 - a1))
    # Step 2
    p2 = nd ^ 4
    a2 = tl.load(scratch_ptr + scratch_base + nd)
    b2 = tl.load(scratch_ptr + scratch_base + p2, mask=p2 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 4) == 0, a2 + b2, b2 - a2))
    # Step 3
    p3 = nd ^ 8
    a3 = tl.load(scratch_ptr + scratch_base + nd)
    b3 = tl.load(scratch_ptr + scratch_base + p3, mask=p3 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 8) == 0, a3 + b3, b3 - a3))
    # Step 4
    p4 = nd ^ 16
    a4 = tl.load(scratch_ptr + scratch_base + nd)
    b4 = tl.load(scratch_ptr + scratch_base + p4, mask=p4 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 16) == 0, a4 + b4, b4 - a4))
    # Step 5
    p5 = nd ^ 32
    a5 = tl.load(scratch_ptr + scratch_base + nd)
    b5 = tl.load(scratch_ptr + scratch_base + p5, mask=p5 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 32) == 0, a5 + b5, b5 - a5))
    # Step 6
    p6 = nd ^ 64
    a6 = tl.load(scratch_ptr + scratch_base + nd)
    b6 = tl.load(scratch_ptr + scratch_base + p6, mask=p6 < PADDED_DIM)
    tl.store(scratch_ptr + scratch_base + nd,
             tl.where((nd & 64) == 0, a6 + b6, b6 - a6))

    # Read result, normalize
    result = tl.load(scratch_ptr + scratch_base + nd)
    inv_sqrt = 1.0 / tl.sqrt(tl.full([1], PADDED_DIM, dtype=tl.float32))
    result = result * inv_sqrt

    # Inverse: multiply signs AFTER WHT
    if IS_INVERSE:
        result = result * signs

    # Store back (truncated to NORMAL_DIM)
    tl.store(x_ptr + x_base + nd, result, mask=nd_valid)


# Persistent scratch
_rotation_scratch = None


def triton_apply_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Apply WHT rotation in a single Triton kernel.

    Args:
        x: (..., dim) float32 tensor (normal channels only).
        sign_flips: (padded_dim,) float32.

    Returns:
        Rotated tensor, same shape.
    """
    return _run_rotation(x, sign_flips, is_inverse=False)


def triton_apply_inverse_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Apply inverse WHT rotation in a single Triton kernel."""
    return _run_rotation(x, sign_flips, is_inverse=True)


def _run_rotation(x: torch.Tensor, sign_flips: torch.Tensor, is_inverse: bool) -> torch.Tensor:
    global _rotation_scratch
    *batch_shape, dim = x.shape
    padded_dim = sign_flips.shape[0]
    device = x.device

    flat = x.reshape(-1, dim).contiguous().float()
    n = flat.shape[0]

    if _rotation_scratch is None or _rotation_scratch.shape[0] < n:
        _rotation_scratch = torch.empty(max(n, 256), padded_dim, dtype=torch.float32, device=device)
    scratch = _rotation_scratch[:n]

    _fused_rotation_kernel[(n,)](
        flat, scratch, sign_flips,
        flat.stride(0), scratch.stride(0),
        NORMAL_DIM=dim,
        PADDED_DIM=padded_dim,
        IS_INVERSE=is_inverse,
    )

    return flat.reshape(*batch_shape, dim)
