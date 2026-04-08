"""Triton-native Walsh-Hadamard rotation via matrix-vector multiply.

Phase 4: Replaces CUDA native butterfly WHT with Triton mat-vec.

Key insight: For d=128, the Hadamard matrix H is 128×128.
A matrix-vector multiply is O(d²) = 16384 FLOPs, but maps perfectly
to Triton's tl.dot() which uses Tensor Cores. The butterfly approach
is O(d log d) = 896 FLOPs but requires __syncthreads() between steps,
which Triton cannot express safely.

Matrix-vector multiply advantages:
  - Single tl.dot() call, no cross-warp sync needed
  - Tensor Core accelerated on Blackwell (SM120)
  - No CUDA JIT compilation → reboot-safe
  - Combines naturally with quantize + pack in fused kernels

The Hadamard matrix is generated once at init time and stored as a
constant tensor. With sign flips folded in, the transform becomes:

  Forward:  y = (H @ diag(signs)) @ x / sqrt(d) = H_signed @ x / sqrt(d)
  Inverse:  x = diag(signs) @ (H @ y / sqrt(d)) = signs * (H @ y / sqrt(d))

We pre-compute H_signed = H * signs[None, :] / sqrt(d) so the forward
transform is a single matrix multiply with no extra ops.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def build_hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Build normalized Hadamard matrix of size dim×dim.

    Uses Sylvester construction. dim must be power of 2.
    Result is already divided by sqrt(dim).
    """
    assert dim > 0 and (dim & (dim - 1)) == 0, f"dim must be power of 2, got {dim}"

    H = torch.ones(1, 1, dtype=torch.float32, device=device)
    while H.shape[0] < dim:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)

    H = H / math.sqrt(dim)
    return H


def build_signed_hadamard(
    sign_flips: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build pre-signed Hadamard matrices for forward and inverse transforms.

    Forward: H_fwd = H * signs[None, :] / sqrt(d)
        y = H_fwd @ x  (single matmul, sign flips folded in)

    Inverse: H_inv = diag(signs) @ H / sqrt(d)
        x = H_inv @ y  (single matmul, sign flips folded in)

    Returns (H_fwd, H_inv) as float32 tensors of shape (padded_dim, padded_dim).
    """
    dim = sign_flips.shape[0]
    H = build_hadamard_matrix(dim, device)

    # Forward: H_fwd[i, j] = H[i, j] * signs[j] / sqrt(d)
    # (H already normalized by sqrt(d))
    H_fwd = H * sign_flips[None, :]

    # Inverse: H_inv[i, j] = signs[i] * H[i, j] / sqrt(d)
    H_inv = sign_flips[:, None] * H

    return H_fwd.contiguous(), H_inv.contiguous()


# ---------------------------------------------------------------------------
# Triton kernels for WHT rotation
# ---------------------------------------------------------------------------

@triton.jit
def _rotate_matvec_kernel(
    x_ptr,           # (N, dim) float32
    out_ptr,         # (N, dim) float32
    H_ptr,           # (padded_dim, padded_dim) float32
    N,
    dim: tl.constexpr,
    padded_dim: tl.constexpr,
):
    """Apply WHT rotation via matrix-vector multiply.

    One program per vector (N vectors total).
    Uses tl.dot for Tensor Core acceleration.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load input vector
    offs = tl.arange(0, padded_dim)
    mask = offs < dim
    x = tl.load(x_ptr + pid * dim + offs, mask=mask, other=0.0)  # (padded_dim,)

    # Matrix-vector multiply: y = H @ x
    # Load H row by row and dot with x
    # For padded_dim=128, this is 128 loads of 128 elements = 16K ops
    y = tl.zeros([padded_dim], dtype=tl.float32)
    for row in range(padded_dim):
        h_row = tl.load(H_ptr + row * padded_dim + offs)  # (padded_dim,)
        dot = tl.sum(h_row * x, axis=0)  # scalar
        y += tl.where(offs == row, dot, 0.0)

    # Store output (truncated to dim)
    tl.store(out_ptr + pid * dim + offs, y, mask=mask)


@triton.jit
def _rotate_matvec_batched_kernel(
    x_ptr,           # (N, dim) float32
    out_ptr,         # (N, dim) float32
    H_ptr,           # (padded_dim, padded_dim) float32
    N,
    dim: tl.constexpr,
    padded_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,  # batch tile size
):
    """Batched WHT rotation: processes BLOCK_N vectors per program.

    More efficient than per-vector kernel for large N.
    Uses 2D tl.dot: (BLOCK_N, padded_dim) @ (padded_dim, padded_dim)
    """
    pid = tl.program_id(0)
    start_n = pid * BLOCK_N

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, padded_dim)

    # Load batch of input vectors: (BLOCK_N, padded_dim)
    n_idx = start_n + offs_n
    mask_n = n_idx < N
    mask_d = offs_d < dim
    mask_2d = mask_n[:, None] & mask_d[None, :]

    x = tl.load(
        x_ptr + n_idx[:, None] * dim + offs_d[None, :],
        mask=mask_2d,
        other=0.0,
    )  # (BLOCK_N, padded_dim)

    # Load full Hadamard matrix: (padded_dim, padded_dim)
    H = tl.load(
        H_ptr + offs_d[:, None] * padded_dim + offs_d[None, :],
    )  # (padded_dim, padded_dim)

    # Batched matrix multiply: y = x @ H^T = (BLOCK_N, padded_dim) @ (padded_dim, padded_dim)
    # Note: H_fwd is already H * signs / sqrt(d), and H is symmetric
    # So x @ H_fwd^T = x @ H_fwd (since each row of H_fwd dot x gives one output element)
    # Actually we need y[i] = sum_j H[i,j] * x[j], which is y = H @ x^T for each vector
    # In batched form: Y = X @ H^T where X is (BLOCK_N, padded_dim)
    y = tl.dot(x, tl.trans(H))  # (BLOCK_N, padded_dim)

    # Store
    tl.store(
        out_ptr + n_idx[:, None] * dim + offs_d[None, :],
        y,
        mask=mask_2d,
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

# Pre-allocated buffers (CUDA graph safe)
_MAX_VECS = 8192 * 32
_input_buf: torch.Tensor | None = None
_output_buf: torch.Tensor | None = None


def _ensure_bufs(dim: int, device: torch.device):
    global _input_buf, _output_buf
    if _input_buf is None or _input_buf.device != device or _input_buf.shape[1] != dim:
        _input_buf = torch.empty(_MAX_VECS, dim, dtype=torch.float32, device=device)
        _output_buf = torch.empty(_MAX_VECS, dim, dtype=torch.float32, device=device)


def triton_apply_rotation(
    x: torch.Tensor,
    H_fwd: torch.Tensor,
) -> torch.Tensor:
    """Apply forward WHT rotation using Triton mat-vec.

    Args:
        x: Input tensor, shape (..., dim), any dtype (converted to float32).
        H_fwd: Pre-signed Hadamard matrix from build_signed_hadamard().

    Returns:
        Rotated tensor, same shape as input, float32.
    """
    orig_shape = x.shape
    dim = x.shape[-1]
    padded_dim = H_fwd.shape[0]
    flat = x.reshape(-1, dim)
    N = flat.shape[0]

    _ensure_bufs(dim, x.device)
    inp = _input_buf[:N]
    out = _output_buf[:N]
    inp.copy_(flat.float() if flat.dtype != torch.float32 else flat)

    BLOCK_N = 4  # Process 4 vectors per program
    grid = ((N + BLOCK_N - 1) // BLOCK_N,)
    _rotate_matvec_batched_kernel[grid](
        inp, out, H_fwd,
        N, dim, padded_dim,
        BLOCK_N=BLOCK_N,
    )

    return out.reshape(orig_shape)


def triton_apply_inverse_rotation(
    x: torch.Tensor,
    H_inv: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse WHT rotation using Triton mat-vec."""
    return triton_apply_rotation(x, H_inv)
