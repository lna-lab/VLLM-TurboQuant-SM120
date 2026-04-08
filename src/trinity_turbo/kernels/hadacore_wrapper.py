"""HadaCore Tensor Core WHT wrapper — Phase 4b optimized.

Stripped all conversion overhead:
- sign_flips pre-converted to bf16 at init
- inplace=True for zero-copy WHT
- No zero_(), no intermediate float() conversions
- Pre-allocated padded buffer reused across calls
"""

from __future__ import annotations

import math
import os

import torch

_module = None

# Pre-allocated buffers (CUDA graph safe)
_MAX_VECS = 8192 * 32
_rot_buf: torch.Tensor | None = None
_signs_bf16: torch.Tensor | None = None
_inv_sqrt_padded: float = 0.0
_normal_dim: int = 0
_padded_dim: int = 0


def _get_module():
    global _module
    if _module is not None:
        return _module

    from torch.utils.cpp_extension import load
    import filelock

    src_dir = os.path.dirname(__file__)
    cu_src = os.path.join(src_dir, "hadamard_transform_cuda.cu")
    cpp_src = os.path.join(src_dir, "hadamard_transform.cpp")
    lock_path = os.path.join(src_dir, ".hadacore_compile.lock")

    with filelock.FileLock(lock_path, timeout=300):
        _module = load(
            name="faster_hadamard_transform",
            sources=[cpp_src, cu_src],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math",
                "-gencode", "arch=compute_120,code=sm_120",
            ],
            verbose=False,
        )

    return _module


def _ensure_bufs(dim: int, padded_dim: int, sign_flips: torch.Tensor, device: torch.device) -> None:
    """One-time init: pre-allocate buffer + pre-convert signs to bf16."""
    global _rot_buf, _signs_bf16, _inv_sqrt_padded, _normal_dim, _padded_dim
    if _rot_buf is None or _rot_buf.device != device or _normal_dim != dim:
        _rot_buf = torch.zeros(_MAX_VECS, padded_dim, dtype=torch.bfloat16, device=device)
        # Pre-convert sign_flips to bf16 (never changes after init)
        full_signs = torch.ones(padded_dim, dtype=torch.bfloat16, device=device)
        full_signs[:dim] = sign_flips[:dim].to(torch.bfloat16)
        _signs_bf16 = full_signs
        _inv_sqrt_padded = 1.0 / math.sqrt(padded_dim)
        _normal_dim = dim
        _padded_dim = padded_dim


def hadacore_apply_rotation(
    x: torch.Tensor,
    sign_flips: torch.Tensor,
) -> torch.Tensor:
    """Forward WHT: y = WHT(diag(signs) * x) / sqrt(d). Zero-overhead wrapper."""
    mod = _get_module()
    orig_shape = x.shape
    dim = x.shape[-1]
    padded_dim = sign_flips.shape[0]

    _ensure_bufs(dim, padded_dim, sign_flips, x.device)

    flat = x.reshape(-1, dim)
    N = flat.shape[0]

    # Direct bf16 write with signs — no zero_(), no intermediate copies
    buf = _rot_buf[:N]
    buf[:, dim:] = 0  # Only zero the padding region (8 elements, not 128)
    buf[:, :dim] = flat.to(torch.bfloat16) * _signs_bf16[:dim]

    # HadaCore WHT inplace — zero allocation
    mod.hadamard_transform(buf, True)

    # Extract, normalize, return as float32
    return (buf[:, :dim].float() * _inv_sqrt_padded).reshape(orig_shape)


def hadacore_apply_inverse_rotation(
    x: torch.Tensor,
    sign_flips: torch.Tensor,
) -> torch.Tensor:
    """Inverse WHT: y = diag(signs) * WHT(x) / sqrt(d). Zero-overhead wrapper."""
    mod = _get_module()
    orig_shape = x.shape
    dim = x.shape[-1]
    padded_dim = sign_flips.shape[0]

    _ensure_bufs(dim, padded_dim, sign_flips, x.device)

    flat = x.reshape(-1, dim)
    N = flat.shape[0]

    buf = _rot_buf[:N]
    buf[:, dim:] = 0
    buf[:, :dim] = flat.to(torch.bfloat16)

    mod.hadamard_transform(buf, True)

    # Normalize + apply signs
    return (buf[:, :dim].float() * _inv_sqrt_padded * sign_flips[:dim]).reshape(orig_shape)
