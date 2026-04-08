"""HadaCore Tensor Core WHT wrapper for TurboQuant Phase 4b.

Meta's HadaCore: CUDA native Tensor Core accelerated Hadamard Transform.
0.003ms for 480 vectors (78x faster than our CUDA native butterfly).
CUDA graph compatible (native CUDA, no Triton SM120 issues).

arXiv: 2412.08832
Source: pytorch-labs/applied-ai

API: hadamard_transform(x, inplace=False) → WHT of x along last dim.
Input must be bf16 or fp16, last dim must be power of 2.
"""

from __future__ import annotations

import math
import os

import torch

_module = None

# Pre-allocated buffers for CUDA graph safety
_MAX_VECS = 8192 * 32
_rot_input_buf: torch.Tensor | None = None
_rot_output_buf: torch.Tensor | None = None
_signs_buf: torch.Tensor | None = None
_normal_dim: int = 0


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


def _ensure_bufs(dim: int, padded_dim: int, device: torch.device) -> None:
    global _rot_input_buf, _rot_output_buf, _signs_buf, _normal_dim
    if (_rot_input_buf is None or _rot_input_buf.device != device
            or _normal_dim != dim):
        _rot_input_buf = torch.empty(_MAX_VECS, padded_dim, dtype=torch.bfloat16, device=device)
        _rot_output_buf = torch.empty(_MAX_VECS, padded_dim, dtype=torch.bfloat16, device=device)
        _normal_dim = dim


def hadacore_apply_rotation(
    x: torch.Tensor,
    sign_flips: torch.Tensor,
) -> torch.Tensor:
    """Apply forward WHT rotation: y = WHT(diag(signs) * x) / sqrt(d).

    Uses HadaCore Tensor Core kernel. CUDA graph safe.
    """
    mod = _get_module()
    orig_shape = x.shape
    dim = x.shape[-1]
    padded_dim = sign_flips.shape[0]

    _ensure_bufs(dim, padded_dim, x.device)

    flat = x.reshape(-1, dim)
    N = flat.shape[0]

    # Copy to pre-allocated buffer, pad, apply signs, convert to bf16
    inp = _rot_input_buf[:N]
    inp.zero_()
    inp[:, :dim] = flat.to(torch.bfloat16) * sign_flips[:dim].to(torch.bfloat16)

    # HadaCore WHT (in-place capable, bf16)
    out = mod.hadamard_transform(inp, False)

    # Normalize by 1/sqrt(padded_dim) and truncate to dim
    out = out[:, :dim] * (1.0 / math.sqrt(padded_dim))

    return out.float().reshape(orig_shape)


def hadacore_apply_inverse_rotation(
    x: torch.Tensor,
    sign_flips: torch.Tensor,
) -> torch.Tensor:
    """Apply inverse WHT rotation: y = diag(signs) * WHT(x) / sqrt(d).

    Inverse: first WHT, then multiply by signs.
    """
    mod = _get_module()
    orig_shape = x.shape
    dim = x.shape[-1]
    padded_dim = sign_flips.shape[0]

    _ensure_bufs(dim, padded_dim, x.device)

    flat = x.reshape(-1, dim)
    N = flat.shape[0]

    inp = _rot_input_buf[:N]
    inp.zero_()
    inp[:, :dim] = flat.to(torch.bfloat16)

    out = mod.hadamard_transform(inp, False)

    # Normalize, apply signs, truncate
    out_f32 = out[:, :dim].float() * (1.0 / math.sqrt(padded_dim))
    out_f32 = out_f32 * sign_flips[:dim]

    return out_f32.reshape(orig_shape)
