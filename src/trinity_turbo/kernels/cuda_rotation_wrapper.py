"""Python wrapper for CUDA native WHT rotation.

CUDA graph safe: all intermediate buffers are pre-allocated.
"""
from __future__ import annotations
import os
import torch

_module = None

# Pre-allocated buffers for CUDA graph compatibility
_MAX_VECS = 8192 * 32  # max vectors (tokens × heads) for graph capture
_rot_input_buf: torch.Tensor | None = None
_rot_output_buf: torch.Tensor | None = None
_normal_dim: int = 0


def _get_module():
    global _module
    if _module is not None:
        return _module
    from torch.utils.cpp_extension import load
    import filelock
    src = os.path.join(os.path.dirname(__file__), "cuda_rotation.cu")
    lock = os.path.join(os.path.dirname(__file__), ".cuda_rot_compile.lock")
    with filelock.FileLock(lock, timeout=300):
        _module = load(name="cuda_rotation", sources=[src],
                       extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
    return _module


def _ensure_bufs(dim: int, device: torch.device) -> None:
    """Pre-allocate rotation buffers once."""
    global _rot_input_buf, _rot_output_buf, _normal_dim
    if (_rot_input_buf is None or _rot_input_buf.device != device
            or _normal_dim != dim):
        _rot_input_buf = torch.empty(_MAX_VECS, dim, dtype=torch.float32, device=device)
        _rot_output_buf = torch.empty(_MAX_VECS, dim, dtype=torch.float32, device=device)
        _normal_dim = dim


def cuda_apply_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Forward WHT rotation using pre-allocated buffers (CUDA graph safe)."""
    mod = _get_module()
    orig_shape = x.shape
    dim = x.shape[-1]
    _ensure_bufs(dim, x.device)

    flat = x.reshape(-1, dim)
    n = flat.shape[0]

    # Copy to pre-allocated input buffer (in-place, no allocation)
    inp = _rot_input_buf[:n]
    inp.copy_(flat.float() if flat.dtype != torch.float32 else flat)

    out = _rot_output_buf[:n]
    mod.cuda_apply_rotation_inplace(inp, out, sign_flips, False)

    return out.reshape(orig_shape)


def cuda_apply_inverse_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    """Inverse WHT rotation using pre-allocated buffers (CUDA graph safe)."""
    mod = _get_module()
    orig_shape = x.shape
    dim = x.shape[-1]
    _ensure_bufs(dim, x.device)

    flat = x.reshape(-1, dim)
    n = flat.shape[0]

    inp = _rot_input_buf[:n]
    inp.copy_(flat.float() if flat.dtype != torch.float32 else flat)

    out = _rot_output_buf[:n]
    mod.cuda_apply_rotation_inplace(inp, out, sign_flips, True)

    return out.reshape(orig_shape)
