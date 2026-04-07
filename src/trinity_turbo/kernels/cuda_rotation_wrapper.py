"""Python wrapper for CUDA native WHT rotation."""
from __future__ import annotations
import os
import torch

_module = None

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

def cuda_apply_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    mod = _get_module()
    flat = x.reshape(-1, x.shape[-1]).contiguous().float()
    result = mod.cuda_apply_rotation(flat, sign_flips, False)
    return result.reshape(x.shape)

def cuda_apply_inverse_rotation(x: torch.Tensor, sign_flips: torch.Tensor) -> torch.Tensor:
    mod = _get_module()
    flat = x.reshape(-1, x.shape[-1]).contiguous().float()
    result = mod.cuda_apply_rotation(flat, sign_flips, True)
    return result.reshape(x.shape)
