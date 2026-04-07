"""Python wrapper for CUDA native fused compress + scatter kernel.

JIT-compiles on first import. Subsequent imports use cached .so.
"""

from __future__ import annotations

import math
import os

import torch

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module

    from torch.utils.cpp_extension import load
    import filelock

    src = os.path.join(os.path.dirname(__file__), "cuda_compress.cu")
    lock_path = os.path.join(os.path.dirname(__file__), ".cuda_compile.lock")

    with filelock.FileLock(lock_path, timeout=300):
        _module = load(
            name="cuda_compress",
            sources=[src],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )

    return _module


_slot_mapping_i32: torch.Tensor | None = None


def fused_compress_scatter(
    x: torch.Tensor,           # (N, num_kv_heads, head_dim) bf16
    kv_cache: torch.Tensor,    # (num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES) uint8
    state,                     # QuantState
    slot_mapping: torch.Tensor, # (N,) int32 or int64
    kv_dim: int,               # 0=K, 1=V
) -> None:
    """Fused compress + scatter: WHT + quantize + pack + write to cache."""
    global _slot_mapping_i32
    mod = _get_module()

    # Convert int64→int32 using pre-allocated buffer (CUDA graph safe)
    if slot_mapping.dtype != torch.int32:
        if _slot_mapping_i32 is None or _slot_mapping_i32.shape[0] < slot_mapping.shape[0]:
            _slot_mapping_i32 = torch.empty(
                max(slot_mapping.shape[0], 512), dtype=torch.int32, device=slot_mapping.device,
            )
        sm = _slot_mapping_i32[:slot_mapping.shape[0]]
        sm.copy_(slot_mapping)
    else:
        sm = slot_mapping

    mod.fused_compress_scatter(
        x if x.is_contiguous() else x.contiguous(),
        kv_cache,
        state.boundaries,
        state.sign_flips,
        sm,
        kv_dim,
        math.sqrt(state.normal_dim),
    )
