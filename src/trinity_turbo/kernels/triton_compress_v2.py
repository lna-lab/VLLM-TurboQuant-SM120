"""Triton fused compress v2: split + norm + scale + quantize + pack → slot.

WHT rotation done in PyTorch (fast_wht.py) BEFORE this kernel.
This kernel fuses 5 PyTorch ops into 1 Triton kernel:
  outlier extract → L2 norm → scale → bucketize → 4-bit pack → slot assembly

60 layers × 2 (K+V) = 120 launches/step (was ~720 with separate PyTorch ops).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET,
    PACKED_OFFSET,
    SLOT_BYTES,
)
from trinity_turbo.kernels.fast_wht import apply_rotation_fast
from trinity_turbo.quant.packing import packed_size


@triton.jit
def _fused_post_wht_kernel(
    x_ptr,             # [..., head_dim] bf16 — original (for outliers)
    rotated_ptr,       # [..., PADDED_DIM] float32 — WHT-rotated normalized normals
    norm_ptr,          # [...] float32 — pre-computed L2 norms
    slot_ptr,          # [..., SLOT_BYTES] uint8
    boundaries_ptr,    # [num_boundaries] float32
    sqrt_d,            # float32
    stride_x: tl.int64,
    stride_rot: tl.int64,
    stride_slot: tl.int64,
    HEAD_DIM: tl.constexpr,
    NUM_OUTLIERS: tl.constexpr,
    NORMAL_DIM: tl.constexpr,
    PADDED_DIM: tl.constexpr,
    NUM_BOUNDARIES: tl.constexpr,
    PACKED_OFF: tl.constexpr,
    NORM_OFF: tl.constexpr,
):
    pid = tl.program_id(0)
    x_base = pid * stride_x
    rot_base = pid * stride_rot
    slot_base = pid * stride_slot

    # ================================================================
    # 1. Outliers → bf16 byte pairs
    # ================================================================
    out_d = tl.arange(0, NUM_OUTLIERS)
    out_val = tl.load(x_ptr + x_base + out_d).to(tl.bfloat16)
    out_u16 = out_val.to(tl.uint16, bitcast=True)
    tl.store(slot_ptr + slot_base + 2 * out_d, (out_u16 & 0xFF).to(tl.uint8))
    tl.store(slot_ptr + slot_base + 2 * out_d + 1, ((out_u16 >> 8) & 0xFF).to(tl.uint8))

    # ================================================================
    # 2. Norm → fp16 bytes
    # ================================================================
    norm_val = tl.load(norm_ptr + pid)
    norm_fp16 = norm_val.to(tl.float16)
    norm_u16 = norm_fp16.to(tl.uint16, bitcast=True)
    tl.store(slot_ptr + slot_base + NORM_OFF, (norm_u16 & 0xFF).to(tl.uint8))
    tl.store(slot_ptr + slot_base + NORM_OFF + 1, ((norm_u16 >> 8) & 0xFF).to(tl.uint8))

    # ================================================================
    # 3. Load rotated, scale, quantize
    # ================================================================
    nd = tl.arange(0, PADDED_DIM)
    nd_valid = nd < NORMAL_DIM

    rotated = tl.load(rot_base + rotated_ptr + nd, mask=nd_valid, other=0.0)
    scaled = rotated * sqrt_d

    idx = tl.zeros([PADDED_DIM], dtype=tl.int32)
    for b_i in range(NUM_BOUNDARIES):
        boundary = tl.load(boundaries_ptr + b_i)
        idx = tl.where(nd_valid & (scaled > boundary), b_i + 1, idx)

    # ================================================================
    # 4. 4-bit pack via rotated_ptr scratch (already consumed)
    # ================================================================
    tl.store(rotated_ptr + rot_base + nd, idx.to(tl.float32), mask=nd_valid)

    byte_offs = tl.arange(0, PADDED_DIM // 2)
    byte_valid = byte_offs < ((NORMAL_DIM + 1) // 2)

    idx_even = tl.load(rotated_ptr + rot_base + 2 * byte_offs, mask=byte_valid, other=0).to(tl.int32)
    idx_odd = tl.load(
        rotated_ptr + rot_base + 2 * byte_offs + 1,
        mask=byte_valid & ((2 * byte_offs + 1) < NORMAL_DIM), other=0,
    ).to(tl.int32)

    packed = ((idx_even << 4) | (idx_odd & 0x0F)).to(tl.uint8)
    tl.store(slot_ptr + slot_base + PACKED_OFF + byte_offs, packed, mask=byte_valid)

    # Padding
    pad = tl.arange(0, 2)
    tl.store(slot_ptr + slot_base + NORM_OFF + 2 + pad, tl.zeros([2], dtype=tl.uint8))


def compress_to_slot_v2(
    x: torch.Tensor,
    state,
) -> torch.Tensor:
    """Fast compress: PyTorch fast WHT + Triton fused post-WHT.

    Split into 2 phases:
      Phase A (PyTorch): split → norm → normalize → WHT rotation (fast_wht)
      Phase B (Triton):  outlier store + norm store + scale + quantize + pack
    """
    *batch_shape, hd = x.shape
    assert hd == state.head_dim
    device = x.device

    flat = x.reshape(-1, hd).contiguous()
    n = flat.shape[0]
    padded_dim = state.sign_flips.shape[0]

    # Phase A: PyTorch (fast WHT)
    normal = flat[:, state.num_outliers:].float()
    norms = torch.norm(normal, dim=-1).clamp(min=1e-8)  # (n,)
    normalized = normal / norms.unsqueeze(-1)
    rotated = apply_rotation_fast(normalized, state.sign_flips)

    # Pad to PADDED_DIM
    if rotated.shape[-1] < padded_dim:
        rotated = torch.nn.functional.pad(rotated, (0, padded_dim - rotated.shape[-1]))
    rotated = rotated.contiguous()

    # Phase B: Triton fused
    slot = torch.zeros(n, SLOT_BYTES, dtype=torch.uint8, device=device)

    _fused_post_wht_kernel[(n,)](
        flat, rotated, norms, slot,
        state.boundaries,
        math.sqrt(state.normal_dim),
        flat.stride(0), rotated.stride(0), slot.stride(0),
        HEAD_DIM=state.head_dim,
        NUM_OUTLIERS=state.num_outliers,
        NORMAL_DIM=state.normal_dim,
        PADDED_DIM=padded_dim,
        NUM_BOUNDARIES=len(state.boundaries),
        PACKED_OFF=PACKED_OFFSET,
        NORM_OFF=NORM_OFFSET,
    )

    return slot.reshape(*batch_shape, SLOT_BYTES)
