"""Triton fused post-WHT compress: quantize + 4-bit pack + slot assembly.

Fuses the post-rotation steps into a single Triton kernel:
  Input: outlier channels (bf16) + rotated normal channels (float32) + norm
  Output: packed uint8 slot (80 bytes)

WHT rotation is done in PyTorch BEFORE calling this kernel.
This eliminates the PyTorch chain: scale → bucketize → pack_indices → slot assembly
(~4 separate CUDA kernels per call → 1 Triton kernel).

60 layers × 1 kernel = 60 kernels per decode step (was ~240+).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET,
    PACKED_BYTES,
    PACKED_OFFSET,
    SLOT_BYTES,
)
from trinity_turbo.quant.rotation import apply_rotation


@triton.jit
def _fused_post_wht_compress_kernel(
    # Input: original vector (for outliers) and pre-rotated normals
    x_ptr,             # [..., head_dim] bf16 — original input
    rotated_ptr,       # [..., normal_dim] float32 — WHT-rotated normals
    # Output
    slot_ptr,          # [..., SLOT_BYTES] uint8
    # Quantization
    boundaries_ptr,    # [num_boundaries] float32
    sqrt_d,            # float32: sqrt(normal_dim)
    # Strides
    stride_x: tl.int64,
    stride_rot: tl.int64,
    stride_slot: tl.int64,
    # Constexprs
    HEAD_DIM: tl.constexpr,
    NUM_OUTLIERS: tl.constexpr,
    NORMAL_DIM: tl.constexpr,
    NORMAL_PAD: tl.constexpr,       # next power of 2 of NORMAL_DIM
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
    # 2. Load rotated normals, compute norm, scale, quantize
    # ================================================================
    nd = tl.arange(0, NORMAL_PAD)
    nd_valid = nd < NORMAL_DIM

    # Load original (pre-rotation) normal channels for norm computation
    orig_normal = tl.load(
        x_ptr + x_base + NUM_OUTLIERS + nd,
        mask=nd_valid & ((NUM_OUTLIERS + nd) < HEAD_DIM), other=0.0,
    ).to(tl.float32)

    # L2 norm
    norm_sq = tl.sum(orig_normal * orig_normal, axis=0)
    norm_val = tl.sqrt(norm_sq + 1e-16)

    # Store norm as fp16
    norm_fp16 = norm_val.to(tl.float16)
    norm_u16 = norm_fp16.to(tl.uint16, bitcast=True)
    tl.store(slot_ptr + slot_base + NORM_OFF, (norm_u16 & 0xFF).to(tl.uint8))
    tl.store(slot_ptr + slot_base + NORM_OFF + 1, ((norm_u16 >> 8) & 0xFF).to(tl.uint8))

    # Load already-rotated normals
    rotated = tl.load(
        rotated_ptr + rot_base + nd,
        mask=nd_valid, other=0.0,
    )

    # Scale: rotated values are WHT(normalized * signs) / sqrt(d)
    # Need to undo normalization cancellation: rotated = WHT(x/norm * signs)/sqrt(d)
    # For quantization, scale to N(0,1): multiply by sqrt(normal_dim)
    scaled = rotated * sqrt_d

    # ================================================================
    # 3. Bucketize: 4-bit quantization (15 boundaries → 16 levels)
    # ================================================================
    idx = tl.zeros([NORMAL_PAD], dtype=tl.int32)
    for b_i in range(NUM_BOUNDARIES):
        boundary = tl.load(boundaries_ptr + b_i)
        idx = tl.where(nd_valid & (scaled > boundary), b_i + 1, idx)

    # ================================================================
    # 4. 4-bit pack: pairs → bytes
    # ================================================================
    # Even indices → hi nibble, odd indices → lo nibble
    byte_offs = tl.arange(0, NORMAL_PAD // 2)
    byte_valid = byte_offs < ((NORMAL_DIM + 1) // 2)

    # Gather even/odd indices from the idx array
    # idx is a register vector — use even/odd masking
    even_nd = 2 * byte_offs
    odd_nd = 2 * byte_offs + 1

    # We need idx[even_nd] and idx[odd_nd].
    # Since idx is a register array, we can't directly index.
    # Workaround: store idx to rotated_ptr (already consumed), read back.
    tl.store(rotated_ptr + rot_base + nd, idx.to(tl.float32), mask=nd_valid)

    idx_even = tl.load(
        rotated_ptr + rot_base + even_nd,
        mask=byte_valid, other=0,
    ).to(tl.int32)
    idx_odd = tl.load(
        rotated_ptr + rot_base + odd_nd,
        mask=byte_valid & (odd_nd < NORMAL_DIM), other=0,
    ).to(tl.int32)

    packed = ((idx_even << 4) | (idx_odd & 0x0F)).to(tl.uint8)
    tl.store(slot_ptr + slot_base + PACKED_OFF + byte_offs, packed, mask=byte_valid)

    # Zero padding
    pad_off = tl.arange(0, 2)
    tl.store(slot_ptr + slot_base + NORM_OFF + 2 + pad_off,
             tl.zeros([2], dtype=tl.uint8))


def fused_compress_to_slot(
    x: torch.Tensor,
    state,
) -> torch.Tensor:
    """Fused compress: WHT (PyTorch) + quantize+pack (Triton).

    Steps:
      1. Split outlier/normal
      2. Normalize normal channels
      3. WHT rotation (PyTorch apply_rotation — correctness proven)
      4. Triton fused: outlier store + norm store + scale + quantize + pack
    """
    *batch_shape, hd = x.shape
    assert hd == state.head_dim
    device = x.device

    flat = x.reshape(-1, hd).contiguous()
    n = flat.shape[0]

    # Step 1-3: split, normalize, rotate (PyTorch)
    normal = flat[:, state.num_outliers:].float()
    norms = torch.norm(normal, dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = normal / norms
    rotated = apply_rotation(normalized, state.sign_flips)

    # Step 4: fused Triton kernel
    slot = torch.zeros(n, SLOT_BYTES, dtype=torch.uint8, device=device)
    normal_pad = state.sign_flips.shape[0]  # 128

    # Pad rotated to NORMAL_PAD for Triton
    if rotated.shape[-1] < normal_pad:
        rotated_padded = torch.nn.functional.pad(rotated, (0, normal_pad - rotated.shape[-1]))
    else:
        rotated_padded = rotated
    rotated_padded = rotated_padded.contiguous()

    _fused_post_wht_compress_kernel[(n,)](
        flat, rotated_padded, slot,
        state.boundaries,
        math.sqrt(state.normal_dim),
        flat.stride(0), rotated_padded.stride(0), slot.stride(0),
        HEAD_DIM=state.head_dim,
        NUM_OUTLIERS=state.num_outliers,
        NORMAL_DIM=state.normal_dim,
        NORMAL_PAD=normal_pad,
        NUM_BOUNDARIES=len(state.boundaries),
        PACKED_OFF=PACKED_OFFSET,
        NORM_OFF=NORM_OFFSET,
    )

    return slot.reshape(*batch_shape, SLOT_BYTES)
