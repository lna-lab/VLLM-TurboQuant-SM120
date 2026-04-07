"""Triton-accelerated KV cache decompression.

Decompresses packed uint8 slots back to bf16 KV vectors.
Normal channels remain in WHT-rotated space — the caller must
pre-rotate Q by the same Walsh-Hadamard transform before attention.

Hot path: called for ALL cached tokens every decode step.
The Triton kernel fuses: 3-bit unpack -> centroid lookup -> scale -> bf16 output.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from trinity_turbo.quant.turboquant import CompressedKV, QuantState, decompress

SLOT_BYTES = 64
OUTLIER_BYTES = 16
PACKED_OFFSET = 16
PACKED_BYTES_3BIT = 45
NORM_OFFSET = 61


# ---------------------------------------------------------------------------
# Triton kernel: fused 3-bit unpack + centroid dequant + norm scale
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_dequant_3bit_kernel(
    packed_ptr,       # [NH, 45] uint8 — packed 3-bit indices
    centroids_ptr,    # [8] float32 — Lloyd-Max centroids
    norm_ptr,         # [NH] float16 — per-vector L2 norms
    output_ptr,       # [NH, BLOCK_D] bfloat16 — dequantized normals
    inv_sqrt_d,       # float — 1.0 / sqrt(normal_dim)
    NH,               # int — total (token, head) pairs
    NORMAL_DIM: tl.constexpr,    # 120
    BLOCK_D: tl.constexpr,       # 128 (next power-of-2 pad)
    PACKED_BYTES: tl.constexpr,  # 45
):
    """Fused unpack + dequant for one (token, head) pair.

    Each program processes one slot:
      1. Unpack 45 bytes -> 120 three-bit indices
      2. Gather centroids from the 8-entry codebook
      3. Undo the pre-quantization sqrt(d) scaling
      4. Multiply by the stored L2 norm
      5. Store as bf16
    """
    pid = tl.program_id(0)
    if pid >= NH:
        return

    d_idx = tl.arange(0, BLOCK_D)
    valid = d_idx < NORMAL_DIM

    # ── 3-bit unpack ─────────────────────────────────────────
    # Layout: 15 groups of 8 values packed into 3 bytes each.
    #   p=0: (b0>>5)&7   p=1: (b0>>2)&7   p=2: ((b0&3)<<1)|(b1>>7)
    #   p=3: (b1>>4)&7   p=4: (b1>>1)&7   p=5: ((b1&1)<<2)|(b2>>6)
    #   p=6: (b2>>3)&7   p=7: b2&7

    g = d_idx // 8   # group  0..14
    p = d_idx % 8    # position 0..7

    # Which of the 3 bytes in the group holds the primary bits?
    byte_in_group = tl.where(p < 3, 0, tl.where(p < 6, 1, 2))
    primary_off = g * 3 + byte_in_group

    packed_base = pid * PACKED_BYTES
    primary = tl.load(
        packed_ptr + packed_base + primary_off,
        mask=valid, other=0,
    ).to(tl.int32)

    # Right-shift for the 6 non-spanning positions
    shift = tl.where(p == 0, 5,
            tl.where(p == 1, 2,
            tl.where(p == 3, 4,
            tl.where(p == 4, 1,
            tl.where(p == 6, 3,
            tl.where(p == 7, 0,
                     0))))))          # p=2,5 overridden below
    simple_val = (primary >> shift) & 7

    # Two positions span a byte boundary (p=2 and p=5)
    is_span = (p == 2) | (p == 5)
    secondary = tl.load(
        packed_ptr + packed_base + primary_off + 1,
        mask=valid & is_span, other=0,
    ).to(tl.int32)

    hi_mask  = tl.where(p == 2, 3, 1)   # keep from primary byte
    hi_shift = tl.where(p == 2, 1, 2)   # <<
    lo_shift = tl.where(p == 2, 7, 6)   # >> for secondary byte
    span_val = ((primary & hi_mask) << hi_shift) | (secondary >> lo_shift)

    indices = tl.where(is_span, span_val, simple_val)

    # ── Centroid lookup ──────────────────────────────────────
    values = tl.load(centroids_ptr + indices, mask=valid, other=0.0)

    # ── Scale: undo sqrt(d) then apply stored norm ───────────
    scaled = values * inv_sqrt_d
    norm_val = tl.load(norm_ptr + pid).to(tl.float32)
    result = scaled * norm_val

    # ── Store bf16 ───────────────────────────────────────────
    tl.store(
        output_ptr + pid * BLOCK_D + d_idx,
        result.to(tl.bfloat16),
        mask=valid,
    )


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------

def decompress_from_slot(
    slot: torch.Tensor,
    state: QuantState,
) -> torch.Tensor:
    """Decompress packed uint8 slots to bf16 KV vectors.

    Normal channels are in WHT-rotated space — Q must be pre-rotated
    by ``apply_rotation(q_normal, state.sign_flips)`` before attention.

    Dispatches to Triton on CUDA, PyTorch fallback on CPU.

    Args:
        slot: Packed slots, shape (..., SLOT_BYTES), dtype uint8.
        state: QuantState for this layer.

    Returns:
        KV vectors, shape (..., head_dim), dtype bfloat16.
    """
    if not slot.is_cuda:
        return _decompress_pytorch(slot, state)

    *batch_shape, s = slot.shape
    assert s == SLOT_BYTES, f"slot size {s} != {SLOT_BYTES}"

    device = slot.device
    nh = max(1, math.prod(batch_shape))
    flat_slot = slot.reshape(nh, SLOT_BYTES)

    # ── Parse slot regions ────────────────────────────────────
    outlier_bf16 = (
        flat_slot[:, :OUTLIER_BYTES]
        .contiguous()
        .view(torch.bfloat16)           # [nh, 8]
    )
    packed = (
        flat_slot[:, PACKED_OFFSET:PACKED_OFFSET + PACKED_BYTES_3BIT]
        .contiguous()                    # [nh, 45] uint8
    )
    norm_fp16 = (
        flat_slot[:, NORM_OFFSET:NORM_OFFSET + 2]
        .contiguous()
        .view(torch.float16)            # [nh, 1]
        .squeeze(-1)                     # [nh]
    )

    # ── Triton: unpack + dequant ──────────────────────────────
    block_d = 128  # next-pow2 of normal_dim=120
    normal_buf = torch.zeros(nh, block_d, dtype=torch.bfloat16, device=device)

    _unpack_dequant_3bit_kernel[(nh,)](
        packed,
        state.centroids,
        norm_fp16,
        normal_buf,
        1.0 / math.sqrt(state.normal_dim),
        nh,
        NORMAL_DIM=state.normal_dim,
        BLOCK_D=block_d,
        PACKED_BYTES=PACKED_BYTES_3BIT,
    )

    # ── Reassemble full head_dim vector ───────────────────────
    output = torch.empty(nh, state.head_dim, dtype=torch.bfloat16, device=device)
    # Current layout: outliers = first num_outliers channels,
    # normals = remaining channels.  (TODO: calibration-based outlier selection)
    output[:, :state.num_outliers] = outlier_bf16
    output[:, state.num_outliers:] = normal_buf[:, :state.normal_dim]

    return output.reshape(*batch_shape, state.head_dim)


def _decompress_pytorch(
    slot: torch.Tensor,
    state: QuantState,
) -> torch.Tensor:
    """Pure-PyTorch fallback (CPU or debugging)."""
    *batch_shape, s = slot.shape
    nh = max(1, math.prod(batch_shape))
    flat_slot = slot.reshape(nh, SLOT_BYTES)

    outlier_bf16 = flat_slot[:, :OUTLIER_BYTES].contiguous().view(torch.bfloat16)
    packed = flat_slot[:, PACKED_OFFSET:PACKED_OFFSET + PACKED_BYTES_3BIT].contiguous()
    norm_fp16 = flat_slot[:, NORM_OFFSET:NORM_OFFSET + 2].contiguous().view(torch.float16)

    compressed = CompressedKV(
        packed_indices=packed,
        norms=norm_fp16,
        outliers=outlier_bf16,
    )
    result = decompress(compressed, state)
    return result.reshape(*batch_shape, state.head_dim)
