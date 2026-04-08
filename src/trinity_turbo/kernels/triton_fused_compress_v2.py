"""Phase 4: Fully fused Triton compress + scatter kernel.

Replaces CUDA native cuda_compress.cu with pure Triton.
WHT rotation via matrix-vector multiply (no __syncthreads__ needed).

Single kernel: input bf16 → outlier extract → norm → WHT rotation →
               quantize → 4-bit pack → scatter to paged KV cache.

Grid: (N_tokens, num_kv_heads)
Each program processes one (token, head) pair for K or V.
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

# Pre-allocated buffers
_MAX_TOKENS = 8192
_slot_mapping_i32: torch.Tensor | None = None


def _ensure_slot_buf(device: torch.device):
    global _slot_mapping_i32
    if _slot_mapping_i32 is None or _slot_mapping_i32.device != device:
        _slot_mapping_i32 = torch.empty(_MAX_TOKENS, dtype=torch.int32, device=device)


@triton.jit
def _fused_compress_scatter_kernel(
    x_ptr,              # (N, num_kv_heads, head_dim) bf16
    cache_ptr,          # (num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES) uint8
    slot_mapping_ptr,   # (N,) int32
    H_ptr,              # (PADDED_DIM, PADDED_DIM) float32 — pre-signed Hadamard
    boundaries_ptr,     # (NUM_LEVELS-1,) float32 — Lloyd-Max boundaries
    kv_dim,             # 0=K, 1=V
    x_stride_t,
    x_stride_h,
    x_stride_d,
    cache_stride_0,     # block
    cache_stride_1,     # kv_dim (0=K, 1=V)
    cache_stride_2,     # token within block
    cache_stride_3,     # head
    cache_stride_4,     # byte
    block_size,
    NUM_OUTLIERS: tl.constexpr,     # 8
    NORMAL_DIM: tl.constexpr,       # 120
    PADDED_DIM: tl.constexpr,       # 128
    HEAD_DIM: tl.constexpr,         # 128
    NUM_BOUNDARIES: tl.constexpr,   # 15 (4-bit = 16 levels)
    PACKED_OFF: tl.constexpr,       # 16
    NORM_OFF: tl.constexpr,         # 76
    SLOT_B: tl.constexpr,           # 80
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Slot mapping → paged cache location
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    blk = slot_idx // block_size
    off = slot_idx % block_size
    cache_base = (blk * cache_stride_0 + kv_dim * cache_stride_1 +
                  off * cache_stride_2 + head_idx * cache_stride_3)
    x_base = token_idx * x_stride_t + head_idx * x_stride_h

    # ================================================================
    # 1. Store outliers as bf16 bytes (first NUM_OUTLIERS channels)
    # ================================================================
    out_offs = tl.arange(0, NUM_OUTLIERS)
    out_vals = tl.load(x_ptr + x_base + out_offs * x_stride_d).to(tl.bfloat16)
    out_u16 = out_vals.to(tl.uint16, bitcast=True)
    tl.store(cache_ptr + cache_base + (2 * out_offs) * cache_stride_4,
             (out_u16 & 0xFF).to(tl.uint8))
    tl.store(cache_ptr + cache_base + (2 * out_offs + 1) * cache_stride_4,
             ((out_u16 >> 8) & 0xFF).to(tl.uint8))

    # ================================================================
    # 2. Load normal channels, compute L2 norm
    # ================================================================
    norm_offs = tl.arange(0, PADDED_DIM)
    norm_mask = norm_offs < NORMAL_DIM
    normal = tl.load(
        x_ptr + x_base + (NUM_OUTLIERS + norm_offs) * x_stride_d,
        mask=norm_mask,
        other=0.0,
    ).to(tl.float32)

    norm_sq = tl.sum(normal * normal, axis=0)
    norm = tl.sqrt(tl.maximum(norm_sq, 1e-16))

    # Store norm as fp16 bytes (use norm_offs as 1-element block for store)
    norm_fp16 = norm.to(tl.float16)
    norm_u16 = norm_fp16.to(tl.uint16, bitcast=True)
    norm_byte_offs = tl.arange(0, 4)  # [0, 1, 2, 3]
    norm_lo = (norm_u16 & 0xFF).to(tl.uint8)
    norm_hi = ((norm_u16 >> 8) & 0xFF).to(tl.uint8)
    zero_u8 = tl.zeros([1], dtype=tl.uint8).to(tl.uint8)
    norm_bytes = tl.where(norm_byte_offs == 0, norm_lo,
                 tl.where(norm_byte_offs == 1, norm_hi, zero_u8))
    tl.store(
        cache_ptr + cache_base + (NORM_OFF + norm_byte_offs) * cache_stride_4,
        norm_bytes.to(tl.uint8),
    )

    # ================================================================
    # 3. Normalize → WHT rotation via matrix-vector multiply
    # ================================================================
    normalized = tl.where(norm_mask, normal / norm, 0.0)

    # y = H_signed @ normalized  (matrix-vector multiply)
    rotated = tl.zeros([PADDED_DIM], dtype=tl.float32)
    for row in range(PADDED_DIM):
        h_row = tl.load(H_ptr + row * PADDED_DIM + norm_offs)
        dot = tl.sum(h_row * normalized, axis=0)
        rotated += tl.where(norm_offs == row, dot, 0.0)

    # ================================================================
    # 4. Scale → Lloyd-Max quantize
    # ================================================================
    sqrt_d = tl.sqrt(float(NORMAL_DIM))
    scaled = rotated * sqrt_d

    idx = tl.zeros([PADDED_DIM], dtype=tl.int32)
    for b in range(NUM_BOUNDARIES):
        boundary = tl.load(boundaries_ptr + b)
        idx = tl.where(norm_mask & (scaled > boundary), b + 1, idx)

    # ================================================================
    # 5. 4-bit pack (2 values per byte)
    # ================================================================
    # Use pairs: byte[i] = (idx[2i] << 4) | idx[2i+1]
    pair_offs = tl.arange(0, PADDED_DIM // 2)
    pair_mask = pair_offs < ((NORMAL_DIM + 1) // 2)

    # Gather even/odd indices via the rotated buffer trick:
    # Store indices to a scratch area then reload
    # Since we can't index dynamically, use the norm_offs trick
    even_idx = tl.sum(tl.where(norm_offs == 2 * pair_offs[:, None], idx[None, :], 0), axis=1)
    odd_valid = (2 * pair_offs + 1) < NORMAL_DIM
    odd_idx = tl.sum(
        tl.where(
            norm_offs == (2 * pair_offs[:, None] + 1),
            idx[None, :],
            0
        ),
        axis=1,
    )
    odd_idx = tl.where(odd_valid, odd_idx, 0)

    packed = ((even_idx << 4) | (odd_idx & 0x0F)).to(tl.uint8)
    tl.store(
        cache_ptr + cache_base + (PACKED_OFF + pair_offs) * cache_stride_4,
        packed,
        mask=pair_mask,
    )


def triton_fused_compress_scatter(
    x: torch.Tensor,           # (N, num_kv_heads, head_dim) bf16
    kv_cache: torch.Tensor,    # (num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES) uint8
    state,                     # QuantState
    H_fwd: torch.Tensor,       # (PADDED_DIM, PADDED_DIM) float32
    slot_mapping: torch.Tensor, # (N,) int32 or int64
    kv_dim: int,               # 0=K, 1=V
) -> None:
    """Phase 4: Fully fused Triton compress + scatter."""
    if x.numel() == 0:
        return

    _ensure_slot_buf(x.device)

    # Convert int64→int32
    if slot_mapping.dtype != torch.int32:
        n = slot_mapping.shape[0]
        sm = _slot_mapping_i32[:n]
        sm.copy_(slot_mapping)
    else:
        sm = slot_mapping

    N = x.shape[0]
    num_kv_heads = x.shape[1]
    grid = (N, num_kv_heads)

    _fused_compress_scatter_kernel[grid](
        x if x.is_contiguous() else x.contiguous(),
        kv_cache,
        sm,
        H_fwd,
        state.boundaries,
        kv_dim,
        x.stride(0), x.stride(1), x.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        kv_cache.stride(3), kv_cache.stride(4),
        kv_cache.shape[2],  # block_size
        NUM_OUTLIERS=state.num_outliers,
        NORMAL_DIM=state.normal_dim,
        PADDED_DIM=state.sign_flips.shape[0],
        HEAD_DIM=state.head_dim,
        NUM_BOUNDARIES=len(state.boundaries),
        PACKED_OFF=PACKED_OFFSET,
        NORM_OFF=NORM_OFFSET,
        SLOT_B=SLOT_BYTES,
    )
