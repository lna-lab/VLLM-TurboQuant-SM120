"""TurboQuant 4-bit unified paged attention — Triton kernel.

Fork of vLLM's triton_unified_attention.py with in-tile TQ4 decompress.
Replaces K/V bf16/fp8 loads with:
  - uint8 byte loads from compressed slots (80 bytes/head/token)
  - In-register 4-bit unpack -> centroid lookup -> norm scale
  - Outlier bf16 byte-pair reconstruction

Performance: vLLM tiled parallelism (BLOCK_M x TILE_SIZE) + tl.dot matmuls.
Memory: TQ4 80 bytes vs FP8 128 bytes = 1.6x compression.
CUDA graph compatible (no .item(), no dynamic ops, all constexpr branches).

Slot layout (80 bytes per token per KV head):
  [0, 16)   8 outlier channels as bf16     (16 bytes)
  [16, 76)  120 normal channels, 4-bit     (60 bytes)
  [76, 78)  L2 norm fp16                   (2 bytes)
  [78, 80)  padding                        (2 bytes)
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helper functions (copied from vLLM to avoid cross-module JIT issues)
# ---------------------------------------------------------------------------

@triton.jit
def _cdiv(x, y):
    return (x + y - 1) // y


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Binary search for the sequence index that owns target_idx."""
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid  # q_block mode
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


# ---------------------------------------------------------------------------
# Main TQ4 unified attention kernel (2D)
# ---------------------------------------------------------------------------

@triton.jit
def kernel_tq4_unified_attention_2d(
    # Pointers — standard
    output_ptr,        # [num_tokens, num_query_heads, head_size]
    query_ptr,         # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,     # [num_blocks, block_size, num_kv_heads, SLOT_BYTES] uint8
    value_cache_ptr,   # [num_blocks, block_size, num_kv_heads, SLOT_BYTES] uint8
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,      # [num_seqs]
    # Scalars — standard
    scale,             # float32 softmax scale
    # TQ4-specific pointers
    centroids_ptr,     # [2^bits] float32 centroid table
    inv_sqrt_d,        # float32: 1/sqrt(normal_dim)
    # Strides
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    block_table_stride: tl.int64,
    stride_k0: tl.int64,   # key_cache stride(0) — per block
    stride_k1: tl.int64,   # stride(1) — per token in block
    stride_k2: tl.int64,   # stride(2) — per KV head
    stride_v0: tl.int64,
    stride_v1: tl.int64,
    stride_v2: tl.int64,
    query_start_len_ptr,  # [num_seqs+1]
    num_seqs: tl.int32,
    # Constexprs — attention
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    # Constexprs — TQ4 slot layout
    TQ_NUM_OUTLIERS: tl.constexpr,   # 8
    TQ_PACKED_OFF: tl.constexpr,     # 16
    TQ_NORM_OFF: tl.constexpr,       # 76
):
    # ---- Program ID mapping ----
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q,
    )
    q_block_start_idx = (
        tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    )
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_start = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_stop - cur_batch_start

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # ---- Index setup ----
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_start + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)

    # ---- Load Q : (BLOCK_M, HEAD_SIZE_PADDED) ----
    q_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    Q = tl.load(
        query_ptr + q_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    # ---- Sequence info ----
    block_table_offset = seq_idx * block_table_stride
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len

    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding window tile pruning ----
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        first_allowed = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed // TILE_SIZE)
        tile_end = tl.minimum((last_allowed // TILE_SIZE) + 1, num_tiles)

    # ---- TQ4 offset precomputation (constant across tiles) ----
    is_out = offs_d < TQ_NUM_OUTLIERS
    normal_d = tl.maximum(offs_d - TQ_NUM_OUTLIERS, 0)
    # Byte offset for first load: outlier lo byte or packed byte
    byte1_off = tl.where(is_out, 2 * offs_d, TQ_PACKED_OFF + normal_d // 2)
    # Byte offset for second load: outlier hi byte (only used where is_out)
    byte2_off = 2 * offs_d + 1
    # 4-bit nibble selection: even normal index -> hi nibble
    is_hi_nibble = (normal_d % 2) == 0

    # ---- Online softmax accumulators ----
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # ---- Tile loop ----
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        # Physical block indices for this tile
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        # ============================================================
        # K DECOMPRESS: (HEAD_SIZE_PADDED, TILE_SIZE) from uint8 slots
        # ============================================================
        k_token_base = (
            physical_block_idx * stride_k0
            + kv_head_idx * stride_k2
            + (seq_offset % BLOCK_SIZE) * stride_k1
        )
        # (TILE_SIZE,) — base byte address per token

        # Load byte1: packed byte for normals, lo byte for outliers
        # Phase 5: transposed K load — byte offset as fast dim for coalesced HBM access
        # Slot is 80 bytes ≤ 128-byte cache line → one fetch serves all bytes per token
        kb1 = tl.load(
            key_cache_ptr + k_token_base[:, None] + byte1_off[None, :],
            mask=tile_mask[:, None] & dim_mask[None, :],
            other=0,
        )  # (TILE_SIZE, HEAD_SIZE_PADDED) uint8 — coalesced!

        # Load byte2: hi byte for outlier bf16 reconstruction
        kb2 = tl.load(
            key_cache_ptr + k_token_base[:, None] + byte2_off[None, :],
            mask=tile_mask[:, None] & is_out[None, :],
            other=0,
        )  # (TILE_SIZE, HEAD_SIZE_PADDED) uint8 — coalesced!

        # Outlier decode: byte pair -> bf16 -> float32
        k_bf16_raw = kb1.to(tl.uint16) | (kb2.to(tl.uint16) << 8)
        K_outlier = k_bf16_raw.to(tl.bfloat16, bitcast=True).to(tl.float32)

        # Normal decode: 4-bit unpack -> centroid lookup
        k_raw = kb1.to(tl.int32)
        k_idx = tl.where(
            is_hi_nibble[None, :],
            (k_raw >> 4) & 0x0F,
            k_raw & 0x0F,
        )
        K_normal = tl.load(
            centroids_ptr + k_idx,
            mask=tile_mask[:, None] & (~is_out)[None, :],
            other=0.0,
        )

        # Load per-token L2 norm (fp16 stored as 2 bytes)
        k_norm_lo = tl.load(
            key_cache_ptr + k_token_base + TQ_NORM_OFF,
            mask=tile_mask, other=0,
        ).to(tl.uint16)
        k_norm_hi = tl.load(
            key_cache_ptr + k_token_base + TQ_NORM_OFF + 1,
            mask=tile_mask, other=0,
        ).to(tl.uint16)
        k_norm = (k_norm_lo | (k_norm_hi << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)
        # (TILE_SIZE,)

        # Scale normals and merge with outliers — Phase 5 transposed path
        K_normal = K_normal * inv_sqrt_d * k_norm[:, None]
        K_T = tl.where(is_out[None, :], K_outlier, K_normal).to(Q.dtype)
        # K_T: (TILE_SIZE, HEAD_SIZE_PADDED) — transpose for dot product
        K = tl.trans(K_T)

        # ============================================================
        # V DECOMPRESS: (TILE_SIZE, HEAD_SIZE_PADDED) from uint8 slots
        # ============================================================
        v_token_base = (
            physical_block_idx * stride_v0
            + kv_head_idx * stride_v2
            + (seq_offset % BLOCK_SIZE) * stride_v1
        )

        vb1 = tl.load(
            value_cache_ptr + v_token_base[:, None] + byte1_off[None, :],
            mask=tile_mask[:, None] & dim_mask[None, :],
            other=0,
        )  # (TILE_SIZE, HEAD_SIZE_PADDED) uint8

        vb2 = tl.load(
            value_cache_ptr + v_token_base[:, None] + byte2_off[None, :],
            mask=tile_mask[:, None] & is_out[None, :],
            other=0,
        )

        v_bf16_raw = vb1.to(tl.uint16) | (vb2.to(tl.uint16) << 8)
        V_outlier = v_bf16_raw.to(tl.bfloat16, bitcast=True).to(tl.float32)

        v_raw = vb1.to(tl.int32)
        v_idx = tl.where(
            is_hi_nibble[None, :],
            (v_raw >> 4) & 0x0F,
            v_raw & 0x0F,
        )
        V_normal = tl.load(
            centroids_ptr + v_idx,
            mask=tile_mask[:, None] & (~is_out)[None, :],
            other=0.0,
        )

        v_norm_lo = tl.load(
            value_cache_ptr + v_token_base + TQ_NORM_OFF,
            mask=tile_mask, other=0,
        ).to(tl.uint16)
        v_norm_hi = tl.load(
            value_cache_ptr + v_token_base + TQ_NORM_OFF + 1,
            mask=tile_mask, other=0,
        ).to(tl.uint16)
        v_norm = (v_norm_lo | (v_norm_hi << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)

        V_normal = V_normal * inv_sqrt_d * v_norm[:, None]
        V = tl.where(is_out[None, :], V_outlier, V_normal).to(Q.dtype)
        # V: (TILE_SIZE, HEAD_SIZE_PADDED)

        # ============================================================
        # STANDARD ATTENTION: causal mask, softmax, accumulate
        # ============================================================
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & (
                (query_abs_pos - seq_offset[None, :]) < SLIDING_WINDOW
            )

        # S : (BLOCK_M, TILE_SIZE)
        S = scale * tl.dot(Q, K)

        S = tl.where(
            query_mask_1[:, None]
            & query_mask_0[:, None]
            & seq_mask,
            S,
            float("-inf"),
        )

        # Online softmax update
        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        if SLIDING_WINDOW > 0:
            qpos_lo_inner = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo_inner - seq_offset[:, None])
                < SLIDING_WINDOW,
                V,
                0.0,
            )

        acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue: normalize and store ----
    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def tq4_unified_attention(
    q: torch.Tensor,          # (num_tokens, num_query_heads, head_size) bf16
    k_cache: torch.Tensor,    # (num_blocks, block_size, num_kv_heads, SLOT_BYTES) uint8
    v_cache: torch.Tensor,    # same
    out: torch.Tensor,        # (num_tokens, num_query_heads, head_size) bf16
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    window_size: tuple[int, int],
    block_table: torch.Tensor,
    centroids: torch.Tensor,  # (2^bits,) float32
    inv_sqrt_d: float,
    num_outliers: int,
    packed_off: int,
    norm_off: int,
) -> None:
    """TQ4 unified attention — Python launcher."""
    block_size = k_cache.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Tile size: always 16 for CUDA graph compatibility (no .item() sync).
    # 16 works well for both decode and short prefill on Blackwell.
    TILE_SIZE = 16

    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)

    grid = (total_num_q_blocks, num_kv_heads)

    kernel_tq4_unified_attention_2d[grid](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_cache,
        value_cache_ptr=v_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        scale=softmax_scale,
        centroids_ptr=centroids,
        inv_sqrt_d=inv_sqrt_d,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        stride_k0=k_cache.stride(0),
        stride_k1=k_cache.stride(1),
        stride_k2=k_cache.stride(2),
        stride_v0=v_cache.stride(0),
        stride_v1=v_cache.stride(1),
        stride_v2=v_cache.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        SLIDING_WINDOW=sliding_window_val,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        TQ_NUM_OUTLIERS=num_outliers,
        TQ_PACKED_OFF=packed_off,
        TQ_NORM_OFF=norm_off,
    )
