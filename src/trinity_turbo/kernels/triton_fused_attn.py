"""Fused TurboQuant 4-bit paged decode attention — vectorized, zero buffers, CUDA graph safe.

4-bit packing: 2 values per byte → trivial unpack (>>4, &0xF).
Vectorized loads: 60 bytes → 120 indices in one tl.load + shift/mask.

Slot layout (80 bytes):
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

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET,
    PACKED_BYTES,
    PACKED_OFFSET,
    SLOT_BYTES,
)


@triton.jit
def _unpack_4bit_dequant_vec(
    slot_base, centroids_ptr, inv_sqrt_d, active,
    PACKED_OFF: tl.constexpr,
    PACKED_B: tl.constexpr,     # 60
    NORMAL_DIM: tl.constexpr,   # 120
    NORMAL_PAD: tl.constexpr,   # 128
):
    """Vectorized 4-bit unpack → centroid lookup. Returns [NORMAL_PAD] float32."""
    # 4-bit unpack: dim d → byte d//2, hi nibble if d%2==0 else lo nibble
    d = tl.arange(0, NORMAL_PAD)  # [0..127], NORMAL_PAD=128 (power of 2)
    valid = d < NORMAL_DIM        # [0..119] active

    byte_idx = d // 2             # [0..59]
    is_hi = (d % 2) == 0

    raw = tl.load(
        slot_base + PACKED_OFF + byte_idx,
        mask=valid & active & (byte_idx < PACKED_B), other=0,
    ).to(tl.int32)
    indices = tl.where(is_hi, (raw >> 4) & 0x0F, raw & 0x0F)

    # Centroid lookup
    values = tl.load(centroids_ptr + indices, mask=valid, other=0.0)
    return tl.where(valid, values * inv_sqrt_d, 0.0)


@triton.jit
def _load_outliers_vec(slot_base, active, NUM_OUTLIERS: tl.constexpr):
    """Vectorized bf16 outlier load. Returns [NUM_OUTLIERS] float32."""
    offs = tl.arange(0, NUM_OUTLIERS)
    lo = tl.load(slot_base + 2 * offs, mask=active, other=0).to(tl.uint16)
    hi = tl.load(slot_base + 2 * offs + 1, mask=active, other=0).to(tl.uint16)
    raw = lo | (hi << 8)
    return raw.to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _load_norm(slot_base, active, NORM_OFF: tl.constexpr):
    """Load fp16 norm scalar."""
    lo = tl.load(slot_base + NORM_OFF, mask=active, other=0).to(tl.uint16)
    hi = tl.load(slot_base + NORM_OFF + 1, mask=active, other=0).to(tl.uint16)
    raw = lo | (hi << 8)
    return raw.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _fused_tq4_paged_decode_kernel(
    Q_ptr, KV_ptr, CENTROIDS_ptr, BT_ptr, SEQLENS_ptr, QSTART_ptr, OUT_ptr,
    sm_scale, inv_sqrt_d,
    stride_q0, stride_q1, stride_q2,
    stride_kv0, stride_kv1, stride_kv2, stride_kv3, stride_kv4,
    stride_bt0, stride_bt1,
    stride_o0, stride_o1, stride_o2,
    max_ctx,  # runtime
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERIES_PER_KV: tl.constexpr,
    NUM_OUTLIERS: tl.constexpr,
    NORMAL_DIM: tl.constexpr,
    NORMAL_PAD: tl.constexpr,
    PACKED_OFF: tl.constexpr,
    PACKED_B: tl.constexpr,
    NORM_OFF: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused 4-bit paged decode attention with vectorized unpack."""
    pid_bh = tl.program_id(0)
    seq_idx = pid_bh // NUM_Q_HEADS
    q_head = pid_bh % NUM_Q_HEADS
    kv_head = q_head // NUM_QUERIES_PER_KV

    seq_len = tl.load(SEQLENS_ptr + seq_idx)
    q_tok = tl.load(QSTART_ptr + seq_idx)

    # Load Q outliers [NUM_OUTLIERS] and Q normals [NORMAL_PAD]
    q_out_offs = tl.arange(0, NUM_OUTLIERS)
    q_outlier = tl.load(
        Q_ptr + q_tok * stride_q0 + q_head * stride_q1 + q_out_offs * stride_q2,
    ).to(tl.float32)

    q_n_offs = tl.arange(0, NORMAL_PAD)
    q_normal = tl.load(
        Q_ptr + q_tok * stride_q0 + q_head * stride_q1 + (NUM_OUTLIERS + q_n_offs) * stride_q2,
        mask=q_n_offs < NORMAL_DIM, other=0.0,
    ).to(tl.float32)

    # Online softmax
    m_i = tl.full([1], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc_out = tl.zeros([NUM_OUTLIERS], dtype=tl.float32)
    acc_nrm = tl.zeros([NORMAL_PAD], dtype=tl.float32)

    for n_base in range(0, max_ctx, BLOCK_N):
        scores = tl.full([BLOCK_N], -float("inf"), dtype=tl.float32)

        for j in range(BLOCK_N):
            nn = n_base + j
            active = nn < seq_len
            lb = nn // BLOCK_SIZE
            tt = nn % BLOCK_SIZE

            pb = tl.load(
                BT_ptr + seq_idx * stride_bt0 + lb * stride_bt1,
                mask=active, other=0,
            ).to(tl.int64)

            k_base = KV_ptr + pb * stride_kv0 + tt * stride_kv2 + kv_head * stride_kv3

            # K score: outlier dot + normal dot × norm
            k_out = _load_outliers_vec(k_base, active, NUM_OUTLIERS)
            out_sc = tl.sum(q_outlier * k_out)

            k_nrm = _unpack_4bit_dequant_vec(
                k_base, CENTROIDS_ptr, inv_sqrt_d, active,
                PACKED_OFF, PACKED_B, NORMAL_DIM, NORMAL_PAD,
            )
            nrm_sc = tl.sum(q_normal * k_nrm)
            k_norm = _load_norm(k_base, active, NORM_OFF)

            sc = (out_sc + nrm_sc * k_norm) * sm_scale
            sc = tl.where(active, sc, -float("inf"))

            j_ar = tl.arange(0, BLOCK_N)
            scores = tl.where(j_ar == j, sc, scores)

        # Softmax update
        m_blk = tl.max(scores, 0)
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha
        acc_out = acc_out * alpha
        acc_nrm = acc_nrm * alpha

        exp_s = tl.exp(scores - m_new)
        mask_n = (n_base + tl.arange(0, BLOCK_N)) < seq_len
        exp_s = tl.where(mask_n, exp_s, 0.0)
        l_i = l_i + tl.sum(exp_s, 0)
        m_i = m_new

        # V accumulation
        for j in range(BLOCK_N):
            nn = n_base + j
            active = nn < seq_len
            lb = nn // BLOCK_SIZE
            tt = nn % BLOCK_SIZE

            pb = tl.load(
                BT_ptr + seq_idx * stride_bt0 + lb * stride_bt1,
                mask=active, other=0,
            ).to(tl.int64)

            v_base = (
                KV_ptr + pb * stride_kv0 + 1 * stride_kv1
                + tt * stride_kv2 + kv_head * stride_kv3
            )

            v_out = _load_outliers_vec(v_base, active, NUM_OUTLIERS)
            v_nrm_vec = _unpack_4bit_dequant_vec(
                v_base, CENTROIDS_ptr, inv_sqrt_d, active,
                PACKED_OFF, PACKED_B, NORMAL_DIM, NORMAL_PAD,
            )
            v_norm = _load_norm(v_base, active, NORM_OFF)

            j_ar = tl.arange(0, BLOCK_N)
            es_j = tl.sum(tl.where(j_ar == j, exp_s, 0.0), 0)
            p_safe = tl.where(active, es_j, 0.0)

            acc_out += p_safe * v_out
            acc_nrm += p_safe * v_nrm_vec * v_norm

    # Normalize
    denom = tl.maximum(l_i, 1e-10)
    acc_out = acc_out / denom
    acc_nrm = acc_nrm / denom

    # Store
    o_offs = tl.arange(0, NUM_OUTLIERS)
    tl.store(
        OUT_ptr + q_tok * stride_o0 + q_head * stride_o1 + o_offs * stride_o2,
        acc_out.to(tl.bfloat16),
    )
    n_offs = tl.arange(0, NORMAL_PAD)
    tl.store(
        OUT_ptr + q_tok * stride_o0 + q_head * stride_o1 + (NUM_OUTLIERS + n_offs) * stride_o2,
        acc_nrm.to(tl.bfloat16),
        mask=n_offs < NORMAL_DIM,
    )


def fused_tq_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    centroids: torch.Tensor,
    quant_state,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    sliding_window: int = 0,
    num_queries_per_kv: int = 1,
) -> torch.Tensor:
    """Fused TurboQuant 4-bit decode attention."""
    kv_u8 = kv_cache.view(torch.uint8)
    num_seqs = seq_lens.shape[0]
    num_heads = query.shape[1]
    num_kv_heads = kv_u8.shape[3]
    block_size = kv_u8.shape[2]

    inv_sqrt_d = 1.0 / math.sqrt(quant_state.normal_dim)
    max_blocks = block_table.shape[1] if block_table.dim() == 2 else 1
    max_ctx = max_blocks * block_size

    grid = (num_seqs * num_heads,)

    _fused_tq4_paged_decode_kernel[grid](
        query, kv_u8, centroids,
        block_table, seq_lens, query_start_loc, output,
        softmax_scale, inv_sqrt_d,
        query.stride(0), query.stride(1), query.stride(2),
        kv_u8.stride(0), kv_u8.stride(1), kv_u8.stride(2),
        kv_u8.stride(3), kv_u8.stride(4),
        block_table.stride(0), block_table.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        max_ctx,
        NUM_Q_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=query.shape[2],
        NUM_QUERIES_PER_KV=num_queries_per_kv,
        NUM_OUTLIERS=quant_state.num_outliers,
        NORMAL_DIM=quant_state.normal_dim,
        NORMAL_PAD=128,
        PACKED_OFF=PACKED_OFFSET,
        PACKED_B=PACKED_BYTES,
        NORM_OFF=NORM_OFFSET,
        BLOCK_N=8,
    )

    return output
