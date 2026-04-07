"""Tests for Phase 3 TQ4 unified attention kernel.

Verifies that the tiled TQ4 kernel (triton_tq4_unified_attention)
produces correct attention output by comparing against the PyTorch
reference decompress + standard attention path.

Tests:
  1. Single-sequence decode (1 query token)
  2. Multi-sequence decode (batch of sequences)
  3. Sliding window attention
  4. Prefill (multiple query tokens per sequence)
  5. Cosine similarity vs reference attention
"""

import math

import pytest
import torch

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET,
    PACKED_OFFSET,
    PACKED_BYTES,
    SLOT_BYTES,
    compress_to_slot,
)
from trinity_turbo.kernels.triton_tq4_unified_attention import tq4_unified_attention
from trinity_turbo.quant.rotation import apply_inverse_rotation, apply_rotation
from trinity_turbo.quant.turboquant import CompressedKV, QuantState, decompress

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def decompress_slot_4bit(slot_u8: torch.Tensor, state: QuantState) -> torch.Tensor:
    """Correct 4-bit decompress from uint8 slot (PyTorch reference).

    decompress_slot_4bit uses a 3-bit Triton kernel that doesn't support 4-bit.
    This function uses the generic PyTorch decompress() which handles any bit width.
    """
    *batch, s = slot_u8.shape
    import math as _math
    nh = max(1, _math.prod(batch))
    flat = slot_u8.reshape(nh, s)

    outlier_bf16 = flat[:, :16].contiguous().view(torch.bfloat16)
    packed = flat[:, PACKED_OFFSET:PACKED_OFFSET + PACKED_BYTES].contiguous()
    norm_fp16 = flat[:, NORM_OFFSET:NORM_OFFSET + 2].contiguous().view(torch.float16)

    compressed = CompressedKV(
        packed_indices=packed, norms=norm_fp16, outliers=outlier_bf16,
    )
    return decompress(compressed, state).reshape(*batch, state.head_dim)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton requires CUDA",
)


def _make_paged_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    state: QuantState,
    block_size: int = 16,
):
    """Compress K/V and scatter into a paged uint8 KV cache.

    Args:
        keys: (seq_len, num_kv_heads, head_dim) bf16
        values: same
        state: QuantState
        block_size: tokens per block

    Returns:
        kv_cache: (num_blocks, 2, block_size, num_kv_heads, slot_bytes) uint8
        block_table: (1, num_blocks) int32
        seq_lens: (1,) int32
    """
    seq_len, num_kv_heads, head_dim = keys.shape
    device = keys.device
    num_blocks = (seq_len + block_size - 1) // block_size

    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES,
        dtype=torch.uint8, device=device,
    )

    for t in range(seq_len):
        blk = t // block_size
        off = t % block_size
        k_slot = compress_to_slot(keys[t:t + 1], state)
        v_slot = compress_to_slot(values[t:t + 1], state)
        kv_cache[blk, 0, off] = k_slot
        kv_cache[blk, 1, off] = v_slot

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    return kv_cache, block_table, seq_lens


def _reference_tq_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    state: QuantState,
    scale: float,
    causal: bool = True,
):
    """PyTorch reference: compress -> decompress -> rotated attention -> inverse rotate.

    Args:
        query: (num_q, num_heads, head_dim) bf16
        keys: (seq_len, num_kv_heads, head_dim) bf16
        values: same
        state: QuantState
        scale: softmax scale

    Returns:
        output: (num_q, num_heads, head_dim) float32
    """
    num_q, num_heads, head_dim = query.shape
    seq_len, num_kv_heads, _ = keys.shape
    heads_per_kv = num_heads // num_kv_heads

    # Compress and decompress (in rotated space)
    k_slots = compress_to_slot(keys, state)
    v_slots = compress_to_slot(values, state)
    k_dec = decompress_slot_4bit(k_slots, state)  # (seq_len, kv_heads, head_dim)
    v_dec = decompress_slot_4bit(v_slots, state)

    # Pre-rotate Q
    q_rot = query.clone().float()
    q_rot[..., state.num_outliers:] = apply_rotation(
        q_rot[..., state.num_outliers:], state.sign_flips,
    )

    # GQA expand
    k_exp = k_dec.unsqueeze(2).expand(-1, -1, heads_per_kv, -1)
    k_exp = k_exp.reshape(seq_len, num_heads, head_dim).float()
    v_exp = v_dec.unsqueeze(2).expand(-1, -1, heads_per_kv, -1)
    v_exp = v_exp.reshape(seq_len, num_heads, head_dim).float()

    # Attention
    scores = torch.einsum("qhd,khd->hqk", q_rot, k_exp) * scale

    if causal:
        # Query positions: last num_q tokens of the sequence
        q_pos = torch.arange(seq_len - num_q, seq_len, device=query.device)
        k_pos = torch.arange(seq_len, device=query.device)
        mask = k_pos[None, :] <= q_pos[:, None]  # (num_q, seq_len)
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", attn, v_exp)

    # Inverse-rotate output
    out[..., state.num_outliers:] = apply_inverse_rotation(
        out[..., state.num_outliers:], state.sign_flips,
    )

    return out


@pytest.fixture
def state():
    return QuantState.create(bits=4, head_dim=128, num_outliers=8, device=DEVICE)


class TestTQ4UnifiedDecode:
    """Test single-token decode (the primary use case)."""

    def test_single_sequence(self, state):
        """1 sequence, 1 query token, 64 context tokens."""
        torch.manual_seed(42)
        seq_len = 64
        num_kv_heads = 2
        num_heads = 4
        head_dim = 128
        block_size = 16
        scale = 1.0 / math.sqrt(head_dim)

        keys = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        values = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        query = torch.randn(1, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        # Reference
        ref = _reference_tq_attention(query, keys, values, state, scale)

        # TQ4 kernel
        kv_cache, block_table, seq_lens = _make_paged_cache(keys, values, state, block_size)
        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]

        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        out = torch.zeros(1, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        # Pre-rotate Q
        q = query.clone().to(torch.bfloat16)
        q[..., state.num_outliers:] = apply_rotation(
            q[..., state.num_outliers:].float(), state.sign_flips,
        ).to(torch.bfloat16)

        tq4_unified_attention(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seq_lens,
            softmax_scale=scale,
            window_size=(-1, -1),
            block_table=block_table,
            centroids=state.centroids,
            inv_sqrt_d=1.0 / math.sqrt(state.normal_dim),
            num_outliers=state.num_outliers,
            packed_off=PACKED_OFFSET,
            norm_off=NORM_OFFSET,
        )

        # Inverse-rotate output
        out_f = out.float()
        out_f[..., state.num_outliers:] = apply_inverse_rotation(
            out_f[..., state.num_outliers:], state.sign_flips,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.reshape(-1), out_f.reshape(-1), dim=0,
        ).item()
        assert cos_sim > 0.95, f"Decode cos_sim={cos_sim:.4f} < 0.95"

    def test_multi_sequence_batch(self, state):
        """4 sequences with different lengths, 1 query each."""
        torch.manual_seed(123)
        num_kv_heads = 2
        num_heads = 4
        head_dim = 128
        block_size = 16
        scale = 1.0 / math.sqrt(head_dim)
        seq_lengths = [32, 48, 16, 64]

        # Build paged cache for all sequences
        max_blocks_per_seq = max((s + block_size - 1) // block_size for s in seq_lengths)
        total_blocks = sum((s + block_size - 1) // block_size for s in seq_lengths)

        kv_cache = torch.zeros(
            total_blocks, 2, block_size, num_kv_heads, SLOT_BYTES,
            dtype=torch.uint8, device=DEVICE,
        )
        block_table = torch.zeros(
            len(seq_lengths), max_blocks_per_seq, dtype=torch.int32, device=DEVICE,
        )

        blk_offset = 0
        all_keys = []
        all_values = []
        for i, sl in enumerate(seq_lengths):
            k = torch.randn(sl, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
            v = torch.randn(sl, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
            all_keys.append(k)
            all_values.append(v)
            n_blks = (sl + block_size - 1) // block_size
            for b in range(n_blks):
                blk_start = b * block_size
                blk_end = min(blk_start + block_size, sl)
                for t in range(blk_start, blk_end):
                    off = t % block_size
                    k_slot = compress_to_slot(k[t:t + 1], state)
                    v_slot = compress_to_slot(v[t:t + 1], state)
                    kv_cache[blk_offset + b, 0, off] = k_slot
                    kv_cache[blk_offset + b, 1, off] = v_slot
                block_table[i, b] = blk_offset + b
            blk_offset += n_blks

        # Queries: 1 per sequence = 4 total
        queries = torch.randn(4, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        seq_lens_t = torch.tensor(seq_lengths, dtype=torch.int32, device=DEVICE)
        cu_seqlens_q = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int32, device=DEVICE,
        )

        # Pre-rotate Q
        q = queries.clone().to(torch.bfloat16)
        q[..., state.num_outliers:] = apply_rotation(
            q[..., state.num_outliers:].float(), state.sign_flips,
        ).to(torch.bfloat16)

        out = torch.zeros(4, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]

        tq4_unified_attention(
            q=q, k_cache=key_cache, v_cache=value_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seq_lens_t,
            softmax_scale=scale, window_size=(-1, -1),
            block_table=block_table, centroids=state.centroids,
            inv_sqrt_d=1.0 / math.sqrt(state.normal_dim),
            num_outliers=state.num_outliers,
            packed_off=PACKED_OFFSET, norm_off=NORM_OFFSET,
        )

        # Inverse-rotate
        out_f = out.float()
        out_f[..., state.num_outliers:] = apply_inverse_rotation(
            out_f[..., state.num_outliers:], state.sign_flips,
        )

        # Check each sequence independently
        for i, sl in enumerate(seq_lengths):
            ref_i = _reference_tq_attention(
                queries[i:i + 1], all_keys[i], all_values[i], state, scale,
            )
            cos_sim = torch.nn.functional.cosine_similarity(
                ref_i.reshape(-1), out_f[i:i + 1].reshape(-1), dim=0,
            ).item()
            assert cos_sim > 0.93, (
                f"Seq {i} (len={sl}) cos_sim={cos_sim:.4f} < 0.93"
            )


class TestTQ4UnifiedSlidingWindow:
    """Test sliding window attention."""

    def test_sliding_window(self, state):
        """Verify that tokens outside the window are ignored."""
        torch.manual_seed(99)
        seq_len = 128
        window = 32  # attend only to last 32 tokens
        num_kv_heads = 2
        num_heads = 4
        head_dim = 128
        block_size = 16
        scale = 1.0 / math.sqrt(head_dim)

        keys = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        values = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        query = torch.randn(1, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        # Reference: only use last `window` tokens
        ref = _reference_tq_attention(
            query, keys[-window:], values[-window:], state, scale, causal=False,
        )

        # TQ4 kernel with sliding window
        kv_cache, block_table, seq_lens = _make_paged_cache(keys, values, state, block_size)
        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]

        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
        out = torch.zeros(1, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        q = query.clone().to(torch.bfloat16)
        q[..., state.num_outliers:] = apply_rotation(
            q[..., state.num_outliers:].float(), state.sign_flips,
        ).to(torch.bfloat16)

        tq4_unified_attention(
            q=q, k_cache=key_cache, v_cache=value_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seq_lens,
            softmax_scale=scale,
            window_size=(window - 1, 0),  # vLLM convention: window_size[0] = window - 1
            block_table=block_table, centroids=state.centroids,
            inv_sqrt_d=1.0 / math.sqrt(state.normal_dim),
            num_outliers=state.num_outliers,
            packed_off=PACKED_OFFSET, norm_off=NORM_OFFSET,
        )

        out_f = out.float()
        out_f[..., state.num_outliers:] = apply_inverse_rotation(
            out_f[..., state.num_outliers:], state.sign_flips,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.reshape(-1), out_f.reshape(-1), dim=0,
        ).item()
        assert cos_sim > 0.90, f"Sliding window cos_sim={cos_sim:.4f} < 0.90"


class TestTQ4UnifiedPrefill:
    """Test multi-token prefill."""

    def test_short_prefill(self, state):
        """Prefill with 4 query tokens attending to 16 context tokens."""
        torch.manual_seed(77)
        context_len = 16
        num_new = 4
        seq_len = context_len + num_new
        num_kv_heads = 2
        num_heads = 4
        head_dim = 128
        block_size = 16
        scale = 1.0 / math.sqrt(head_dim)

        keys = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        values = torch.randn(seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        queries = torch.randn(num_new, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

        # Reference
        ref = _reference_tq_attention(queries, keys, values, state, scale, causal=True)

        # TQ4 kernel
        kv_cache, block_table, _ = _make_paged_cache(keys, values, state, block_size)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        cu_seqlens_q = torch.tensor([0, num_new], dtype=torch.int32, device=DEVICE)

        q = queries.clone().to(torch.bfloat16)
        q[..., state.num_outliers:] = apply_rotation(
            q[..., state.num_outliers:].float(), state.sign_flips,
        ).to(torch.bfloat16)

        out = torch.zeros(num_new, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]

        tq4_unified_attention(
            q=q, k_cache=key_cache, v_cache=value_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, seqused_k=seq_lens,
            softmax_scale=scale, window_size=(-1, -1),
            block_table=block_table, centroids=state.centroids,
            inv_sqrt_d=1.0 / math.sqrt(state.normal_dim),
            num_outliers=state.num_outliers,
            packed_off=PACKED_OFFSET, norm_off=NORM_OFFSET,
        )

        out_f = out.float()
        out_f[..., state.num_outliers:] = apply_inverse_rotation(
            out_f[..., state.num_outliers:], state.sign_flips,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            ref.reshape(-1), out_f.reshape(-1), dim=0,
        ).item()
        assert cos_sim > 0.90, f"Prefill cos_sim={cos_sim:.4f} < 0.90"
