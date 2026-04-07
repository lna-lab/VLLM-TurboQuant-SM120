"""Tests for fused (tiled) TurboQuant decode attention."""

import math

import pytest
import torch

from trinity_turbo.kernels.triton_compress import SLOT_BYTES, compress_to_slot
from trinity_turbo.kernels.triton_fused_attn import fused_tq_decode_attention
from trinity_turbo.quant.rotation import apply_inverse_rotation, apply_rotation
from trinity_turbo.quant.turboquant import QuantState, compress, full_decompress


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fused_vs_reference():
    """Fused tiled attention should match reference PyTorch attention."""
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    bits, head_dim, num_outliers = 4, 128, 8
    num_kv_heads, num_heads = 2, 4
    block_size, seq_len, num_blocks = 16, 48, 4

    state = QuantState.create(bits, head_dim, num_outliers, device)

    k_orig = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_orig = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    q = torch.randn(1, num_heads, head_dim, device=device, dtype=torch.bfloat16)

    # Build compressed KV cache
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES,
                           dtype=torch.uint8, device=device)
    for t in range(seq_len):
        bi, bo = t // block_size, t % block_size
        kv_cache[bi, 0, bo] = compress_to_slot(k_orig[t:t+1], state)
        kv_cache[bi, 1, bo] = compress_to_slot(v_orig[t:t+1], state)

    block_table = torch.arange(3, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], device=device, dtype=torch.int32)

    # ── Fused path ──
    q_rot = q.clone()
    q_rot[..., num_outliers:] = apply_rotation(
        q[..., num_outliers:].float(), state.sign_flips,
    ).to(torch.bfloat16)

    output_fused = torch.zeros(1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    fused_tq_decode_attention(
        query=q_rot, kv_cache=kv_cache, centroids=state.centroids,
        quant_state=state, block_table=block_table, seq_lens=seq_lens,
        query_start_loc=query_start_loc, output=output_fused,
        softmax_scale=1.0 / math.sqrt(head_dim),
        num_queries_per_kv=num_heads // num_kv_heads,
    )

    # Inverse rotate output
    out_n = output_fused[..., num_outliers:].float()
    output_fused[..., num_outliers:] = apply_inverse_rotation(out_n, state.sign_flips).to(torch.bfloat16)

    # ── Reference path ──
    k_c = compress(k_orig.reshape(-1, head_dim), state)
    v_c = compress(v_orig.reshape(-1, head_dim), state)
    k_r = full_decompress(k_c, state).reshape(seq_len, num_kv_heads, head_dim)
    v_r = full_decompress(v_c, state).reshape(seq_len, num_kv_heads, head_dim)

    hpk = num_heads // num_kv_heads
    k_e = k_r.unsqueeze(2).expand(-1, -1, hpk, -1).reshape(seq_len, num_heads, head_dim)
    v_e = v_r.unsqueeze(2).expand(-1, -1, hpk, -1).reshape(seq_len, num_heads, head_dim)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k_e.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    output_ref = torch.einsum("hqk,khd->qhd", attn, v_e.float())

    cos_sim = torch.nn.functional.cosine_similarity(
        output_fused.float().reshape(-1), output_ref.float().reshape(-1), dim=0,
    ).item()
    assert cos_sim > 0.85, f"Fused vs reference cosine similarity {cos_sim:.4f} < 0.85"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_fused_output_nonzero():
    """Output should be non-zero and correct shape."""
    device = torch.device("cuda:0")
    torch.manual_seed(123)

    state = QuantState.create(3, 128, 8, device)

    kv_cache = torch.zeros(2, 2, 16, 2, SLOT_BYTES, dtype=torch.uint8, device=device)
    k = torch.randn(1, 2, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 128, device=device, dtype=torch.bfloat16)
    kv_u8 = kv_cache.view(torch.uint8)
    kv_u8[0, 0, 0] = compress_to_slot(k, state)
    kv_u8[0, 1, 0] = compress_to_slot(v, state)

    q = torch.randn(1, 4, 128, device=device, dtype=torch.bfloat16)
    output = torch.zeros_like(q)

    fused_tq_decode_attention(
        query=q, kv_cache=kv_cache, centroids=state.centroids,
        quant_state=state,
        block_table=torch.tensor([[0]], device=device, dtype=torch.int32),
        seq_lens=torch.tensor([1], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], device=device, dtype=torch.int32),
        output=output, softmax_scale=1.0/math.sqrt(128),
        num_queries_per_kv=2,
    )

    assert output.shape == q.shape
    assert not torch.all(output == 0)
