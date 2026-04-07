"""Integration test for the Phase 2 compress → cache → decompress → attend flow.

Tests the full pipeline WITHOUT vLLM, using synthetic cache tensors and
metadata to verify correctness of:
  1. compress_to_slot → scatter to uint8 cache
  2. gather used blocks → decompress_from_slot → bf16 temporary
  3. Pre-rotate Q via WHT
  4. Attention in rotated space → inverse-rotate output
"""

import math

import pytest
import torch

from trinity_turbo.kernels.triton_compress import SLOT_BYTES, compress_to_slot
from trinity_turbo.kernels.triton_decompress import decompress_from_slot
from trinity_turbo.quant.rotation import apply_inverse_rotation, apply_rotation
from trinity_turbo.quant.turboquant import QuantState


@pytest.fixture
def setup():
    """Create a minimal paged KV cache scenario."""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2026)

    num_blocks = 8
    block_size = 16
    num_kv_heads = 2
    head_dim = 128
    bits = 3
    num_outliers = 8

    state = QuantState.create(bits, head_dim, num_outliers, device)

    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES,
        dtype=torch.uint8, device=device,
    )

    return dict(
        device=device, state=state, kv_cache=kv_cache,
        num_blocks=num_blocks, block_size=block_size,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
    )


def test_cache_write_read_roundtrip(setup):
    """Compress → scatter → gather → decompress preserves vectors."""
    s = setup
    state, kv_cache = s["state"], s["kv_cache"]
    block_size = s["block_size"]
    device = s["device"]

    num_tokens = 4
    key = torch.randn(num_tokens, s["num_kv_heads"], s["head_dim"],
                       device=device, dtype=torch.bfloat16)

    # Slot mapping: tokens go to block 2, offsets 0-3
    slot_mapping = torch.arange(num_tokens, device=device) + 2 * block_size

    # Write
    k_slots = compress_to_slot(key, state)
    block_idx = slot_mapping // block_size
    offset = slot_mapping % block_size
    kv_cache[block_idx, 0, offset] = k_slots

    # Read
    used_block = torch.tensor([2], device=device)
    gathered_k = kv_cache[used_block, 0]
    flat_k = gathered_k.reshape(-1, SLOT_BYTES)
    dec_k = decompress_from_slot(flat_k, state)
    dec_k = dec_k.reshape(1, block_size, s["num_kv_heads"], s["head_dim"])

    recon_key = dec_k[0, :num_tokens]
    ref = decompress_from_slot(k_slots, state)

    torch.testing.assert_close(recon_key, ref, atol=1e-2, rtol=1e-2)


def test_q_rotation_dot_product(setup):
    """Rotated-space dot product ≈ original (with truncation tolerance).

    WHT pads 120→128, rotates, truncates back to 120.
    The 8 lost coordinates introduce ~6% error.
    """
    s = setup
    state = s["state"]
    device = s["device"]
    head_dim = s["head_dim"]

    # Use UNIT-NORMALIZED vectors to keep error bounded
    q = torch.randn(64, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(64, head_dim, device=device, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)

    # Rotate both Q and K normal channels
    q_out = q.clone()
    q_out[:, state.num_outliers:] = apply_rotation(
        q[:, state.num_outliers:], state.sign_flips,
    )
    k_out = k.clone()
    k_out[:, state.num_outliers:] = apply_rotation(
        k[:, state.num_outliers:], state.sign_flips,
    )

    dots_orig = (q * k).sum(dim=-1)
    dots_rot = (q_out * k_out).sum(dim=-1)

    # ~6% truncation error for unit vectors
    torch.testing.assert_close(dots_orig, dots_rot, atol=0.15, rtol=0.15)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_full_attention_quality(setup):
    """Full pipeline: compress KV + rotate Q + attend + inverse-rotate output."""
    s = setup
    state = s["state"]
    device = s["device"]
    head_dim = s["head_dim"]
    num_kv_heads = s["num_kv_heads"]

    seq_len = 32
    num_heads = 4  # GQA: 4 query heads, 2 KV heads

    q = torch.randn(1, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

    # --- Reference: standard attention ---
    heads_per_kv = num_heads // num_kv_heads
    k_exp = k.unsqueeze(2).expand(-1, -1, heads_per_kv, -1).reshape(seq_len, num_heads, head_dim)
    v_exp = v.unsqueeze(2).expand(-1, -1, heads_per_kv, -1).reshape(seq_len, num_heads, head_dim)

    scale = 1.0 / math.sqrt(head_dim)
    scores_ref = torch.einsum("qhd,khd->hqk", q.float(), k_exp.float()) * scale
    attn_ref = torch.softmax(scores_ref, dim=-1)
    out_ref = torch.einsum("hqk,khd->qhd", attn_ref, v_exp.float())

    # --- TurboQuant: compress KV → rotated decompress → rotate Q → attend ---
    k_slots = compress_to_slot(k, state)
    v_slots = compress_to_slot(v, state)
    k_dec = decompress_from_slot(k_slots, state)  # rotated space
    v_dec = decompress_from_slot(v_slots, state)  # rotated space

    # Rotate Q normal channels
    q_rot = q.clone()
    q_rot[..., state.num_outliers:] = apply_rotation(
        q[..., state.num_outliers:].float(), state.sign_flips,
    ).to(q.dtype)

    # GQA expand
    k_dec_exp = k_dec.unsqueeze(2).expand(-1, -1, heads_per_kv, -1).reshape(seq_len, num_heads, head_dim)
    v_dec_exp = v_dec.unsqueeze(2).expand(-1, -1, heads_per_kv, -1).reshape(seq_len, num_heads, head_dim)

    # Attention in rotated space
    scores_tq = torch.einsum("qhd,khd->hqk", q_rot.float(), k_dec_exp.float()) * scale
    attn_tq = torch.softmax(scores_tq, dim=-1)
    out_tq = torch.einsum("hqk,khd->qhd", attn_tq, v_dec_exp.float())

    # Inverse-rotate output normal channels back to original space
    out_tq_normal = out_tq[..., state.num_outliers:]
    out_tq[..., state.num_outliers:] = apply_inverse_rotation(
        out_tq_normal, state.sign_flips,
    )

    # Compare
    cos_sim = torch.nn.functional.cosine_similarity(
        out_ref.reshape(-1), out_tq.reshape(-1), dim=0,
    ).item()
    assert cos_sim > 0.90, f"Attention output cosine similarity {cos_sim:.4f} < 0.90"
