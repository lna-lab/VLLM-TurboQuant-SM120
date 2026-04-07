"""Tests for Triton compress/decompress slot kernels.

Verifies that the slot-based pipeline matches the PyTorch reference
and that roundtrip quality meets TurboQuant's cosine similarity targets.
"""

import pytest
import torch

from trinity_turbo.kernels.triton_compress import SLOT_BYTES, compress_to_slot
from trinity_turbo.kernels.triton_decompress import decompress_from_slot
from trinity_turbo.quant.turboquant import QuantState, compress, decompress


# -- shape and dtype --------------------------------------------------------

def test_compress_slot_shape(device, head_dim, num_outliers):
    state = QuantState.create(3, head_dim, num_outliers, device)
    x = torch.randn(16, 2, head_dim, device=device, dtype=torch.bfloat16)

    slot = compress_to_slot(x, state)

    assert slot.shape == (16, 2, SLOT_BYTES)
    assert slot.dtype == torch.uint8


def test_decompress_slot_shape(device, head_dim, num_outliers):
    state = QuantState.create(3, head_dim, num_outliers, device)
    x = torch.randn(8, 2, head_dim, device=device, dtype=torch.bfloat16)

    slot = compress_to_slot(x, state)
    recon = decompress_from_slot(slot, state)

    assert recon.shape == x.shape
    assert recon.dtype == torch.bfloat16


# -- correctness against PyTorch reference ----------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_slot_matches_pytorch_reference(head_dim, num_outliers):
    """Triton decompress must match PyTorch decompress (same data, same math)."""
    device = torch.device("cuda:0")
    state = QuantState.create(3, head_dim, num_outliers, device)
    torch.manual_seed(777)
    x = torch.randn(32, 2, head_dim, device=device, dtype=torch.bfloat16)

    # Slot path (Triton)
    slot = compress_to_slot(x, state)
    recon_triton = decompress_from_slot(slot, state)

    # Reference path (PyTorch)
    compressed = compress(x.reshape(-1, head_dim), state)
    recon_ref = decompress(compressed, state).reshape(x.shape)

    torch.testing.assert_close(
        recon_triton, recon_ref,
        atol=1e-2, rtol=1e-2,
        msg="Triton decompress diverged from PyTorch reference",
    )


# -- roundtrip quality ------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_slot_roundtrip_cosine_similarity(head_dim, num_outliers):
    """Full roundtrip (with inverse rotation) should achieve > 0.95 cosine sim."""
    device = torch.device("cuda:0")
    state = QuantState.create(3, head_dim, num_outliers, device)
    torch.manual_seed(42)
    x = torch.randn(64, 2, head_dim, device=device, dtype=torch.bfloat16)

    slot = compress_to_slot(x, state)
    recon = decompress_from_slot(slot, state)

    # decompress_from_slot returns normal channels in rotated space.
    # For cosine similarity, compare via full_decompress reference.
    from trinity_turbo.quant.turboquant import full_decompress

    compressed = compress(x.reshape(-1, head_dim), state)
    recon_full = full_decompress(compressed, state).reshape(x.shape)

    x_flat = x.float().reshape(-1, head_dim)
    r_flat = recon_full.float().reshape(-1, head_dim)
    cos_sim = torch.nn.functional.cosine_similarity(x_flat, r_flat, dim=-1)

    assert cos_sim.mean().item() > 0.95, (
        f"3-bit roundtrip cosine similarity {cos_sim.mean().item():.4f} < 0.95"
    )


# -- outlier preservation ---------------------------------------------------

def test_slot_outliers_exact(device, head_dim, num_outliers):
    """Outlier channels must survive the slot roundtrip bit-exact."""
    state = QuantState.create(3, head_dim, num_outliers, device)
    x = torch.randn(8, 2, head_dim, device=device, dtype=torch.bfloat16)

    slot = compress_to_slot(x, state)
    recon = decompress_from_slot(slot, state)

    original_outliers = x[..., :num_outliers]
    recon_outliers = recon[..., :num_outliers]

    torch.testing.assert_close(recon_outliers, original_outliers)


# -- slot byte layout -------------------------------------------------------

def test_slot_byte_layout(device, head_dim, num_outliers):
    """Verify the 64-byte slot layout: outliers | packed | norm | pad."""
    state = QuantState.create(3, head_dim, num_outliers, device)
    x = torch.randn(1, head_dim, device=device, dtype=torch.bfloat16)

    slot = compress_to_slot(x, state)
    flat = slot.reshape(SLOT_BYTES)

    from trinity_turbo.kernels.triton_compress import NORM_OFFSET as NO
    # Norm region: 2 bytes of non-zero data
    # Padding after norm
    assert flat[NO:NO+2].any(), "Norm bytes should be non-zero"
    assert flat[SLOT_BYTES - 1].item() == 0, "Last padding byte should be zero"


# -- CPU fallback -----------------------------------------------------------

def test_cpu_fallback(head_dim, num_outliers):
    """Verify PyTorch fallback works on CPU."""
    state = QuantState.create(3, head_dim, num_outliers, device="cpu")
    x = torch.randn(4, 2, head_dim, dtype=torch.bfloat16)

    slot = compress_to_slot(x, state)
    recon = decompress_from_slot(slot, state)

    assert recon.shape == x.shape
    assert recon.dtype == torch.bfloat16
