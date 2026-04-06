"""Tests for TurboQuant compress/decompress pipeline."""

import torch
import pytest

from trinity_turbo.quant.turboquant import QuantState, compress, decompress, full_decompress


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_compress_decompress_shape(bits, device, head_dim, num_outliers):
    """Compressed and decompressed tensors should have correct shapes."""
    state = QuantState.create(bits, head_dim, num_outliers, device)
    x = torch.randn(16, 2, head_dim, device=device, dtype=torch.bfloat16)

    compressed = compress(x, state)
    reconstructed = decompress(compressed, state)

    assert reconstructed.shape == x.shape
    assert reconstructed.dtype == torch.bfloat16


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_cosine_similarity(bits, device, head_dim, num_outliers):
    """Compression quality: cosine similarity should be > 0.95 for 3-4 bit.

    NOTE: decompress() returns normal channels in rotated space (WHT applied).
    For correct similarity measurement, we use full_decompress() which applies
    inverse rotation, or we compare in rotated space.
    """
    state = QuantState.create(bits, head_dim, num_outliers, device)
    torch.manual_seed(123)
    x = torch.randn(64, 2, head_dim, device=device, dtype=torch.bfloat16)

    compressed = compress(x, state)
    reconstructed = full_decompress(compressed, state)

    # Cosine similarity per vector
    x_flat = x.float().view(-1, head_dim)
    r_flat = reconstructed.float().view(-1, head_dim)
    cos_sim = torch.nn.functional.cosine_similarity(x_flat, r_flat, dim=-1)
    mean_sim = cos_sim.mean().item()

    if bits >= 3:
        assert mean_sim > 0.95, f"{bits}-bit cosine similarity {mean_sim:.4f} < 0.95"
    else:
        # 2-bit is more lossy but should still be reasonable
        assert mean_sim > 0.85, f"{bits}-bit cosine similarity {mean_sim:.4f} < 0.85"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_outliers_preserved(bits, device, head_dim, num_outliers):
    """Outlier channels should be perfectly preserved."""
    state = QuantState.create(bits, head_dim, num_outliers, device)
    x = torch.randn(8, 2, head_dim, device=device, dtype=torch.bfloat16)

    compressed = compress(x, state)
    reconstructed = decompress(compressed, state)

    original_outliers = x[..., state.outlier_indices]
    recon_outliers = reconstructed[..., state.outlier_indices]

    torch.testing.assert_close(recon_outliers, original_outliers)


def test_slot_bytes(device, head_dim, num_outliers):
    """Verify slot_bytes calculation for 3-bit Trinity config."""
    state = QuantState.create(3, head_dim, num_outliers, device)
    # 8 outliers * 2 bytes (bf16) = 16
    # 120 normal channels * 3 bits / 8 = 45 bytes
    # 1 norm * 2 bytes (fp16) = 2
    # Total = 63 bytes
    assert state.slot_bytes == 63


def test_compression_ratio(device, head_dim, num_outliers):
    """Verify compression ratio for 3-bit."""
    state = QuantState.create(3, head_dim, num_outliers, device)
    fp8_bytes = head_dim * 1  # FP8 baseline: 1 byte per element
    ratio = fp8_bytes / state.slot_bytes
    assert ratio > 1.5, f"3-bit compression ratio {ratio:.2f}x vs FP8 should be > 1.5x"


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_norms_positive(bits, device, head_dim, num_outliers):
    """Stored norms should always be positive."""
    state = QuantState.create(bits, head_dim, num_outliers, device)
    x = torch.randn(32, 2, head_dim, device=device, dtype=torch.bfloat16)

    compressed = compress(x, state)
    assert (compressed.norms > 0).all()
