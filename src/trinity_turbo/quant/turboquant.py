"""Core TurboQuant encode/decode: PyTorch reference implementation.

This module provides the pure-PyTorch version of TurboQuant compression
and decompression. Used for correctness verification against Triton kernels
and as a fallback path.

Algorithm:
  1. Split outlier channels (kept as bf16) from normal channels
  2. Compute L2 norm of normal channels
  3. Normalize: x_normal / norm
  4. Apply Walsh-Hadamard rotation (via sign flips)
  5. Scalar quantize each coordinate using Lloyd-Max codebook
  6. Pack indices into compact bit representation

Decompression:
  1. Unpack indices
  2. Look up centroids from codebook
  3. Scale by stored norm
  4. Inverse rotation is NOT applied here — Q is pre-rotated instead

Reference: TurboQuant (ICLR 2026, arXiv 2504.19874).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from trinity_turbo.quant.codebook import get_codebook_tensors
from trinity_turbo.quant.packing import pack_indices, packed_size, unpack_indices
from trinity_turbo.quant.rotation import apply_inverse_rotation, apply_rotation, generate_sign_flips


@dataclass
class CompressedKV:
    """Compressed KV cache entry for a single head."""

    packed_indices: torch.Tensor  # uint8, shape (..., packed_bytes)
    norms: torch.Tensor  # float16, shape (..., 1)
    outliers: torch.Tensor  # bfloat16, shape (..., num_outliers)


@dataclass
class QuantState:
    """Per-layer quantization state, initialized once at model load."""

    bits: int
    head_dim: int
    num_outliers: int
    normal_dim: int  # head_dim - num_outliers
    sign_flips: torch.Tensor  # (normal_dim,) float32
    boundaries: torch.Tensor  # (2^bits - 1,) float32
    centroids: torch.Tensor  # (2^bits,) float32
    outlier_indices: torch.Tensor  # (num_outliers,) int64
    normal_indices: torch.Tensor  # (normal_dim,) int64
    slot_bytes: int  # Total bytes per token per KV head in packed format

    @classmethod
    def create(
        cls,
        bits: int,
        head_dim: int,
        num_outliers: int,
        device: torch.device | str = "cpu",
        seed: int = 42,
    ) -> QuantState:
        normal_dim = head_dim - num_outliers
        sign_flips = generate_sign_flips(normal_dim, seed=seed, device=device)
        boundaries, centroids = get_codebook_tensors(bits, device)

        # For now, outliers are the first num_outliers channels.
        # TODO: calibration-based outlier selection
        outlier_indices = torch.arange(num_outliers, device=device)
        normal_indices = torch.arange(num_outliers, head_dim, device=device)

        # Slot layout: [outlier_bf16 | packed_normal | norm_fp16]
        outlier_bytes = num_outliers * 2  # bf16
        normal_packed_bytes = packed_size(normal_dim, bits)
        norm_bytes = 2  # fp16
        slot_bytes = outlier_bytes + normal_packed_bytes + norm_bytes

        return cls(
            bits=bits,
            head_dim=head_dim,
            num_outliers=num_outliers,
            normal_dim=normal_dim,
            sign_flips=sign_flips,
            boundaries=boundaries,
            centroids=centroids,
            outlier_indices=outlier_indices,
            normal_indices=normal_indices,
            slot_bytes=slot_bytes,
        )


def compress(x: torch.Tensor, state: QuantState) -> CompressedKV:
    """Compress KV vectors using TurboQuant.

    Args:
        x: Input tensor of shape (..., head_dim) in bf16/fp16.
        state: Quantization state for this layer.

    Returns:
        CompressedKV with packed indices, norms, and outlier values.
    """
    # 1. Split outlier and normal channels
    outliers = x[..., state.outlier_indices].to(torch.bfloat16)
    normal = x[..., state.normal_indices].float()

    # 2. Compute L2 norm
    norms = torch.norm(normal, dim=-1, keepdim=True)
    norms = norms.clamp(min=1e-8)

    # 3. Normalize
    normalized = normal / norms

    # 4. Apply rotation (Walsh-Hadamard with sign flips)
    rotated = apply_rotation(normalized, state.sign_flips)

    # 5. Scale to unit variance for Lloyd-Max (N(0, 1/d) -> N(0, 1))
    # After rotation + normalization, coordinates are ~N(0, 1/d)
    d = state.normal_dim
    scaled = rotated * (d ** 0.5)

    # 6. Quantize using Lloyd-Max boundaries
    indices = torch.bucketize(scaled, state.boundaries).to(torch.uint8)

    # 7. Pack indices
    packed = pack_indices(indices, state.bits)

    return CompressedKV(
        packed_indices=packed,
        norms=norms.to(torch.float16),
        outliers=outliers,
    )


def full_decompress(compressed: CompressedKV, state: QuantState) -> torch.Tensor:
    """Fully decompress KV vectors including inverse rotation.

    This applies the inverse Walsh-Hadamard transform to recover vectors
    in the original space. Use for quality measurement and debugging.
    For attention, use decompress() + pre-rotated Q instead.
    """
    # 1. Unpack indices
    indices = unpack_indices(compressed.packed_indices, state.bits, state.normal_dim)

    # 2. Look up centroids
    centroids = state.centroids[indices.long()]

    # 3. Undo scaling
    d = state.normal_dim
    unscaled = centroids / (d ** 0.5)

    # 4. Inverse rotation
    # apply_rotation does: x * signs -> WHT (with 1/sqrt(d) normalization)
    # WHT is orthogonal: H * H^T = I when normalized by 1/sqrt(d)
    # So inverse of (sign_flip then WHT/sqrt(d)) is (WHT/sqrt(d) then sign_flip)
    # = (WHT * sign_flip * x) / d ... but apply_rotation does it in the right order
    # Actually: WHT with 1/sqrt(d) is self-inverse. sign_flip is self-inverse.
    # Composition: if f(x) = WHT(x * signs) / sqrt(d), then
    # f(f(x)) = WHT(WHT(x * signs) / sqrt(d) * signs) / sqrt(d) = x
    # Because WHT(signs * WHT(signs * x) / sqrt(d)) / sqrt(d)
    # = WHT(signs * WHT(signs * x)) / d = x (WHT is involutory when normalized)
    # So apply_rotation IS its own inverse. Just apply it again.
    inv_rotated = apply_inverse_rotation(unscaled, state.sign_flips)

    # 5. Scale by stored norm
    reconstructed_normal = inv_rotated * compressed.norms.float()

    # 6. Reassemble with outliers
    *batch_shape, _ = reconstructed_normal.shape
    output = torch.zeros(*batch_shape, state.head_dim, dtype=torch.bfloat16,
                          device=reconstructed_normal.device)
    output[..., state.outlier_indices] = compressed.outliers
    output[..., state.normal_indices] = reconstructed_normal.to(torch.bfloat16)

    return output


def decompress(compressed: CompressedKV, state: QuantState) -> torch.Tensor:
    """Decompress KV vectors from TurboQuant format.

    NOTE: This does NOT apply inverse rotation. The caller must pre-rotate
    Q by the same transform before computing attention.

    Args:
        compressed: CompressedKV from compress().
        state: Quantization state for this layer.

    Returns:
        Reconstructed tensor of shape (..., head_dim) in bfloat16.
        Normal channels are in rotated space (not original space).
    """
    # 1. Unpack indices
    indices = unpack_indices(compressed.packed_indices, state.bits, state.normal_dim)

    # 2. Look up centroids
    centroids = state.centroids[indices.long()]

    # 3. Undo scaling (N(0,1) -> N(0, 1/d))
    d = state.normal_dim
    unscaled = centroids / (d ** 0.5)

    # 4. Scale by stored norm
    reconstructed_normal = unscaled * compressed.norms.float()

    # 5. Reassemble with outliers
    *batch_shape, _ = reconstructed_normal.shape
    output = torch.zeros(*batch_shape, state.head_dim, dtype=torch.bfloat16,
                          device=reconstructed_normal.device)
    output[..., state.outlier_indices] = compressed.outliers
    output[..., state.normal_indices] = reconstructed_normal.to(torch.bfloat16)

    return output
