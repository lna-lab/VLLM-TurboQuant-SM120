"""Bit-packing utilities for quantized KV cache storage.

Packs quantization indices (2-4 bit) into uint8 byte arrays.
Supports 2-bit (4 values/byte), 3-bit (8 values/3 bytes), and 4-bit (2 values/byte).
"""

from __future__ import annotations

import torch


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack quantization indices into compact byte representation.

    Args:
        indices: Quantization level indices, shape (..., num_elements).
                 Values must be in [0, 2^bits).
        bits: Bits per index (2, 3, or 4).

    Returns:
        Packed uint8 tensor. Shape (..., packed_size) where:
          - 2-bit: packed_size = ceil(num_elements / 4)
          - 3-bit: packed_size = ceil(num_elements * 3 / 8)
          - 4-bit: packed_size = ceil(num_elements / 2)
    """
    if bits == 4:
        return _pack_4bit(indices)
    elif bits == 2:
        return _pack_2bit(indices)
    elif bits == 3:
        return _pack_3bit(indices)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def unpack_indices(packed: torch.Tensor, bits: int, num_elements: int) -> torch.Tensor:
    """Unpack quantization indices from compact byte representation.

    Args:
        packed: Packed uint8 tensor from pack_indices().
        bits: Bits per index (2, 3, or 4).
        num_elements: Original number of elements.

    Returns:
        Unpacked indices tensor, shape (..., num_elements).
    """
    if bits == 4:
        return _unpack_4bit(packed, num_elements)
    elif bits == 2:
        return _unpack_2bit(packed, num_elements)
    elif bits == 3:
        return _unpack_3bit(packed, num_elements)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def packed_size(num_elements: int, bits: int) -> int:
    """Calculate packed byte size for given element count and bit width."""
    return (num_elements * bits + 7) // 8


# -- 4-bit packing: 2 values per byte --

def _pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    *batch, n = indices.shape
    indices = indices.to(torch.uint8)
    # Pad to even
    if n % 2 != 0:
        indices = torch.nn.functional.pad(indices, (0, 1))
        n += 1
    indices = indices.view(*batch, n // 2, 2)
    packed = (indices[..., 0] << 4) | indices[..., 1]
    return packed


def _unpack_4bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    *batch, p = packed.shape
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    unpacked = torch.stack([hi, lo], dim=-1).view(*batch, -1)
    return unpacked[..., :num_elements]


# -- 2-bit packing: 4 values per byte --

def _pack_2bit(indices: torch.Tensor) -> torch.Tensor:
    *batch, n = indices.shape
    indices = indices.to(torch.uint8)
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad > 0:
        indices = torch.nn.functional.pad(indices, (0, pad))
    n_padded = indices.shape[-1]
    indices = indices.view(*batch, n_padded // 4, 4)
    packed = (
        (indices[..., 0] << 6)
        | (indices[..., 1] << 4)
        | (indices[..., 2] << 2)
        | indices[..., 3]
    )
    return packed


def _unpack_2bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    *batch, p = packed.shape
    v0 = (packed >> 6) & 0x03
    v1 = (packed >> 4) & 0x03
    v2 = (packed >> 2) & 0x03
    v3 = packed & 0x03
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1).view(*batch, -1)
    return unpacked[..., :num_elements]


# -- 3-bit packing: 8 values per 3 bytes --

def _pack_3bit(indices: torch.Tensor) -> torch.Tensor:
    *batch, n = indices.shape
    indices = indices.to(torch.uint8)
    # Pad to multiple of 8
    pad = (8 - n % 8) % 8
    if pad > 0:
        indices = torch.nn.functional.pad(indices, (0, pad))
    n_padded = indices.shape[-1]
    indices = indices.view(*batch, n_padded // 8, 8)

    # Pack 8 × 3-bit values into 3 bytes (24 bits)
    # Byte 0: v0[2:0] v1[2:0] v2[2:1]
    # Byte 1: v2[0] v3[2:0] v4[2:0] v5[2]
    # Byte 2: v5[1:0] v6[2:0] v7[2:0]
    b0 = (indices[..., 0] << 5) | (indices[..., 1] << 2) | (indices[..., 2] >> 1)
    b1 = ((indices[..., 2] & 1) << 7) | (indices[..., 3] << 4) | (indices[..., 4] << 1) | (indices[..., 5] >> 2)
    b2 = ((indices[..., 5] & 3) << 6) | (indices[..., 6] << 3) | indices[..., 7]

    packed = torch.stack([b0, b1, b2], dim=-1).view(*batch, -1)
    return packed


def _unpack_3bit(packed: torch.Tensor, num_elements: int) -> torch.Tensor:
    *batch, p = packed.shape
    packed = packed.view(*batch, -1, 3)

    b0, b1, b2 = packed[..., 0], packed[..., 1], packed[..., 2]
    v0 = (b0 >> 5) & 0x07
    v1 = (b0 >> 2) & 0x07
    v2 = ((b0 & 0x03) << 1) | ((b1 >> 7) & 0x01)
    v3 = (b1 >> 4) & 0x07
    v4 = (b1 >> 1) & 0x07
    v5 = ((b1 & 0x01) << 2) | ((b2 >> 6) & 0x03)
    v6 = (b2 >> 3) & 0x07
    v7 = b2 & 0x07

    unpacked = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).view(*batch, -1)
    return unpacked[..., :num_elements]
