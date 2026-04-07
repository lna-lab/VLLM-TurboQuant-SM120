"""Compress bf16 KV vectors into packed uint8 slots for KV cache storage.

Slot layout (4-bit, 80 bytes per token per KV head):
  [0, 16)   8 outlier channels as bf16     (16 bytes)
  [16, 76)  120 normal channels, 4-bit packed indices  (60 bytes)
  [76, 78)  L2 norm as fp16                (2 bytes)
  [78, 80)  zero padding                   (2 bytes)

4-bit packing: 2 values per byte, trivial unpack (>>4, &0xF).
This enables vectorized load in fused Triton attention kernels.
"""

from __future__ import annotations

import torch

from trinity_turbo.quant.turboquant import QuantState, compress

SLOT_BYTES = 80
OUTLIER_BYTES = 16   # 8 bf16 = 16 bytes
PACKED_OFFSET = 16
PACKED_BYTES = 60    # 120 channels × 4 bits / 8 = 60 bytes
NORM_OFFSET = 76     # 16 + 60


def compress_to_slot(
    x: torch.Tensor,
    state: QuantState,
) -> torch.Tensor:
    """Compress bf16 KV vectors into packed uint8 slots.

    Args:
        x: KV tensor, shape (..., head_dim), dtype bf16/fp16.
        state: QuantState for this layer.

    Returns:
        Packed slots, shape (..., SLOT_BYTES), dtype uint8.
    """
    *batch_shape, hd = x.shape
    assert hd == state.head_dim

    flat = x.reshape(-1, hd)
    n = flat.shape[0]
    device = x.device

    compressed = compress(flat, state)

    slot = torch.zeros(n, SLOT_BYTES, dtype=torch.uint8, device=device)

    # Outliers: bf16 [n, 8] -> uint8 [n, 16]
    slot[:, :OUTLIER_BYTES] = compressed.outliers.contiguous().view(torch.uint8)

    # Packed indices: uint8 [n, packed_size]
    packed = compressed.packed_indices
    slot[:, PACKED_OFFSET:PACKED_OFFSET + packed.shape[-1]] = packed

    # Norm: fp16 [n, 1] -> uint8 [n, 2]
    slot[:, NORM_OFFSET:NORM_OFFSET + 2] = compressed.norms.contiguous().view(torch.uint8)

    return slot.reshape(*batch_shape, SLOT_BYTES)
