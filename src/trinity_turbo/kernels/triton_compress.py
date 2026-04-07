"""Compress bf16 KV vectors into packed uint8 slots for KV cache storage.

Uses the tested PyTorch TurboQuant pipeline, then assembles the result
into a flat uint8 slot layout compatible with vLLM's paged KV cache.

Slot layout (64 bytes per token per KV head):
  [0, 16)   8 outlier channels as bf16     (16 bytes)
  [16, 61)  120 normal channels, 3-bit packed indices  (45 bytes)
  [61, 63)  L2 norm as fp16                (2 bytes)
  [63, 64)  zero padding                   (1 byte)

Phase 2+: quantize + pack will be fused into a single Triton kernel.
"""

from __future__ import annotations

import torch

from trinity_turbo.quant.turboquant import QuantState, compress

SLOT_BYTES = 64
OUTLIER_BYTES = 16   # 8 bf16 = 16 bytes
PACKED_OFFSET = 16
NORM_OFFSET = 61


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
    assert hd == state.head_dim, f"head_dim mismatch: expected {state.head_dim}, got {hd}"

    flat = x.reshape(-1, hd)
    n = flat.shape[0]
    device = x.device

    # --- Reuse the tested PyTorch pipeline ---
    compressed = compress(flat, state)

    # --- Assemble slot ---
    slot = torch.zeros(n, SLOT_BYTES, dtype=torch.uint8, device=device)

    # Outliers: bf16 [n, 8] -> uint8 [n, 16]
    slot[:, :OUTLIER_BYTES] = compressed.outliers.contiguous().view(torch.uint8)

    # Packed indices: uint8 [n, packed_size]
    packed = compressed.packed_indices
    slot[:, PACKED_OFFSET:PACKED_OFFSET + packed.shape[-1]] = packed

    # Norm: fp16 [n, 1] -> uint8 [n, 2]
    slot[:, NORM_OFFSET:NORM_OFFSET + 2] = compressed.norms.contiguous().view(torch.uint8)

    return slot.reshape(*batch_shape, SLOT_BYTES)
