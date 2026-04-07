"""TrinityTurbo attention implementation — Phase 3: unified tiled attention.

Uses vLLM's tiled attention structure (BLOCK_M x TILE_SIZE) with TQ4
in-register decompress at the K/V load paths. This replaces the Phase 2+
fused kernel which was structurally slow (1 program = 1 seq×head, serial scan).

Phase 3 flow:
  1. do_kv_cache_update: compress K/V -> uint8 slots (unchanged)
  2. forward:
     a. Pre-rotate Q normal channels (Walsh-Hadamard)
     b. Call tq4_unified_attention (tiled, parallel, tl.dot)
     c. Inverse-rotate output normal channels
"""

from __future__ import annotations

import logging
import math

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from trinity_turbo.config import get_global_config
from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET,
    PACKED_OFFSET,
    SLOT_BYTES,
    compress_to_slot,
)
from trinity_turbo.kernels.cuda_compress_wrapper import fused_compress_scatter
from trinity_turbo.kernels.triton_tq4_unified_attention import (
    tq4_unified_attention,
)
from trinity_turbo.kernels.cuda_rotation_wrapper import (
    cuda_apply_rotation,
    cuda_apply_inverse_rotation,
)
from trinity_turbo.quant.turboquant import QuantState

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(TritonAttentionImpl):
    """Phase 3 attention with tiled TQ4 decompress inside unified_attention.

    Uses vLLM's BLOCK_M x TILE_SIZE tiled parallelism + tl.dot matmuls.
    TQ4 decompress happens in-register per tile — zero extra HBM buffers.
    CUDA graph compatible.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = get_global_config()
        self.quant_state = QuantState.create(
            bits=config.bits,
            head_dim=self.head_size,
            num_outliers=config.num_outlier_channels,
            device="cuda",
        )
        # Use "auto" so base class doesn't try fp8 cast on our uint8 cache
        self.kv_cache_dtype = "auto"
        self._inv_sqrt_d = 1.0 / math.sqrt(self.quant_state.normal_dim)

        if not hasattr(TrinityTurboAttentionImpl, "_logged"):
            logger.info(
                "TrinityTurboAttentionImpl Phase 3 (tiled unified): "
                "heads=%d, kv_heads=%d, head_size=%d, bits=%d, "
                "slot_bytes=%d, outliers=%d",
                self.num_heads, self.num_kv_heads, self.head_size,
                config.bits, SLOT_BYTES, config.num_outlier_channels,
            )
            TrinityTurboAttentionImpl._logged = True

    # ------------------------------------------------------------------
    # KV cache write: compress to TQ4 slots (unchanged from Phase 2+)
    # ------------------------------------------------------------------

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        kv_u8 = kv_cache.view(torch.uint8)

        # CUDA fused compress + scatter
        fused_compress_scatter(key, kv_u8, self.quant_state, slot_mapping, kv_dim=0)
        fused_compress_scatter(value, kv_u8, self.quant_state, slot_mapping, kv_dim=1)

    # ------------------------------------------------------------------
    # Forward: tiled TQ4 unified attention
    # ------------------------------------------------------------------

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata: TritonAttentionMetadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        assert output is not None

        if attn_metadata is None:
            return output.fill_(0)

        # Encoder attention: fall back to base class (no TQ4 cache)
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        st = self.quant_state

        # --- Unbind K/V cache as uint8 ---
        kv_u8 = kv_cache.view(torch.uint8)
        key_cache = kv_u8[:, 0, :, :, :]   # (num_blocks, block_size, num_kv_heads, slot_bytes)
        value_cache = kv_u8[:, 1, :, :, :]

        # --- Pre-rotate Q ---
        q = query[:num_actual_tokens].to(torch.bfloat16)
        q_normal = q[..., st.num_outliers:].float()
        q[..., st.num_outliers:] = cuda_apply_rotation(
            q_normal, st.sign_flips,
        ).to(torch.bfloat16)

        # --- TQ4 unified attention ---
        out_slice = output[:num_actual_tokens]
        tq4_unified_attention(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            out=out_slice,
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            softmax_scale=self.scale,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            centroids=st.centroids,
            inv_sqrt_d=self._inv_sqrt_d,
            num_outliers=st.num_outliers,
            packed_off=PACKED_OFFSET,
            norm_off=NORM_OFFSET,
        )

        # --- Inverse-rotate output normal channels ---
        out_normal = out_slice[..., st.num_outliers:].float()
        out_inv = cuda_apply_inverse_rotation(out_normal, st.sign_flips)
        out_slice[..., st.num_outliers:] = out_inv.to(out_slice.dtype)

        return output
