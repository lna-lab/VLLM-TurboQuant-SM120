"""TrinityTurbo attention implementation — Phase 2+ fused.

Fused TurboQuant decode attention: decompresses 3-bit KV cache directly
inside the Triton attention kernel. Zero extra memory buffers.
CUDA graph compatible.
"""

from __future__ import annotations

import logging

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)

from trinity_turbo.config import get_global_config
from trinity_turbo.kernels.triton_compress import SLOT_BYTES, compress_to_slot
from trinity_turbo.kernels.triton_fused_attn import fused_tq_decode_attention
from trinity_turbo.quant.rotation import apply_inverse_rotation, apply_rotation
from trinity_turbo.quant.turboquant import QuantState

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(TritonAttentionImpl):
    """Phase 2+ attention with fused TurboQuant decompress+attention.

    Zero bf16 buffers. CUDA graph compatible.
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
        self.kv_cache_dtype = "auto"

        if not hasattr(TrinityTurboAttentionImpl, "_logged"):
            logger.info(
                "TrinityTurboAttentionImpl Phase 2+ (fused paged): "
                "heads=%d, kv_heads=%d, head_size=%d, bits=%d",
                self.num_heads, self.num_kv_heads, self.head_size, config.bits,
            )
            TrinityTurboAttentionImpl._logged = True

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        kv_u8 = kv_cache.view(torch.uint8)
        block_size = kv_u8.shape[2]
        k_slots = compress_to_slot(key, self.quant_state)
        v_slots = compress_to_slot(value, self.quant_state)
        block_idx = slot_mapping // block_size
        offset = slot_mapping % block_size
        kv_u8[block_idx, 0, offset] = k_slots
        kv_u8[block_idx, 1, offset] = v_slots

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        assert output is not None
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        st = self.quant_state

        # Pre-rotate Q
        q = query[:num_actual_tokens].to(torch.bfloat16)
        q_normal = q[..., st.num_outliers:].float()
        q[..., st.num_outliers:] = apply_rotation(q_normal, st.sign_flips).to(torch.bfloat16)

        # Sliding window
        sw = 0
        if self.sliding_window is not None and self.sliding_window[0] > 0:
            sw = self.sliding_window[0] + 1

        # Fused decode attention
        out_slice = output[:num_actual_tokens]
        fused_tq_decode_attention(
            query=q,
            kv_cache=kv_cache,
            centroids=st.centroids,
            quant_state=st,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            query_start_loc=attn_metadata.query_start_loc,
            output=out_slice,
            softmax_scale=self.scale,
            sliding_window=sw,
            num_queries_per_kv=self.num_queries_per_kv,
        )

        # Inverse-rotate output normal channels
        out_normal = out_slice[..., st.num_outliers:].float()
        out_inv = apply_inverse_rotation(out_normal, st.sign_flips)
        out_slice[..., st.num_outliers:] = out_inv.to(out_slice.dtype)

        return output
