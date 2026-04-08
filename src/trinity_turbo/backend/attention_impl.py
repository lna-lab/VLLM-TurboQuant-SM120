"""TrinityTurbo attention — Phase 4a: hybrid (CUDA native graph + Triton eager).

Triton on SM120 Blackwell has CUDA graph capture issues (illegal memory access).
SGLang #19799 confirms this is a known Triton/SM12x ecosystem issue.

Phase 4a hybrid strategy:
  - compress: CUDA native (Phase 3e, CUDA graph proven)
  - rotation: CUDA native for graph capture, Triton mat-vec for eager rotation
  - attention: Triton (Phase 3, CUDA graph proven)

When Triton SM120 support matures → Phase 4b: all-Triton.
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
)
from trinity_turbo.kernels.cuda_compress_wrapper import (
    fused_compress_scatter,
    _ensure_slot_mapping_buf,
)
from trinity_turbo.kernels.triton_tq4_unified_attention import (
    tq4_unified_attention,
)
from trinity_turbo.kernels.cuda_rotation_wrapper import (
    cuda_apply_rotation,
    cuda_apply_inverse_rotation,
    _ensure_bufs as _ensure_rotation_bufs,
)
from trinity_turbo.quant.turboquant import QuantState

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(TritonAttentionImpl):
    """Phase 4a: hybrid CUDA native (graph) + Triton (eager).

    CUDA native compress + rotation for CUDA graph compatibility.
    Triton TQ4 attention (proven graph-safe since Phase 3).
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
        self._inv_sqrt_d = 1.0 / math.sqrt(self.quant_state.normal_dim)

        # Pre-allocate all buffers for CUDA graph compatibility
        _ensure_slot_mapping_buf(torch.device("cuda"))
        _ensure_rotation_bufs(self.quant_state.normal_dim, torch.device("cuda"))

        if not hasattr(TrinityTurboAttentionImpl, "_logged"):
            logger.info(
                "TrinityTurboAttentionImpl Phase 4a (hybrid: CUDA native graph + Triton attn): "
                "heads=%d, kv_heads=%d, head_size=%d, bits=%d, "
                "slot_bytes=%d, outliers=%d",
                self.num_heads, self.num_kv_heads, self.head_size,
                config.bits, SLOT_BYTES, config.num_outlier_channels,
            )
            TrinityTurboAttentionImpl._logged = True

    # ------------------------------------------------------------------
    # KV cache write: CUDA native fused compress + scatter (graph-safe)
    # ------------------------------------------------------------------

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        kv_u8 = kv_cache.view(torch.uint8)

        fused_compress_scatter(key, kv_u8, self.quant_state, slot_mapping, kv_dim=0)
        fused_compress_scatter(value, kv_u8, self.quant_state, slot_mapping, kv_dim=1)

    # ------------------------------------------------------------------
    # Forward: CUDA native rotation + Triton TQ4 attention (graph-safe)
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

        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        st = self.quant_state

        # --- Unbind K/V cache as uint8 ---
        kv_u8 = kv_cache.view(torch.uint8)
        key_cache = kv_u8[:, 0, :, :, :]
        value_cache = kv_u8[:, 1, :, :, :]

        # --- Pre-rotate Q (CUDA native WHT, graph-safe) ---
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

        # --- Inverse-rotate output (CUDA native WHT, graph-safe) ---
        out_normal = out_slice[..., st.num_outliers:].float()
        out_inv = cuda_apply_inverse_rotation(out_normal, st.sign_flips)
        out_slice[..., st.num_outliers:] = out_inv.to(out_slice.dtype)

        return output
