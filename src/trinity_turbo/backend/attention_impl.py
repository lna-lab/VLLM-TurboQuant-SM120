"""TrinityTurbo attention implementation — Phase 4: all-Triton, no CUDA JIT.

Phase 4 replaces all CUDA native kernels with pure Triton:
  - compress: triton_fused_compress_v2 (WHT mat-vec + quantize + pack + scatter)
  - rotation: triton_hadamard (WHT as matrix-vector multiply via tl.dot)
  - attention: triton_tq4_unified_attention (unchanged from Phase 3)

Benefits over Phase 3:
  - No CUDA JIT compilation → reboot-safe, no race conditions
  - WHT via Tensor Core mat-vec → faster on Blackwell
  - Single Triton kernel for compress + scatter → fewer kernel launches
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
from trinity_turbo.kernels.triton_fused_compress_v2 import (
    triton_fused_compress_scatter,
)
from trinity_turbo.kernels.triton_tq4_unified_attention import (
    tq4_unified_attention,
)
from trinity_turbo.kernels.triton_hadamard import (
    build_signed_hadamard,
    triton_apply_rotation,
    triton_apply_inverse_rotation,
    _ensure_bufs as _ensure_rotation_bufs,
)
from trinity_turbo.kernels.triton_fused_compress_v2 import (
    _ensure_slot_buf,
)
from trinity_turbo.quant.turboquant import QuantState

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(TritonAttentionImpl):
    """Phase 4: all-Triton attention with mat-vec WHT.

    No CUDA native JIT required. Reboot-safe.
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

        # Build pre-signed Hadamard matrices (once at init)
        self.H_fwd, self.H_inv = build_signed_hadamard(
            self.quant_state.sign_flips, torch.device("cuda"),
        )

        # Pre-allocate buffers for CUDA graph compatibility
        _ensure_slot_buf(torch.device("cuda"))
        _ensure_rotation_bufs(self.quant_state.normal_dim, torch.device("cuda"))

        if not hasattr(TrinityTurboAttentionImpl, "_logged"):
            logger.info(
                "TrinityTurboAttentionImpl Phase 4 (all-Triton, mat-vec WHT): "
                "heads=%d, kv_heads=%d, head_size=%d, bits=%d, "
                "slot_bytes=%d, outliers=%d",
                self.num_heads, self.num_kv_heads, self.head_size,
                config.bits, SLOT_BYTES, config.num_outlier_channels,
            )
            TrinityTurboAttentionImpl._logged = True

    # ------------------------------------------------------------------
    # KV cache write: Triton fused compress + scatter (Phase 4)
    # ------------------------------------------------------------------

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        kv_u8 = kv_cache.view(torch.uint8)

        triton_fused_compress_scatter(
            key, kv_u8, self.quant_state, self.H_fwd, slot_mapping, kv_dim=0,
        )
        triton_fused_compress_scatter(
            value, kv_u8, self.quant_state, self.H_fwd, slot_mapping, kv_dim=1,
        )

    # ------------------------------------------------------------------
    # Forward: tiled TQ4 unified attention (Phase 4: Triton rotation)
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
        padded_dim = self.H_fwd.shape[0]

        # --- Unbind K/V cache as uint8 ---
        kv_u8 = kv_cache.view(torch.uint8)
        key_cache = kv_u8[:, 0, :, :, :]
        value_cache = kv_u8[:, 1, :, :, :]

        # --- Pre-rotate Q (Triton mat-vec WHT) ---
        q = query[:num_actual_tokens].to(torch.bfloat16)
        q_normal = q[..., st.num_outliers:].float()
        # Pad to padded_dim for mat-vec
        if q_normal.shape[-1] < padded_dim:
            q_padded = torch.nn.functional.pad(
                q_normal, (0, padded_dim - q_normal.shape[-1]),
            )
        else:
            q_padded = q_normal
        q_rotated = triton_apply_rotation(q_padded, self.H_fwd)
        q[..., st.num_outliers:] = q_rotated[..., :st.normal_dim].to(torch.bfloat16)

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

        # --- Inverse-rotate output (Triton mat-vec WHT) ---
        out_normal = out_slice[..., st.num_outliers:].float()
        if out_normal.shape[-1] < padded_dim:
            out_padded = torch.nn.functional.pad(
                out_normal, (0, padded_dim - out_normal.shape[-1]),
            )
        else:
            out_padded = out_normal
        out_inv = triton_apply_inverse_rotation(out_padded, self.H_inv)
        out_slice[..., st.num_outliers:] = out_inv[..., :st.normal_dim].to(out_slice.dtype)

        return output
