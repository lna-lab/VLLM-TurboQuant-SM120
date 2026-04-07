"""TrinityTurbo attention implementation — Phase 2.

Extends TritonAttentionImpl with TurboQuant KV cache compression.
All layers (sliding window and global) use compressed uint8 format.

Phase 2 strategy: "decompress-all-blocks"
  - Cache write: bf16 K/V → TurboQuant compress → uint8 scatter to pages
  - Cache read: decompress ALL blocks → fixed-size bf16 buffer (CUDA graph safe)
  - Q: pre-rotate normal channels via WHT (preserves dot products)
  - Attention: standard Triton paged attention on decompressed bf16
  - Output: inverse-rotate normal channels (undo V rotation)
"""

from __future__ import annotations

import logging

import torch

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionImpl,
    TritonAttentionMetadata,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

from trinity_turbo.config import get_global_config
from trinity_turbo.kernels.triton_compress import SLOT_BYTES, compress_to_slot
from trinity_turbo.kernels.triton_decompress import decompress_from_slot
from trinity_turbo.quant.rotation import apply_inverse_rotation, apply_rotation
from trinity_turbo.quant.turboquant import QuantState

logger = logging.getLogger(__name__)


class TrinityTurboAttentionImpl(TritonAttentionImpl):
    """Phase 2 attention with TurboQuant-compressed KV cache.

    KV cache shape: [num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES=64]
    with dtype fp8 (1 byte/element) → 64 bytes per head per token.
    Standard fp8 uses 128 bytes → 2× memory reduction.

    CUDA graph compatible: all operations have fixed output shapes.
    Decompression buffers are shared across layers via class variable.
    """

    # Shared decompression buffers across all layer instances.
    # Only one layer executes at a time, so sharing is safe.
    _shared_dec_k: torch.Tensor | None = None
    _shared_dec_v: torch.Tensor | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        config = get_global_config()
        self.quant_state = QuantState.create(
            bits=config.bits,
            head_dim=self.head_size,
            num_outliers=config.num_outlier_channels,
            device="cuda",
        )

        # Force bf16 attention path — our decompress outputs bf16, not fp8.
        self.kv_cache_dtype = "auto"

        # Lazily created bf16 descale (identity)
        self._descale_ones: torch.Tensor | None = None

        if not hasattr(TrinityTurboAttentionImpl, "_logged_phase2"):
            logger.info(
                "TrinityTurboAttentionImpl Phase 2 active: "
                "heads=%d, kv_heads=%d, head_size=%d, slot_bytes=%d, bits=%d",
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                SLOT_BYTES,
                config.bits,
            )
            TrinityTurboAttentionImpl._logged_phase2 = True

    # ------------------------------------------------------------------
    # Cache write: compress and scatter
    # ------------------------------------------------------------------

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Compress new K/V and scatter to the uint8 KV cache pages."""
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

    # ------------------------------------------------------------------
    # Attention: decompress ALL → rotate Q → paged attention
    # ------------------------------------------------------------------

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not supported "
                "by TrinityTurboAttentionImpl"
            )

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens
        device = kv_cache.device

        # ── View cache as uint8 ──────────────────────────────────
        kv_u8 = kv_cache.view(torch.uint8)
        num_blocks = kv_u8.shape[0]
        block_size = kv_u8.shape[2]

        # ── Decompress ALL blocks → fixed-size bf16 ──────────────
        # No dynamic ops (unique/gather/remap) → CUDA graph compatible.
        # Shared buffers across layers: only one layer active at a time.
        buf_shape = (num_blocks, block_size, self.num_kv_heads, self.head_size)
        cls = TrinityTurboAttentionImpl
        if cls._shared_dec_k is None or cls._shared_dec_k.shape != buf_shape:
            cls._shared_dec_k = torch.empty(
                buf_shape, dtype=torch.bfloat16, device=device,
            )
            cls._shared_dec_v = torch.empty(
                buf_shape, dtype=torch.bfloat16, device=device,
            )

        # K: [num_blocks, block_size, num_kv_heads, SLOT_BYTES] → bf16
        flat_k = kv_u8[:, 0].reshape(-1, SLOT_BYTES)
        dec_k_flat = decompress_from_slot(flat_k, self.quant_state)
        cls._shared_dec_k[:] = dec_k_flat.reshape(buf_shape)

        # V
        flat_v = kv_u8[:, 1].reshape(-1, SLOT_BYTES)
        dec_v_flat = decompress_from_slot(flat_v, self.quant_state)
        cls._shared_dec_v[:] = dec_v_flat.reshape(buf_shape)

        # ── Pre-rotate Q normal channels ─────────────────────────
        q = self._rotate_query(query[:num_actual_tokens])

        # ── Descale = identity (our data is bf16, not FP8) ───────
        cu_seqlens_q = attn_metadata.query_start_loc
        num_seqs = cu_seqlens_q.shape[0] - 1
        descale_shape = (num_seqs, self.num_kv_heads)
        if (
            self._descale_ones is None
            or self._descale_ones.shape != descale_shape
            or self._descale_ones.device != device
        ):
            self._descale_ones = torch.ones(
                descale_shape, dtype=torch.float32, device=device,
            )

        # ── Ensure bf16 for Q ────────────────────────────────────
        if q.dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)

        # ── Standard Triton paged attention ──────────────────────
        # Uses original block_table (no remapping needed).
        out_slice = output[:num_actual_tokens]
        unified_attention(
            q=q,
            k=cls._shared_dec_k,
            v=cls._shared_dec_v,
            out=out_slice,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=self._descale_ones,
            v_descale=self._descale_ones,
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=None,  # bf16 output, no fp8 quantization
            mm_prefix_range=None,
        )

        # ── Inverse-rotate output normal channels ────────────────
        st = self.quant_state
        out_normal = out_slice[..., st.num_outliers:].float()
        out_inv = apply_inverse_rotation(out_normal, st.sign_flips)
        out_slice[..., st.num_outliers:] = out_inv.to(out_slice.dtype)

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rotate_query(self, q: torch.Tensor) -> torch.Tensor:
        """Pre-rotate Q normal channels by WHT for TurboQuant attention."""
        st = self.quant_state
        q_bf16 = q.to(torch.bfloat16)
        q_out = q_bf16.clone()
        q_normal = q_bf16[..., st.num_outliers:].float()
        q_out[..., st.num_outliers:] = apply_rotation(
            q_normal, st.sign_flips,
        ).to(torch.bfloat16)
        return q_out
