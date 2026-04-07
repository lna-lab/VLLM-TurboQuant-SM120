"""Micro-benchmark: isolate TQ4 overhead per component.

Simulates 1 decode step for 60 layers, measuring:
  1. compress_to_slot (KV cache write)
  2. apply_rotation (Q pre-rotate)
  3. tq4_unified_attention (Triton kernel)
  4. apply_inverse_rotation (output post-rotate)
  5. vLLM baseline: unified_attention (Triton FP8)
"""
import math
import time
import torch
from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET, PACKED_OFFSET, SLOT_BYTES, compress_to_slot,
)
from trinity_turbo.kernels.triton_tq4_unified_attention import tq4_unified_attention
from trinity_turbo.quant.rotation import apply_rotation, apply_inverse_rotation
from trinity_turbo.quant.turboquant import QuantState

# Also import vLLM's standard unified_attention for comparison
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

DEVICE = "cuda"
NUM_LAYERS = 60
WARMUP = 5
ITERS = 50

# Trinity config (per TP rank)
NUM_KV_HEADS = 2      # 8 total / TP=4
NUM_Q_HEADS = 16       # 64 total / TP=4
HEAD_DIM = 128
BLOCK_SIZE = 16
BATCH_SIZE = 1         # decode: 1 query per seq
CONTEXT_LEN = 256      # tokens already in cache


def make_tq4_cache(state, num_seqs=1):
    """Pre-fill a paged TQ4 cache."""
    num_blocks = (CONTEXT_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_cache = torch.zeros(
        num_blocks * num_seqs, 2, BLOCK_SIZE, NUM_KV_HEADS, SLOT_BYTES,
        dtype=torch.uint8, device=DEVICE,
    )
    # Fill with random compressed data
    for b in range(num_blocks):
        for t in range(BLOCK_SIZE):
            k = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            v = torch.randn(1, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            kv_cache[b, 0, t] = compress_to_slot(k, state)
            kv_cache[b, 1, t] = compress_to_slot(v, state)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    seq_lens = torch.tensor([CONTEXT_LEN], dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    return kv_cache, block_table, seq_lens, cu_seqlens_q


def make_fp8_cache(num_seqs=1):
    """Pre-fill a paged FP8 cache."""
    num_blocks = (CONTEXT_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_cache = torch.randn(
        num_blocks * num_seqs, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
        device=DEVICE, dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn(
        num_blocks * num_seqs, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
        device=DEVICE, dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    seq_lens = torch.tensor([CONTEXT_LEN], dtype=torch.int32, device=DEVICE)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    return k_cache, v_cache, block_table, seq_lens, cu_seqlens_q


def bench(fn, label, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"  {label}: {ms:.3f} ms")
    return ms


def main():
    state = QuantState.create(bits=4, head_dim=HEAD_DIM, num_outliers=8, device=DEVICE)
    inv_sqrt_d = 1.0 / math.sqrt(state.normal_dim)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Pre-build caches
    print(f"Building caches (context={CONTEXT_LEN}, layers={NUM_LAYERS})...")
    kv_cache, bt, sl, cu = make_tq4_cache(state)
    k_fp8, v_fp8, bt_fp8, sl_fp8, cu_fp8 = make_fp8_cache()

    # Inputs
    key = torch.randn(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    value = torch.randn(BATCH_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    query = torch.randn(BATCH_SIZE, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    output = torch.zeros(BATCH_SIZE, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    k_scale = torch.ones(1, NUM_KV_HEADS, device=DEVICE, dtype=torch.float32)
    v_scale = torch.ones(1, NUM_KV_HEADS, device=DEVICE, dtype=torch.float32)

    key_cache_tq = kv_cache[:, 0]
    val_cache_tq = kv_cache[:, 1]

    print(f"\n=== Per-component timing ({ITERS} iters, context={CONTEXT_LEN}) ===\n")

    # 1. compress_to_slot (KV write — called 60× per decode step)
    def do_compress():
        compress_to_slot(key, state)
        compress_to_slot(value, state)
    t_compress = bench(do_compress, "compress_to_slot (1 layer, K+V)")

    # 2. apply_rotation (Q pre-rotate)
    def do_rotate():
        q_normal = query[..., state.num_outliers:].float()
        apply_rotation(q_normal, state.sign_flips)
    t_rotate = bench(do_rotate, "apply_rotation (Q, 1 layer)")

    # 3. TQ4 attention kernel
    def do_tq4_attn():
        q = query.clone()
        tq4_unified_attention(
            q=q, k_cache=key_cache_tq, v_cache=val_cache_tq, out=output,
            cu_seqlens_q=cu, seqused_k=sl, softmax_scale=scale,
            window_size=(-1, -1), block_table=bt, centroids=state.centroids,
            inv_sqrt_d=inv_sqrt_d, num_outliers=state.num_outliers,
            packed_off=PACKED_OFFSET, norm_off=NORM_OFFSET,
        )
    t_tq4 = bench(do_tq4_attn, "tq4_unified_attention (1 layer)")

    # 4. apply_inverse_rotation (output post-rotate)
    def do_inv_rotate():
        out_normal = output[..., state.num_outliers:].float()
        apply_inverse_rotation(out_normal, state.sign_flips)
    t_inv = bench(do_inv_rotate, "apply_inverse_rotation (1 layer)")

    # 5. FP8 unified_attention (vLLM Triton baseline)
    def do_fp8_attn():
        unified_attention(
            q=query, k=k_fp8, v=v_fp8, out=output,
            cu_seqlens_q=cu_fp8, max_seqlen_q=1,
            seqused_k=sl_fp8, max_seqlen_k=CONTEXT_LEN,
            softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt_fp8, softcap=0,
            q_descale=None, k_descale=k_scale, v_descale=v_scale,
        )
    t_fp8 = bench(do_fp8_attn, "unified_attention FP8 (1 layer)")

    # Summary: extrapolate to 60 layers
    print(f"\n=== Extrapolated to {NUM_LAYERS} layers (1 decode step) ===\n")
    tq4_total = (t_compress + t_rotate + t_tq4 + t_inv) * NUM_LAYERS
    fp8_total = t_fp8 * NUM_LAYERS
    print(f"  TQ4 total: {tq4_total:.1f} ms  "
          f"(compress={t_compress*NUM_LAYERS:.1f} "
          f"rotate={t_rotate*NUM_LAYERS:.1f} "
          f"attn={t_tq4*NUM_LAYERS:.1f} "
          f"inv_rot={t_inv*NUM_LAYERS:.1f})")
    print(f"  FP8 total: {fp8_total:.1f} ms  (attn only)")
    print(f"  TQ4/FP8:   {tq4_total/fp8_total:.2f}×")
    print(f"\n  Estimated tok/s (attention-only, no GEMM):")
    print(f"    FP8: {1000/fp8_total:.0f} tok/s")
    print(f"    TQ4: {1000/tq4_total:.0f} tok/s")

    # Breakdown
    print(f"\n=== Breakdown (% of TQ4 total) ===\n")
    for name, t in [("compress", t_compress), ("rotate_q", t_rotate),
                     ("tq4_attn", t_tq4), ("inv_rotate", t_inv)]:
        pct = t * NUM_LAYERS / tq4_total * 100
        print(f"  {name}: {pct:.1f}%  ({t*NUM_LAYERS:.1f} ms)")


if __name__ == "__main__":
    main()
