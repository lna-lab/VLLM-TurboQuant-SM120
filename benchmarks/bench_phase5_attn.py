"""Phase 5a micro-benchmark: K transposed load coalescing impact.

Measures attention kernel latency at various sequence lengths to quantify
the bandwidth improvement from coalesced K loads (byte offset as fast dim).

Usage:
    CUDA_VISIBLE_DEVICES=4 python benchmarks/bench_phase5_attn.py
"""
import math
import time

import torch

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET, PACKED_OFFSET, SLOT_BYTES, compress_to_slot,
)
from trinity_turbo.kernels.triton_tq4_unified_attention import tq4_unified_attention
from trinity_turbo.quant.rotation import apply_rotation
from trinity_turbo.quant.turboquant import QuantState

DEVICE = "cuda"


def make_batch_decode_data(
    num_seqs, seq_len, num_kv_heads=2, num_heads=4, head_dim=128, block_size=16,
):
    """Create batch decode scenario: num_seqs concurrent 1-token decodes."""
    state = QuantState.create(bits=4, head_dim=head_dim, num_outliers=8, device=DEVICE)
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = blocks_per_seq * num_seqs

    # Allocate paged KV cache — fill with random compressed data
    kv_cache = torch.randint(
        0, 256, (total_blocks, 2, block_size, num_kv_heads, SLOT_BYTES),
        dtype=torch.uint8, device=DEVICE,
    )

    # Block table: each sequence gets contiguous blocks
    block_table = torch.zeros(num_seqs, blocks_per_seq, dtype=torch.int32, device=DEVICE)
    for s in range(num_seqs):
        block_table[s] = torch.arange(
            s * blocks_per_seq, (s + 1) * blocks_per_seq, dtype=torch.int32, device=DEVICE,
        )

    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=DEVICE)

    # One query per sequence
    queries = torch.randn(num_seqs, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    q = queries.clone()
    q[..., state.num_outliers:] = apply_rotation(
        q[..., state.num_outliers:].float(), state.sign_flips,
    ).to(torch.bfloat16)

    cu_seqlens_q = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=DEVICE)
    out = torch.zeros(num_seqs, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)

    return q, kv_cache, block_table, seq_lens, cu_seqlens_q, out, state


def bench_batch_decode(num_seqs, seq_len, warmup=20, iters=200):
    """Benchmark batch decode: num_seqs concurrent 1-token attention calls."""
    q, kv_cache, block_table, seq_lens, cu_seqlens_q, out, state = \
        make_batch_decode_data(num_seqs, seq_len)
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    scale = 1.0 / math.sqrt(128)

    kwargs = dict(
        q=q, k_cache=key_cache, v_cache=value_cache, out=out,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seq_lens,
        softmax_scale=scale, window_size=(-1, -1),
        block_table=block_table, centroids=state.centroids,
        inv_sqrt_d=1.0 / math.sqrt(state.normal_dim),
        num_outliers=state.num_outliers,
        packed_off=PACKED_OFFSET, norm_off=NORM_OFFSET,
    )

    # Warmup
    for _ in range(warmup):
        tq4_unified_attention(**kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        tq4_unified_attention(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_us = elapsed / iters * 1e6

    # Total KV bytes read: num_seqs * seq_len * 80 bytes * 2 (K+V) * num_kv_heads
    kv_bytes = num_seqs * seq_len * SLOT_BYTES * 2 * 2
    bandwidth_gbs = kv_bytes / (elapsed / iters) / 1e9

    return latency_us, bandwidth_gbs


def main():
    print("=" * 70)
    print("Phase 5a Attention Kernel Benchmark — K Transposed Coalesced Load")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: batch decode, 4 heads, 2 KV heads, head_dim=128, TQ4 80-byte slots")
    print()

    # Test 1: Fixed seq_len=1024, vary concurrency
    print("--- Batch size scaling (seq_len=1024) ---")
    seq_len = 1024
    batch_sizes = [1, 4, 8, 16, 32, 64]
    print(f"{'batch':>6} {'latency_us':>12} {'KV_BW_GB/s':>12} {'util_%':>8}")
    print("-" * 44)

    for bs in batch_sizes:
        lat, bw = bench_batch_decode(bs, seq_len)
        util = bw / 1100 * 100
        print(f"{bs:>6} {lat:>12.1f} {bw:>12.1f} {util:>8.1f}")

    # Test 2: Fixed batch=16, vary seq_len
    print()
    print("--- Sequence length scaling (batch=16) ---")
    batch = 16
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    print(f"{'seq_len':>8} {'latency_us':>12} {'KV_BW_GB/s':>12} {'util_%':>8}")
    print("-" * 44)

    for sl in seq_lengths:
        lat, bw = bench_batch_decode(batch, sl)
        util = bw / 1100 * 100
        print(f"{sl:>8} {lat:>12.1f} {bw:>12.1f} {util:>8.1f}")


if __name__ == "__main__":
    main()
