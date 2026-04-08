"""Phase 5a A/B comparison: old (non-coalesced K) vs new (transposed K).

Runs both kernels on identical data and compares bandwidth.
"""
import math
import time

import torch

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET, PACKED_OFFSET, SLOT_BYTES,
)
from trinity_turbo.quant.rotation import apply_rotation
from trinity_turbo.quant.turboquant import QuantState

# Import both versions
from trinity_turbo.kernels.triton_tq4_unified_attention import (
    tq4_unified_attention as attn_phase5,
)
from trinity_turbo.kernels.triton_tq4_unified_attention_old import (
    tq4_unified_attention as attn_phase4a,
)

DEVICE = "cuda"


def make_data(num_seqs, seq_len, num_kv_heads=2, num_heads=4, head_dim=128, block_size=16):
    state = QuantState.create(bits=4, head_dim=head_dim, num_outliers=8, device=DEVICE)
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = blocks_per_seq * num_seqs

    kv_cache = torch.randint(
        0, 256, (total_blocks, 2, block_size, num_kv_heads, SLOT_BYTES),
        dtype=torch.uint8, device=DEVICE,
    )
    block_table = torch.zeros(num_seqs, blocks_per_seq, dtype=torch.int32, device=DEVICE)
    for s in range(num_seqs):
        block_table[s] = torch.arange(
            s * blocks_per_seq, (s + 1) * blocks_per_seq, dtype=torch.int32, device=DEVICE,
        )
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=DEVICE)

    q = torch.randn(num_seqs, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    q[..., state.num_outliers:] = apply_rotation(
        q[..., state.num_outliers:].float(), state.sign_flips,
    ).to(torch.bfloat16)

    cu_seqlens_q = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=DEVICE)
    return q, kv_cache, block_table, seq_lens, cu_seqlens_q, state


def bench_fn(attn_fn, q, kv_cache, block_table, seq_lens, cu_seqlens_q, state, warmup=30, iters=300):
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    out = torch.zeros_like(q)
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

    for _ in range(warmup):
        attn_fn(**kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        attn_fn(**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_us = elapsed / iters * 1e6
    num_seqs = q.shape[0]
    seq_len = seq_lens[0].item()
    kv_bytes = num_seqs * seq_len * SLOT_BYTES * 2 * 2
    bandwidth_gbs = kv_bytes / (elapsed / iters) / 1e9

    return latency_us, bandwidth_gbs


def main():
    print("=" * 72)
    print("Phase 5a A/B Comparison — K Load Coalescing")
    print("=" * 72)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    configs = [
        (1, 1024),
        (8, 1024),
        (16, 1024),
        (32, 1024),
        (64, 1024),
        (16, 128),
        (16, 512),
        (16, 2048),
        (16, 4096),
        (64, 4096),
    ]

    print(f"{'batch':>6} {'seq_len':>8} {'Phase4a_us':>11} {'Phase5_us':>11} "
          f"{'4a_GB/s':>9} {'5_GB/s':>9} {'speedup':>8}")
    print("-" * 72)

    for num_seqs, seq_len in configs:
        q, kv_cache, block_table, seq_lens, cu_seqlens_q, state = \
            make_data(num_seqs, seq_len)

        lat_old, bw_old = bench_fn(
            attn_phase4a, q, kv_cache, block_table, seq_lens, cu_seqlens_q, state,
        )
        lat_new, bw_new = bench_fn(
            attn_phase5, q, kv_cache, block_table, seq_lens, cu_seqlens_q, state,
        )

        speedup = lat_old / lat_new if lat_new > 0 else 0
        print(f"{num_seqs:>6} {seq_len:>8} {lat_old:>11.1f} {lat_new:>11.1f} "
              f"{bw_old:>9.1f} {bw_new:>9.1f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
