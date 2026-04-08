"""TILE_SIZE tuning benchmark — find optimal tile size for SM120 Blackwell.

Now that Triton 3.6 fixes SM120 CUDA graph, we can explore tile sizes
beyond the conservative TILE_SIZE=16 used in Phase 4a.
"""
import math
import time
import torch
import triton

from trinity_turbo.kernels.triton_compress import (
    NORM_OFFSET, PACKED_OFFSET, SLOT_BYTES,
)
from trinity_turbo.kernels.triton_tq4_unified_attention import (
    kernel_tq4_unified_attention_2d,
)
from trinity_turbo.quant.rotation import apply_rotation
from trinity_turbo.quant.turboquant import QuantState

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
    out = torch.zeros(num_seqs, num_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    return q, kv_cache, block_table, seq_lens, cu_seqlens_q, out, state


def bench_tile_size(tile_size, num_seqs, seq_len, warmup=30, iters=300):
    q, kv_cache, block_table, seq_lens, cu_seqlens_q, out, state = \
        make_data(num_seqs, seq_len)

    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]
    block_size = 16
    scale = 1.0 / math.sqrt(128)
    num_query_heads = 4
    num_kv_heads = 2
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = 128

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    total_num_q_blocks = num_seqs // BLOCK_Q + num_seqs
    HEAD_SIZE_PADDED = triton.next_power_of_2(head_size)

    grid = (total_num_q_blocks, num_kv_heads)

    kwargs = dict(
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        scale=scale,
        centroids_ptr=state.centroids,
        inv_sqrt_d=1.0 / math.sqrt(state.normal_dim),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        stride_k0=key_cache.stride(0),
        stride_k1=key_cache.stride(1),
        stride_k2=key_cache.stride(2),
        stride_v0=value_cache.stride(0),
        stride_v1=value_cache.stride(1),
        stride_v2=value_cache.stride(2),
        query_start_len_ptr=cu_seqlens_q,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        SLIDING_WINDOW=0,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        TQ_NUM_OUTLIERS=state.num_outliers,
        TQ_PACKED_OFF=PACKED_OFFSET,
        TQ_NORM_OFF=NORM_OFFSET,
    )

    # Warmup
    for _ in range(warmup):
        kernel_tq4_unified_attention_2d[grid](**kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        kernel_tq4_unified_attention_2d[grid](**kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_us = elapsed / iters * 1e6
    kv_bytes = num_seqs * seq_len * SLOT_BYTES * 2 * 2
    bandwidth_gbs = kv_bytes / (elapsed / iters) / 1e9
    return latency_us, bandwidth_gbs


def main():
    print("=" * 60)
    print("TILE_SIZE Tuning — SM120 Blackwell")
    print("=" * 60)

    # Try TILE_SIZE = 8, 16, 32, 64
    tile_sizes = [8, 16, 32]

    for num_seqs, seq_len in [(16, 1024), (64, 1024), (64, 4096)]:
        print(f"\n--- batch={num_seqs}, seq_len={seq_len} ---")
        print(f"{'TILE':>6} {'lat_us':>10} {'BW_GB/s':>10} {'vs_16':>8}")

        base_lat = None
        for ts in tile_sizes:
            try:
                lat, bw = bench_tile_size(ts, num_seqs, seq_len)
                if ts == 16:
                    base_lat = lat
                ratio = base_lat / lat if base_lat else 0
                print(f"{ts:>6} {lat:>10.1f} {bw:>10.1f} {ratio:>7.2f}x")
            except Exception as e:
                print(f"{ts:>6} FAILED: {e}")


if __name__ == "__main__":
    main()
