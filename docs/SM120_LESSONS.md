# SM120 (Blackwell) Optimization Lessons

Hard-won findings from optimizing Triton + CUDA kernels on NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0). These lessons apply to any memory-bound Triton kernel running on SM120.

Tested with: Triton 3.6.0, CUDA 12.8, PyTorch 2.10, driver 580.126.

## 1. The L2 Cache Changes Everything

**SM120's 96 MB L2 cache absorbs scattered byte-level loads from small data structures.**

We expected that loading 80-byte KV cache slots byte-by-byte (scattered `tl.load` with non-contiguous offsets) would waste 99.5% of HBM bandwidth, since each 1-byte load fetches a 128-byte cache line. We implemented coalesced loads by transposing the access pattern (making byte offset the fast/inner dimension instead of the token index).

**Result: 4% slower, not faster.**

Why: The 80-byte slot fits within a single 128-byte cache line. The first byte access loads the entire slot into L2. All subsequent byte accesses within that slot are L2 hits. The L2 cache effectively converts scattered byte loads into a single cache-line fetch per token — exactly what our "optimization" was trying to achieve manually.

The `tl.trans()` call needed before `tl.dot()` added MMA register layout conversion overhead that outweighed any memory access improvement.

**Takeaway:** On SM120, don't optimize memory access patterns for data structures smaller than 128 bytes. The L2 cache handles it. Focus your optimization effort on **compute density** (doing more useful work per byte read) instead.

**When this lesson does NOT apply:**
- Data structures larger than one cache line (> 128 bytes per token)
- Access patterns that span many cache lines per token
- Scenarios where L2 capacity is exhausted (e.g., millions of active tokens)

## 2. Triton 3.6 Fixes SM120 CUDA Graph Capture

**The `tl.dot()` scratch space bug is resolved in Triton 3.6.0.**

In earlier Triton versions, `tl.dot()` on SM12x allocated scratch memory at addresses that differed between CUDA graph capture and replay, causing silent data corruption or crashes. This forced us to use workarounds (CUDA native kernels for all compute, or disabling CUDA graph).

As of Triton 3.6.0 + CUDA 12.8, all three patterns work correctly:

```python
# All of these capture and replay correctly on SM120:
# 1. Simple tl.dot
# 2. uint8 → bf16 cast → tl.dot (quantized KV decompress pattern)
# 3. Full paged attention kernel with tl.dot inside a tile loop
```

Test: `tests/test_triton_sm120_cudagraph.py`

**Takeaway:** If you were avoiding Triton kernels in CUDA graphs on Blackwell, upgrade to Triton >= 3.6.0.

## 3. torch.compile Hurts on SM120 (Use Bare CUDA Graph)

**Disabling torch.compile and using CUDA graph alone is 17% faster.**

```
With torch.compile + CUDA graph:  2927 tok/s @64 concurrent
Without torch.compile, graph only: 3432 tok/s @64 concurrent  (+17%)
```

vLLM launch flags:
```
-cc.mode none              # disable torch.compile (inductor)
-cc.cudagraph_mode full    # keep CUDA graph
```

Why: torch.compile's inductor backend generates cuBLAS calls that have compatibility issues on SM120. Even without the compatibility issue, inductor's compilation and dispatch overhead exceeds its optimization benefit for this workload (small per-token compute, dominated by memory access).

**Takeaway:** Profile before assuming torch.compile helps. For memory-bound inference workloads on SM120, bare CUDA graph often wins.

## 4. HadaCore Tensor Core WHT: Not for Small Dimensions

**Meta's [HadaCore](https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform) is 3.5x slower than a butterfly WHT at head_dim=128.**

| Batch Size | CUDA Native Butterfly | HadaCore Tensor Core | Ratio |
|:----------:|:---------------------:|:--------------------:|:-----:|
| 1          | 8.1 us                | 37.4 us              | 0.22x |
| 256        | 9.0 us                | 29.0 us              | 0.31x |
| 4096       | 8.1 us                | 28.2 us              | 0.29x |

HadaCore uses `mma.sync` Tensor Core instructions for the Hadamard transform. The MMA setup overhead (register marshaling, warp synchronization) dominates at small dimensions. The break-even point is likely around dim=512-1024.

HadaCore builds on SM120 with `arch=compute_120,code=sm_120`. No source changes needed — it takes the SM90 code path (`#else` branch in the `__CUDA_ARCH__` conditional), which uses `cp.async.cg.shared.global` without L2 cache hints.

**Takeaway:** Use butterfly WHT for head_dim <= 256. Consider HadaCore for head_dim >= 2048 (Llama-3 style models).

## 5. TILE_SIZE=16 Is the SM120 Sweet Spot

For our TQ4 paged attention kernel (128-dim heads, 4-bit packed KV cache):

| TILE_SIZE | Result |
|:---------:|:-------|
| 8         | Fails: `tl.dot` requires inner dimension K >= 16 |
| **16**    | **Optimal.** 110 GB/s at batch=64, 190 us latency |
| 32        | 2x slower. Register pressure halves SM occupancy |

The constraint is `tl.dot(P, V)` where P is `(BLOCK_M, TILE_SIZE)` — TILE_SIZE is the matmul inner dimension (K), which must be >= 16 for Triton's MMA lowering.

At TILE_SIZE=32, the kernel uses too many registers (decompress + accumulate for 32 tokens per tile), reducing occupancy below the threshold needed to hide memory latency.

## 6. Bandwidth Characteristics

Phase 4a attention kernel bandwidth on a single RTX PRO 6000:

```
Batch size scaling (seq_len=1024, KV bandwidth):
  batch= 1:    1.7 GB/s  (0.2% of 1,100 GB/s peak)
  batch= 8:   13.8 GB/s  (1.3%)
  batch=16:   27.6 GB/s  (2.5%)
  batch=32:   55.0 GB/s  (5.0%)
  batch=64:  110.0 GB/s  (10.0%)
```

Key observation: **Latency is constant (~190 us) regardless of batch size.** Bandwidth scales linearly with batch. The kernel is latency-bound at low batch sizes and transitions to bandwidth-bound at high concurrency.

At 10% peak HBM utilization with 64 concurrent requests, the bottleneck is the low arithmetic intensity of the decompression (few FLOPs per byte loaded). This is fundamental to any quantized KV cache attention scheme, not specific to TQ4.

## Summary

| Optimization | Expected Impact | Actual Impact | Root Cause |
|:-------------|:---------------:|:-------------:|:-----------|
| Coalesced K loads via transpose | +61x bandwidth | **-4%** | L2 cache absorbs scattered loads |
| HadaCore Tensor Core WHT | Faster WHT | **-3.5x** | MMA overhead at dim=128 |
| torch.compile + CUDA graph | Faster overall | **-17%** | Inductor overhead > optimization |
| Bare CUDA graph (Triton 3.6) | Same as before | **Works!** | Triton 3.6 fixes scratch alloc |
| TILE_SIZE=32 | Better throughput | **-2x** | Register pressure kills occupancy |

The recurring theme: **SM120's hardware (large L2, fast SMs) compensates for "unoptimized" access patterns, but punishes approaches that add overhead (tl.trans, inductor, larger tiles).** The winning strategy is minimal overhead: bare CUDA graph, simple memory access patterns, right-sized tiles.
