# VLLM-TurboQuant-SM120

TurboQuant 4-bit KV cache compression for vLLM, targeting long-context MoE models on NVIDIA Blackwell (SM120) GPUs.

## Results

**3x FP8 throughput at high concurrency. 37.5% less KV cache VRAM.**

**Model:** Trinity-Large-Thinking-W4A16 (398B MoE, TP=4)
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96 GB each)
**Context:** 256K tokens, CUDA graphs enabled, Triton 3.6

### Throughput (128 output tokens)

| Concurrency | FP8 Baseline | TQ4 Phase 4a | TQ4 / FP8 |
|:-----------:|:------------:|:------------:|:---------:|
| 1           | 138.3 tok/s  | 104.4 tok/s  | 76%       |
| 8           | 674.3        | 635.5        | 94%       |
| 16          | **1146.3** (FP8 max) | 1197.3 | **104%**  |
| 32          | OOM          | 2477.0       | **216%**  |
| 48          | OOM          | 2988.9       | **261%**  |
| 64          | OOM          | **3432.4**   | **299%**  |

FP8 tops out at 16 concurrent requests (KV cache OOM). TQ4's 1.6x memory compression allows 64+ concurrent requests, reaching **3x FP8 peak throughput**.

### VRAM Savings

| Metric | FP8 (128 bytes/head) | TQ4 (80 bytes/head) | Savings |
|:-------|:--------------------:|:--------------------:|:-------:|
| Bytes per token per KV head | 128 | 80 | **37.5%** |
| VRAM for 1.28M tokens | 36.6 GiB | 22.8 GiB | **13.8 GiB freed** |

### KV Cache Capacity

| Config       | KV Cache Tokens | Max Parallel @ 256K |
|:------------:|:---------------:|:-------------------:|
| FP8 baseline | 1,279,232       | ~18x                |
| TQ4 4-bit    | 2,040,448       | ~28x                |
| Improvement  | **1.60x**       | **1.56x**           |

### Quality

- Cosine similarity vs reference: 0.9999
- Outlier channels: bit-exact preservation
- 67+ tests passing (unit + integration + end-to-end)

## How It Works

TurboQuant compresses KV cache vectors from 128 bytes (FP8) to 80 bytes (TQ4) per head per token:

1. **Outlier preservation** — 8 highest-variance channels kept at bf16 (16 bytes)
2. **Walsh-Hadamard rotation** — distributes remaining 120 channels into near-Gaussian distribution
3. **Lloyd-Max 4-bit quantization** — optimal scalar quantization (16 centroids)
4. **Compact packing** — 120 channels at 4 bits = 60 bytes, plus 2-byte fp16 norm

Slot layout (80 bytes per token per KV head):

```
[0,16)  outliers bf16    16 bytes
[16,76) packed 4-bit     60 bytes
[76,78) L2 norm fp16      2 bytes
[78,80) padding            2 bytes
```

At decode time, decompression happens **inside the attention kernel** — each tile loads compressed uint8 slots and reconstructs bf16 K/V vectors in registers before `tl.dot`. No intermediate buffers, no extra HBM bandwidth.

## Architecture

Phase 4a hybrid: CUDA native kernels for compute, Triton for attention, bare CUDA graph (no torch.compile).

| Component | Implementation | Role |
|:----------|:--------------|:-----|
| KV cache write | CUDA native (`cuda_compress.cu`) | Fused WHT + quantize + pack + scatter. 1 kernel per (token, head). |
| Attention | Triton (`triton_tq4_unified_attention.py`) | Fork of vLLM's `kernel_unified_attention_2d` with in-tile TQ4 decompress. |
| Q/output rotation | CUDA native (`cuda_rotation.cu`) | Fused WHT butterfly with `__syncthreads()`. 1 kernel per vector. |

All three are CUDA graph compatible on SM120 (Blackwell).

### Why CUDA native for WHT?

The Walsh-Hadamard butterfly requires cross-element synchronization at each of 7 steps (128 elements = 4 warps). Triton lacks warp-level barriers (`__syncthreads()`), causing data races in cross-warp butterfly steps. CUDA native kernels solve this cleanly.

We also evaluated [HadaCore](https://github.com/pytorch-labs/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform) (Meta's Tensor Core accelerated WHT). It builds and runs correctly on SM120, but at head_dim=128 the simple butterfly WHT is 3.5x faster. HadaCore's Tensor Core overhead only pays off at dim >= 2048.

### Why no torch.compile?

Phase 4a disables `torch.compile` (`-cc.mode none`) and uses bare CUDA graph (`-cc.cudagraph_mode full`). This is intentional:

- torch.compile's inductor generates cuBLAS calls that are incompatible with SM120
- Removing inductor overhead **increased** throughput by 17% (2927 → 3432 tok/s)
- CUDA graph alone provides zero kernel launch overhead without inductor's compilation cost

### Triton 3.6 SM120 CUDA Graph Fix

As of Triton 3.6.0 + CUDA 12.8, the SM120 CUDA graph capture issue with `tl.dot()` scratch space is **resolved**. Earlier Triton versions allocated scratch memory at addresses that differed between capture and replay on SM12x. This is no longer an issue — our test suite confirms all three kernel types (simple dot, TQ4-style decompress+dot, and the full attention kernel) capture and replay correctly.

## Quick Start

```bash
pip install -e .

TRINITY_TURBO_ENABLED=1 \
vllm serve /path/to/Trinity-Large-Thinking-W4A16 \
    --tensor-parallel-size 4 \
    --attention-backend CUSTOM \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.92
```

### Configuration

All settings via environment variables with `TRINITY_TURBO_` prefix:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `TRINITY_TURBO_ENABLED` | `1` | Master switch |
| `TRINITY_TURBO_BITS` | `4` | Quantization bits (2, 3, or 4) |
| `TRINITY_TURBO_NUM_OUTLIER_CHANNELS` | `8` | Channels preserved at bf16 |

## Requirements

- vLLM >= 0.19.0
- Python >= 3.12
- CUDA >= 12.8 (SM120 Blackwell optimized)
- PyTorch >= 2.10
- Triton >= 3.6 (required for SM120 CUDA graph support)

## Project Structure

```
src/trinity_turbo/
  plugin.py                         vLLM plugin entry point
  config.py                         Environment-based configuration
  backend/
    attention_backend.py             Custom attention backend (KV cache shape)
    attention_impl.py                Phase 3 attention forward pass
    cache_spec.py                    Compressed page size specs
  kernels/
    cuda_compress.cu                 CUDA fused compress + scatter
    cuda_rotation.cu                 CUDA fused WHT rotation
    triton_tq4_unified_attention.py  TQ4 tiled decode attention
    triton_compress.py               PyTorch compress (reference)
    fast_wht.py                      Fast WHT (double-buffer)
  quant/
    turboquant.py                    TurboQuant core (compress/decompress)
    codebook.py                      Lloyd-Max codebook generation
    packing.py                       Bit-packing utilities
    rotation.py                      WHT rotation (reference)
tests/
  test_tq4_unified_attn.py          Phase 3 kernel tests
  test_turboquant.py                Core quantization tests
  ...                               67 tests total
benchmarks/
  bench_quick.py                    Throughput benchmark
  profile_overhead.py               Per-component profiling
```

## Optimization Journey

```
Phase 2+:  0.2 tok/s    fused kernel (structural defeat)
Phase 3:   9.7 tok/s    unified tiled attention              x49
Phase 3+:  46.2          + CUDA graph                        x4.8
Phase 3a:  64.6          + fast WHT (double-buffer)           x1.4
Phase 3d: 104.7          + CUDA native compress/rotate        x1.6
Phase 3e: 2926.8 @64     + high-concurrency serving          (FP8 2.55x)
Phase 4a: 3432.4 @64     + no torch.compile + graph only     (FP8 2.99x)
──────────────────────────────────────────────────────────────
Total: 0.2 → 3432.4 = 17,162x improvement
```

## Roadmap

- [x] Phase 1 — TurboQuant core + vLLM plugin skeleton
- [x] Phase 2 — Compressed KV cache + Triton decompress (35x concurrency)
- [x] Phase 2+ — Fused Triton decode attention + CUDA graph
- [x] Phase 3 — Tiled unified attention + CUDA native compress/rotate (104.7 tok/s)
- [x] Phase 3e — High-concurrency serving (2927 tok/s @64, FP8 2.55x)
- [x] Phase 4a — Hybrid strategy: no torch.compile + CUDA graph only (3432 tok/s @64, FP8 2.99x)
- [ ] Phase 5 — Compute density: speculative decoding, dynamic KV re-quantization

## SM120 (Blackwell) Lessons Learned

See [docs/SM120_LESSONS.md](docs/SM120_LESSONS.md) for detailed findings on optimizing Triton + CUDA kernels for RTX PRO 6000 Blackwell.

## References

- [TurboQuant: Extreme KV Cache Quantization (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [HadaCore: Tensor Core Accelerated Hadamard Transform (Meta, arXiv:2412.08832)](https://arxiv.org/abs/2412.08832)
- [vLLM](https://github.com/vllm-project/vllm)
- [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant)

## License

Apache 2.0

---

Built at [Lna-Lab](https://github.com/lna-lab).
