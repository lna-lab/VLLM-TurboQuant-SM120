# VLLM-TurboQuant-SM120

TurboQuant 4-bit KV cache compression for vLLM, targeting long-context MoE models on Blackwell GPUs.

## Results

**Model:** Trinity-Large-Thinking-W4A16 (398B MoE, TP=4)
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96 GB each)
**Context:** 256K tokens, CUDA graphs enabled

### Throughput (128 output tokens)

| Concurrency | FP8 Baseline (tok/s) | TQ4 (tok/s) | TQ4/FP8 |
|:-----------:|:--------------------:|:------------:|:--------:|
| 1           | 138.3                | 104.7        | 76%      |
| 4           | 393.6                | 339.0        | 86%      |
| 8           | 674.3                | 659.9        | 98%      |
| 16          | 1146.3               | **1236.8**   | **108%** |

At 16 concurrent requests, TQ4 **surpasses** FP8 throughput while using 1.6x less KV cache memory.

### KV Cache Capacity

| Config       | KV Cache Tokens | Max Parallel @ 256K |
|:------------:|:---------------:|:-------------------:|
| FP8 baseline | 1,279,232       | ~18x                |
| TQ4 4-bit    | 2,040,448       | ~28x                |
| Improvement  | **1.60x**       | **1.56x**           |

### Quality

- Cosine similarity vs reference: 0.9999
- Outlier channels: bit-exact preservation
- 67 tests passing (unit + integration + end-to-end)

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

The plugin consists of three kernel layers:

| Component | Implementation | Role |
|:----------|:--------------|:-----|
| KV cache write | CUDA native (`cuda_compress.cu`) | Fused WHT + quantize + pack + scatter. 1 kernel per (token, head). |
| Attention | Triton (`triton_tq4_unified_attention.py`) | Fork of vLLM's `kernel_unified_attention_2d` with in-tile TQ4 decompress. |
| Q/output rotation | CUDA native (`cuda_rotation.cu`) | Fused WHT butterfly with `__syncthreads()`. 1 kernel per vector. |

All three are CUDA graph compatible.

### Why CUDA native for WHT?

The Walsh-Hadamard butterfly requires cross-element synchronization at each of 7 steps (128 elements = 4 warps). Triton lacks warp-level barriers (`__syncthreads()`), causing data races in cross-warp butterfly steps. CUDA native kernels solve this cleanly.

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
- Triton >= 3.0

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

## Roadmap

- [x] Phase 1 — TurboQuant core + vLLM plugin skeleton
- [x] Phase 2 — Compressed KV cache + Triton decompress (35x concurrency)
- [x] Phase 2+ — Fused Triton decode attention + CUDA graph
- [x] Phase 3 — Tiled unified attention + CUDA native compress/rotate (104.7 tok/s)
- [ ] Phase 4 — WHT as matrix-vector multiply (full Triton, no CUDA native)
- [ ] Phase 5 — Gate-based eviction + cross-layer reconstruction

## References

- [TurboQuant: Extreme KV Cache Quantization (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- [vLLM](https://github.com/vllm-project/vllm)
- [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant)

## License

Apache 2.0

---

Built at [Lna-Lab](https://github.com/lna-lab).
