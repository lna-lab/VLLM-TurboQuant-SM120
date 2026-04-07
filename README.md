# VLLM-TurboQuant-SM120

Built for a very specific goal: buying more concurrency on RTX PRO 6000 Blackwell by compressing KV cache intelligently.
Layer-aware KV cache compression for Trinity-Large-Thinking on vLLM, optimized for SM120 Blackwell GPUs.

## What It Does

A vLLM plugin that compresses the KV cache of Trinity-Large-Thinking (398B MoE) using architecture-aware TurboQuant quantization:

- **45 sliding window layers**: passthrough (already bounded at 4096 tokens)
- **15 global attention layers**: TurboQuant 3-bit compression

This exploits Trinity's 3:1 sliding:global attention pattern to achieve significant memory reduction where it matters, without touching the layers that don't need it.

## Quick Start

```bash
pip install -e .

TRINITY_TURBO_ENABLED=1 \
TRINITY_TURBO_BITS=3 \
vllm serve /path/to/Trinity-Large-Thinking-W4A16 \
    --tensor-parallel-size 4 \
    --attention-backend CUSTOM \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```

## Requirements

- vLLM >= 0.19.0
- Python >= 3.12
- NVIDIA GPU (SM120 Blackwell optimized, works on any CUDA GPU)

## License

Apache 2.0
