# ⚡ VLLM-TurboQuant-SM120

> **Built for a very specific goal: buying more concurrency on RTX PRO 6000 Blackwell by compressing KV cache intelligently.**

Layer-aware KV cache compression for Trinity-Large-Thinking on vLLM, optimized for SM120 Blackwell GPUs.

---

## 🎯 Who Is This For?

This project is for people pushing RTX PRO 6000 Blackwell to the edge with large long-context models.
If your goal is to raise `-c`, fit more concurrent sessions, and delay OOM by compressing KV cache intelligently, this repository is for you.
It is niche by design, but the need is real.

---

## 🔬 What It Does

A vLLM plugin that compresses the KV cache of Trinity-Large-Thinking (398B MoE) using **TurboQuant 3-bit quantization**:

- 🔄 **45 sliding window layers** → passthrough (bounded at 4096 tokens)
- 🗜️ **15 global attention layers** → 3-bit TurboQuant compression
- 📐 **128 → 64 bytes per KV head** — 2× memory reduction per page
- 🧠 **Walsh-Hadamard rotation** — distributes quantization error evenly across dimensions
- 💎 **Outlier preservation** — 8 critical channels kept at full bf16 precision

This exploits Trinity's 3:1 sliding:global attention pattern to achieve significant memory reduction where it matters, without touching the layers that don't need it.

---

## 📊 Benchmark Results

**Hardware:** 4× NVIDIA RTX PRO 6000 Blackwell (96 GB each)
**Model:** Trinity-Large-Thinking-W4A16 (398B MoE, TP=4)
**Context:** 256K tokens max, eager mode

### 🚀 Maximum Concurrency (256K context)

```
┌───────────────────┬──────────────────┬─────────────┐
│ Configuration     │ Max Parallel @   │ KV Cache    │
│                   │ 256K tokens      │ Tokens      │
├───────────────────┼──────────────────┼─────────────┤
│ ❌ FP8 baseline   │       18x        │  ~1.3M      │
│ ⚡ TurboQuant 3b  │       35x        │   2.6M      │
├───────────────────┼──────────────────┼─────────────┤
│ 📈 Improvement    │     1.94×        │   2.01×     │
└───────────────────┴──────────────────┴─────────────┘
```

### ⏱️ Throughput (512 output tokens, 256K context window)

```
┌─────────────┬──────────────┬──────────────────┐
│ Parallelism │ Per-request  │ Aggregate        │
│             │ tok/s        │ tok/s            │
├─────────────┼──────────────┼──────────────────┤
│  1 request  │     8.2      │        8.2       │
│  8 parallel │     8.2      │       65.5       │
│ 16 parallel │     8.1      │      129.1       │
│ 32 parallel │     8.1      │      259.2       │
├─────────────┼──────────────┼──────────────────┤
│ 📈 Scaling  │   perfect linear (no degradation)│
└─────────────┴──────────────┴──────────────────┘
```

### ✅ Quality Verification

```
┌─────────────────────────┬──────────┐
│ Metric                  │ Result   │
├─────────────────────────┼──────────┤
│ 3-bit cosine similarity │ > 0.95   │
│ Outlier preservation    │ bit-exact│
│ Arithmetic reasoning    │ ✅ correct│
│ Code generation         │ ✅ correct│
│ Logical reasoning       │ ✅ correct│
└─────────────────────────┴──────────┘
```

---

## 🏗️ Architecture

```
New K/V tokens
  │
  ├─ 🗜️ TurboQuant compress (outlier split → L2 norm → WHT → Lloyd-Max 3-bit → pack)
  ├─ 📦 Scatter to uint8 KV cache (64 bytes/slot instead of 128)
  │
Cache read (per attention layer)
  │
  ├─ 📖 Decompress all blocks → bf16 temporary
  ├─ 🔄 Pre-rotate Q via Walsh-Hadamard Transform
  ├─ ⚡ Standard Triton paged attention
  └─ 🔄 Inverse-rotate output (undo V rotation)
```

**Slot layout (64 bytes per token per KV head):**

```
┌─────────────────┬───────────────────────┬──────┬─────┐
│ outliers (bf16)  │ packed indices (uint8) │ norm │ pad │
│ 8ch × 2B = 16B  │ 120ch × 3bit = 45B    │ 2B   │ 1B  │
└─────────────────┴───────────────────────┴──────┴─────┘
```

---

## 🚀 Quick Start

```bash
pip install -e .

TRINITY_TURBO_ENABLED=1 \
TRINITY_TURBO_BITS=3 \
vllm serve /path/to/Trinity-Large-Thinking-W4A16 \
    --tensor-parallel-size 4 \
    --attention-backend CUSTOM \
    --kv-cache-dtype fp8_e4m3 \
    --max-model-len 262144 \
    --enforce-eager \
    --gpu-memory-utilization 0.95
```

---

## 📋 Requirements

- vLLM >= 0.19.0
- Python >= 3.12
- NVIDIA GPU (SM120 Blackwell optimized, works on any CUDA GPU)
- PyTorch >= 2.10
- Triton >= 3.0

---

## 🗺️ Roadmap

- [x] **Phase 1** — Plugin skeleton + TurboQuant core (51 tests passing)
- [x] **Phase 2** — Compressed KV cache + Triton decompress kernel (61 tests, 35× concurrency)
- [ ] **Phase 2+** — Fused Triton attention kernel (decompress-in-tile, CUDA graph support)
- [ ] **Phase 3** — cuTile SM120-native kernels
- [ ] **Phase 4** — Gate-based eviction for even higher compression
- [ ] **Phase 5** — Cross-layer KV reconstruction

---

## 🤝 Contributing

Ideas, benchmarks, and PRs are welcome! This is a niche project born out of real need — if you're running into the same walls with long-context MoE models on Blackwell, let's build together.

- 🐛 **Found a bug?** Open an issue
- 💡 **Have an idea?** Start a discussion
- 🔧 **Want to hack?** Check the roadmap above — Phase 2+ is where the action is

---

## 📜 License

Apache 2.0

---

*Built with ❤️ at [Lna-Lab](https://github.com/lna-lab) — a seaside personal lab where AI and humans navigate the future together.*
