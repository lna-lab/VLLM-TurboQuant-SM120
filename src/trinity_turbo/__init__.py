"""trinity-turbo: Layer-aware KV cache compression for Trinity-Large-Thinking on vLLM.

Optimized for SM120 Blackwell (RTX PRO 6000).
Combines TurboQuant 3-bit quantization with architecture-aware layer routing:
  - 45 sliding window layers: passthrough (already bounded at 4096 tokens)
  - 15 global attention layers: TurboQuant compress + optional eviction + cross-layer fusion
"""

__version__ = "0.1.0"
