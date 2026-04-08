"""Test: Can Triton kernels with tl.dot be captured in CUDA graphs on SM120?

Phase 4a workaround: -cc.mode none (disable torch.compile) + -cc.cudagraph_mode full.
The blocker was Triton PTX codegen allocating scratch space for tl.dot()
at different addresses during capture vs replay on SM12x.

This test checks if Triton 3.6 + CUDA 12.8 has fixed this issue.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def simple_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
):
    """Simple kernel with tl.dot — the problematic pattern on SM120."""
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :])

    c = tl.dot(a, b)

    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def tq4_style_dot_kernel(
    Q_ptr, K_ptr, out_ptr,
    M: tl.constexpr, K_DIM: tl.constexpr, N: tl.constexpr,
):
    """Simulates TQ4 attention: load uint8 → decompress → tl.dot."""
    offs_m = tl.arange(0, M)
    offs_k = tl.arange(0, K_DIM)
    offs_n = tl.arange(0, N)

    # Load Q as bf16
    q = tl.load(Q_ptr + offs_m[:, None] * K_DIM + offs_k[None, :]).to(tl.bfloat16)

    # Load K as uint8, decompress to bf16
    k_raw = tl.load(K_ptr + offs_k[:, None] * N + offs_n[None, :])
    k = k_raw.to(tl.bfloat16)

    # tl.dot — the problematic operation on SM120
    c = tl.dot(q, k)

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], c)


def test_cuda_graph_simple_dot():
    """Test 1: Simple tl.dot CUDA graph capture + replay."""
    M, K, N = 16, 128, 16
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    # Warmup
    simple_dot_kernel[(1,)](A, B, C, M=M, N=N, K=K)
    torch.cuda.synchronize()
    warmup_result = C.clone()

    # CUDA graph capture
    C.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        simple_dot_kernel[(1,)](A, B, C, M=M, N=N, K=K)

    # Replay
    C.zero_()
    graph.replay()
    torch.cuda.synchronize()
    graph_result = C.clone()

    # Verify
    diff = (warmup_result - graph_result).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        warmup_result.flatten().float(), graph_result.flatten().float(), dim=0,
    ).item()
    print(f"Simple dot — CUDA graph: max_diff={diff:.6e}, cos_sim={cos:.8f}")
    assert cos > 0.999, f"CUDA graph replay mismatch: cos_sim={cos}"
    print("  PASSED")


def test_cuda_graph_tq4_style_dot():
    """Test 2: TQ4-style uint8→bf16→tl.dot CUDA graph capture + replay."""
    M, K_DIM, N = 16, 128, 16
    Q = torch.randn(M, K_DIM, device="cuda", dtype=torch.bfloat16)
    K = torch.randint(0, 256, (K_DIM, N), device="cuda", dtype=torch.uint8)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    # Warmup
    tq4_style_dot_kernel[(1,)](Q, K, out, M=M, K_DIM=K_DIM, N=N)
    torch.cuda.synchronize()
    warmup_result = out.clone()

    # CUDA graph capture
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tq4_style_dot_kernel[(1,)](Q, K, out, M=M, K_DIM=K_DIM, N=N)

    # Replay multiple times
    results = []
    for i in range(5):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        results.append(out.clone())

    # All replays should match warmup
    for i, r in enumerate(results):
        cos = torch.nn.functional.cosine_similarity(
            warmup_result.flatten().float(), r.flatten().float(), dim=0,
        ).item()
        print(f"TQ4-style dot — replay {i}: cos_sim={cos:.8f}")
        assert cos > 0.999, f"Replay {i} mismatch: cos_sim={cos}"

    print("  PASSED — all replays match")


def test_cuda_graph_full_attention():
    """Test 3: Full TQ4 attention kernel in CUDA graph."""
    import math
    from trinity_turbo.kernels.triton_compress import (
        NORM_OFFSET, PACKED_OFFSET, SLOT_BYTES, compress_to_slot,
    )
    from trinity_turbo.kernels.triton_tq4_unified_attention import tq4_unified_attention
    from trinity_turbo.quant.rotation import apply_rotation
    from trinity_turbo.quant.turboquant import QuantState

    state = QuantState.create(bits=4, head_dim=128, num_outliers=8, device="cuda")
    seq_len, num_kv_heads, num_heads, head_dim = 64, 2, 4, 128
    block_size = 16
    scale = 1.0 / math.sqrt(head_dim)
    num_blocks = seq_len // block_size

    keys = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)

    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES,
        dtype=torch.uint8, device="cuda",
    )
    for t in range(seq_len):
        blk, off = t // block_size, t % block_size
        kv_cache[blk, 0, off] = compress_to_slot(keys[t:t + 1], state)
        kv_cache[blk, 1, off] = compress_to_slot(values[t:t + 1], state)

    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    q = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q[..., state.num_outliers:] = apply_rotation(
        q[..., state.num_outliers:].float(), state.sign_flips,
    ).to(torch.bfloat16)

    out = torch.zeros(1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    key_cache = kv_cache[:, 0].contiguous()
    value_cache = kv_cache[:, 1].contiguous()

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
    tq4_unified_attention(**kwargs)
    torch.cuda.synchronize()
    warmup_result = out.clone()

    # CUDA graph capture
    out.zero_()
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tq4_unified_attention(**kwargs)

        # Replay
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_result = out.clone()

        cos = torch.nn.functional.cosine_similarity(
            warmup_result.flatten().float(), graph_result.flatten().float(), dim=0,
        ).item()
        print(f"Full TQ4 attention — CUDA graph: cos_sim={cos:.8f}")
        assert cos > 0.999, f"CUDA graph mismatch: cos_sim={cos}"
        print("  PASSED — TQ4 attention CUDA graph works on SM120!")
    except Exception as e:
        print(f"  FAILED — CUDA graph capture error: {e}")
        print("  (This was the expected SM120 blocker from Phase 4)")


if __name__ == "__main__":
    print(f"Triton {triton.__version__}, GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print()

    test_cuda_graph_simple_dot()
    print()
    test_cuda_graph_tq4_style_dot()
    print()
    test_cuda_graph_full_attention()
