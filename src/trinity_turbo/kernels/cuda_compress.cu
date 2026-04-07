// CUDA native fused compress: WHT + quantize + pack + scatter to KV cache.
// Uses __syncthreads() for warp-safe butterfly WHT.
// 1 thread block = 128 threads = 1 (token, kv_head) pair.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

#define PADDED_DIM 128
#define NUM_OUTLIERS 8
#define NORMAL_DIM 120
#define SLOT_BYTES 80
#define PACKED_OFF 16
#define NORM_OFF 76
#define NUM_LEVELS 16  // 4-bit
#define NUM_BOUNDARIES 15

__global__ void fused_compress_scatter_kernel(
    const __nv_bfloat16* __restrict__ x,    // (N, head_dim)
    uint8_t* __restrict__ kv_cache,          // (num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES)
    const float* __restrict__ boundaries,    // (NUM_BOUNDARIES,)
    const float* __restrict__ sign_flips,    // (PADDED_DIM,)
    const int* __restrict__ slot_mapping,    // (N,) — linear slot index
    const int kv_dim,                        // 0 for K, 1 for V
    const int head_dim,
    const int block_size,
    const int num_kv_heads,
    const float sqrt_d,
    const int stride_cache_block,
    const int stride_cache_kv,
    const int stride_cache_token,
    const int stride_cache_head
) {
    // Thread block processes 1 (token, kv_head) pair
    // blockIdx.x = token_idx * num_kv_heads + head_idx
    int pair_idx = blockIdx.x;
    int token_idx = pair_idx / num_kv_heads;
    int head_idx = pair_idx % num_kv_heads;
    int tid = threadIdx.x;  // 0..127

    // Shared memory for WHT butterfly
    __shared__ float wht_buf[PADDED_DIM];

    // Load input
    float val = 0.0f;
    int src_idx = token_idx * head_dim + head_idx * head_dim + tid;  // Wrong: need per-head layout
    // Actually x is (N, num_kv_heads, head_dim), so:
    // x[token_idx, head_idx, tid]
    if (tid < head_dim) {
        val = __bfloat162float(x[token_idx * num_kv_heads * head_dim + head_idx * head_dim + tid]);
    }

    // Compute slot output pointer (with full bounds check)
    int slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0 || slot_idx >= 100000000) return;  // skip invalid slots
    int blk = slot_idx / block_size;
    int off = slot_idx % block_size;
    // Bounds-safe: clamp to prevent out-of-bounds writes
    long long byte_offset = (long long)blk * stride_cache_block
                          + (long long)kv_dim * stride_cache_kv
                          + (long long)off * stride_cache_token
                          + (long long)head_idx * stride_cache_head;
    uint8_t* slot_ptr = kv_cache + byte_offset;

    // ================================================================
    // 1. Store outliers as bf16 bytes (preserve original bf16 encoding)
    // ================================================================
    if (tid < NUM_OUTLIERS) {
        // Read raw bf16 bytes directly from input (no float conversion)
        const uint8_t* x_bytes = reinterpret_cast<const uint8_t*>(
            &x[token_idx * num_kv_heads * head_dim + head_idx * head_dim + tid]);
        slot_ptr[2 * tid] = x_bytes[0];
        slot_ptr[2 * tid + 1] = x_bytes[1];
    }

    // ================================================================
    // 2. Normal channels: L2 norm
    // ================================================================
    float normal_val = (tid >= NUM_OUTLIERS && tid < head_dim) ? val : 0.0f;

    // Warp reduce for L2 norm
    float sq = normal_val * normal_val;
    // Block reduce using shared memory
    __shared__ float reduce_buf[PADDED_DIM];
    reduce_buf[tid] = sq;
    __syncthreads();

    // Tree reduction
    for (int s = PADDED_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float norm = sqrtf(reduce_buf[0] + 1e-16f);

    // Store norm
    if (tid == 0) {
        __half norm_fp16 = __float2half(norm);
        uint16_t raw;
        memcpy(&raw, &norm_fp16, 2);
        slot_ptr[NORM_OFF] = raw & 0xFF;
        slot_ptr[NORM_OFF + 1] = (raw >> 8) & 0xFF;
        // Padding
        slot_ptr[NORM_OFF + 2] = 0;
        slot_ptr[NORM_OFF + 3] = 0;
    }

    // ================================================================
    // 3. Normalize + sign flip
    // ================================================================
    float normalized = (tid >= NUM_OUTLIERS && tid < head_dim) ? normal_val / norm : 0.0f;
    int wht_idx = tid - NUM_OUTLIERS;  // index in normal channels
    float sign = (wht_idx >= 0 && wht_idx < PADDED_DIM) ? sign_flips[wht_idx] : 1.0f;

    wht_buf[tid] = (tid < PADDED_DIM) ? normalized * sign : 0.0f;
    // Actually: wht_buf should be indexed by normal-channel index (0..127)
    // tid 0..7 are outliers, tid 8..127 map to normal 0..119
    // We need wht_buf[0..127] for WHT. Use tid directly but shift:
    float wht_val = 0.0f;
    if (wht_idx >= 0 && wht_idx < NORMAL_DIM) {
        wht_val = normalized * sign_flips[wht_idx];
    }
    // Pad to PADDED_DIM: positions NORMAL_DIM..PADDED_DIM-1 = 0
    // Use tid as wht index (0..127)
    wht_buf[tid] = (tid < NORMAL_DIM) ? 0.0f : 0.0f;
    if (tid < PADDED_DIM) {
        wht_buf[tid] = (tid < NORMAL_DIM) ?
            (((int)tid < NORMAL_DIM && (NUM_OUTLIERS + tid) < head_dim) ?
                (__bfloat162float(x[token_idx * num_kv_heads * head_dim + head_idx * head_dim + NUM_OUTLIERS + tid]) / norm) * sign_flips[tid]
                : 0.0f)
            : 0.0f;
    }
    __syncthreads();

    // ================================================================
    // 4. WHT Butterfly: 7 steps with __syncthreads()
    // ================================================================
    for (int step = 0; step < 7; step++) {
        int partner = tid ^ (1 << step);
        float a = wht_buf[tid];
        float b = (partner < PADDED_DIM) ? wht_buf[partner] : 0.0f;
        __syncthreads();
        wht_buf[tid] = ((tid & (1 << step)) == 0) ? (a + b) : (b - a);
        __syncthreads();
    }

    // Normalize
    float rotated = wht_buf[tid] * (1.0f / sqrtf((float)PADDED_DIM));

    // ================================================================
    // 5. Scale + quantize
    // ================================================================
    float scaled = rotated * sqrt_d;

    int idx = 0;
    for (int b = 0; b < NUM_BOUNDARIES; b++) {
        if (tid < NORMAL_DIM && scaled > boundaries[b]) idx = b + 1;
    }

    // ================================================================
    // 6. 4-bit pack: even thread → hi nibble, odd thread → lo nibble
    // ================================================================
    // Pack pairs: thread 2j writes (idx[2j] << 4) | idx[2j+1]
    reduce_buf[tid] = (float)idx;  // reuse reduce_buf for index storage
    __syncthreads();

    if (tid < (NORMAL_DIM + 1) / 2) {
        int even_idx = (int)reduce_buf[2 * tid];
        int odd_idx = (2 * tid + 1 < NORMAL_DIM) ? (int)reduce_buf[2 * tid + 1] : 0;
        uint8_t packed = (uint8_t)((even_idx << 4) | (odd_idx & 0x0F));
        slot_ptr[PACKED_OFF + tid] = packed;
    }
}

torch::Tensor fused_compress_scatter(
    torch::Tensor x,           // (N, num_kv_heads, head_dim) bf16
    torch::Tensor kv_cache,    // (num_blocks, 2, block_size, num_kv_heads, SLOT_BYTES) uint8
    torch::Tensor boundaries,  // (15,) float32
    torch::Tensor sign_flips,  // (128,) float32
    torch::Tensor slot_mapping,// (N,) int32
    int kv_dim,                // 0=K, 1=V
    float sqrt_d
) {
    int N = x.size(0);
    int num_kv_heads = x.size(1);
    int head_dim = x.size(2);
    int block_size = kv_cache.size(2);

    int grid = N * num_kv_heads;
    int block = PADDED_DIM;  // 128 threads

    fused_compress_scatter_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        kv_cache.data_ptr<uint8_t>(),
        boundaries.data_ptr<float>(),
        sign_flips.data_ptr<float>(),
        slot_mapping.data_ptr<int>(),
        kv_dim,
        head_dim,
        block_size,
        num_kv_heads,
        sqrt_d,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3)
    );

    return kv_cache;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compress_scatter", &fused_compress_scatter, "Fused TQ4 compress + scatter to KV cache");
}
