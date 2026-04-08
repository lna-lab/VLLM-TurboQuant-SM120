// CUDA native fused WHT rotation for Q pre-rotate and output inverse-rotate.
// Uses __syncthreads() for warp-safe butterfly.
// 1 thread block = 128 threads = 1 vector (token × head).
// Processes normal channels only (indices num_outliers..head_dim-1).
//
// CUDA graph safe: accepts pre-allocated output buffer (no allocation inside).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

#define PADDED_DIM 128

__global__ void fused_rotation_kernel(
    const float* __restrict__ input,     // (N, dim) float32 — normal channels
    float* __restrict__ output,          // (N, dim) float32
    const float* __restrict__ sign_flips, // (PADDED_DIM,) float32
    int dim,                              // normal_dim (120)
    int is_inverse                        // 0=forward, 1=inverse
) {
    int vec_idx = blockIdx.x;
    int tid = threadIdx.x;  // 0..127

    __shared__ float wht_buf[PADDED_DIM];

    // Load input (pad to PADDED_DIM)
    float val = (tid < dim) ? input[vec_idx * dim + tid] : 0.0f;

    // Forward: multiply signs BEFORE WHT
    float sign = sign_flips[tid];
    if (!is_inverse) {
        val = val * sign;
    }

    wht_buf[tid] = val;
    __syncthreads();

    // 7-step butterfly WHT
    for (int step = 0; step < 7; step++) {
        int partner = tid ^ (1 << step);
        float a = wht_buf[tid];
        float b = (partner < PADDED_DIM) ? wht_buf[partner] : 0.0f;
        __syncthreads();
        wht_buf[tid] = ((tid & (1 << step)) == 0) ? (a + b) : (b - a);
        __syncthreads();
    }

    // Normalize by 1/sqrt(PADDED_DIM)
    float result = wht_buf[tid] * (1.0f / 11.313708498984761f);  // 1/sqrt(128)

    // Inverse: multiply signs AFTER WHT
    if (is_inverse) {
        result = result * sign;
    }

    // Store (truncated to dim)
    if (tid < dim) {
        output[vec_idx * dim + tid] = result;
    }
}

// Version that accepts pre-allocated output (CUDA graph safe)
void cuda_apply_rotation_inplace(
    torch::Tensor input,          // (N, dim) float32, contiguous
    torch::Tensor output,         // (N, dim) float32, contiguous, pre-allocated
    torch::Tensor sign_flips,     // (PADDED_DIM,) float32
    bool is_inverse
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(output.is_cuda() && output.is_contiguous());
    TORCH_CHECK(sign_flips.is_cuda());
    TORCH_CHECK(input.sizes() == output.sizes());

    int N = input.size(0);
    int dim = input.size(1);

    if (N == 0) return;

    fused_rotation_kernel<<<N, PADDED_DIM>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sign_flips.data_ptr<float>(),
        dim,
        is_inverse ? 1 : 0
    );
}

// Legacy version that allocates output (NOT graph safe, kept for tests)
torch::Tensor cuda_apply_rotation(
    torch::Tensor input,
    torch::Tensor sign_flips,
    bool is_inverse
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous());
    TORCH_CHECK(sign_flips.is_cuda());

    int N = input.size(0);
    int dim = input.size(1);

    auto output = torch::empty_like(input);

    if (N == 0) return output;

    fused_rotation_kernel<<<N, PADDED_DIM>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sign_flips.data_ptr<float>(),
        dim,
        is_inverse ? 1 : 0
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_apply_rotation", &cuda_apply_rotation, "CUDA fused WHT rotation (allocating)");
    m.def("cuda_apply_rotation_inplace", &cuda_apply_rotation_inplace, "CUDA fused WHT rotation (pre-allocated output)");
}
