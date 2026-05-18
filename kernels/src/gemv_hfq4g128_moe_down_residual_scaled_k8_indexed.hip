#include <hip/hip_runtime.h>

// HFQ4G128 variant of the indexed MoE down GEMV with scaled residual.
//
// Gemma 4 26B-A4B's MoE down_proj has K=704 = moe_intermediate_size, which
// isn't 256-aligned so the quantizer's fallback chain lands on HFQ4G128
// (per-row, padded last group). This kernel mirrors the Q8_0 indexed
// variant: reads expert_id from `topk_indices[krank]` (device), weight
// base from `expert_ptrs[expert_id]`, and scale = `topk_weights[krank] *
// per_expert_scale[expert_id]` — all device-side, so the whole MoE branch
// becomes hipGraph-capturable and the legacy CPU per-expert loop's 60
// D2H syncs per token disappear.
//
// HFQ4G128 row layout (matches `gemv_hfq4g128`): groups_per_row = ceil(K/128),
// each group = 8 B header [f32 scale, f32 zero] + 64 B packed nibbles (128
// weights × 4 bits / 8 = 64 B). Row stride = groups_per_row * 72. Caller
// fills the trailing partial-group weight slots with `min_val` so the
// padding contributes zero after dequant; the in-range check below is
// defense-in-depth for completeness.
//
// Grid: [M, K_TOP, 1]. Block: [32, 1, 1]. atomicAdd into x_residual[row].

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_moe_down_residual_scaled_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs,   // [n_exp] device ptrs
    const int*   __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ topk_weights,                // [k_top]
    const float* __restrict__ per_expert_scale,            // [n_exp]
    const float* __restrict__ hidden_batch,                // [k_top × K]
    float*       __restrict__ x_residual,                  // [M]
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = hidden_batch + (long long)krank * K;

    const int groups_per_row = (K + 127) / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 72;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 128 weights / 32 threads = 4 weights per thread.
        const int base_idx = g * 128 + tid * 4;
        const int byte_off = tid * 2;
        const unsigned char b0 = nibbles[byte_off];
        const unsigned char b1 = nibbles[byte_off + 1];

        const float v0 = scale * (float)(b0 & 0xF) + zero;
        const float v1 = scale * (float)(b0 >> 4)  + zero;
        const float v2 = scale * (float)(b1 & 0xF) + zero;
        const float v3 = scale * (float)(b1 >> 4)  + zero;

        if (base_idx + 3 < K) {
            acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
                 + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
        } else {
            if (base_idx + 0 < K) acc += v0 * x[base_idx + 0];
            if (base_idx + 1 < K) acc += v1 * x[base_idx + 1];
            if (base_idx + 2 < K) acc += v2 * x[base_idx + 2];
            if (base_idx + 3 < K) acc += v3 * x[base_idx + 3];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const float scale = topk_weights[krank] * per_expert_scale[expert_id];
        atomicAdd(&x_residual[row], scale * acc);
    }
}
