#include <hip/hip_runtime.h>

// Batched N-token HFQ4G128 indexed MoE down + scaled residual.
//
// Same shape contract as `gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched`
// but for HFQ4G128 layout (72 B groups, ceil(K/128) groups per row,
// partial-last-group support). Used by Gemma 4 26B-A4B-it where mi=704
// isn't 256-aligned and down_proj lands on HFQ4G128.
//
// Gemma 4 keeps a per_expert_scale tensor separate from topk_weights
// (HF: `router.per_expert_scale`). The HFQ4G256 reference kernel folds
// it into topk_weights upstream, but our Gemma 4 path keeps them split
// so the renorm semantics match the reference exactly. We multiply
// both here.
//
// Shapes:
//   expert_ptrs      : [n_exp]
//   topk_indices     : [N × K_TOP]
//   topk_weights     : [N × K_TOP]
//   per_expert_scale : [n_exp]
//   hidden_batch     : [N × K_TOP × K]    (gelu(gate)*up for each (token, expert) pair)
//   x_residual       : [N × M]            accumulated in-place via atomicAdd
//
// Grid: (M, K_TOP, N). Block = 32 threads.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs,
    const int*   __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ per_expert_scale,
    const float* __restrict__ hidden_batch,
    float*       __restrict__ x_residual,
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        hidden_batch + ((long long)bid * K_TOP + krank) * K;

    const int groups_per_row = (K + 127) / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 72;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

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
        const float scale = topk_weights[routing_base + krank]
                          * per_expert_scale[expert_id];
        atomicAdd(&x_residual[(long long)bid * M + row], scale * acc);
    }
}
