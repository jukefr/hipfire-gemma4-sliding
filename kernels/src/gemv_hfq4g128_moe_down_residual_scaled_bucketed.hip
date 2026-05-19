#include <hip/hip_runtime.h>

// Routing-bucketed MoE down + scaled residual (HFQ4G128), LDS-staged.
//
// Sister to the LDS-staged gate_up bucketed kernel. Each block stages the
// down-weight row into LDS once at the start, then iterates over all
// tokens routed to the expert, atomicAdd-ing into x_residual for each.
//
// Shapes:
//   expert_ptrs       : [n_exp]
//   expert_offsets    : [n_exp + 1]
//   expert_token_list : [N × K_TOP]    token_idx * K_TOP + krank, sorted by expert
//   topk_weights      : [N × K_TOP]
//   per_expert_scale  : [n_exp]
//   hidden_batch      : [N × K_TOP × K]
//   x_residual        : [N × M]        atomicAdd into existing residual
//
// Grid: (M, n_exp). Block: 32 threads.
// LDS budget per block: ceil(K/128) * 72 bytes (432 B for Gemma 4 K=704).
// Bounded by MAX_LDS_DOWN_BYTES = 1152 B (K up to 2048).

#define MAX_LDS_DOWN_BYTES 1152

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g128_moe_down_residual_scaled_bucketed(
    const unsigned long long* __restrict__ expert_ptrs,    // [n_exp]
    const int* __restrict__ expert_offsets,                 // [n_exp + 1]
    const int* __restrict__ expert_token_list,              // [N × K_TOP]
    const float* __restrict__ topk_weights,                 // [N × K_TOP]
    const float* __restrict__ per_expert_scale,             // [n_exp]
    const float* __restrict__ hidden_batch,                 // [N × K_TOP × K]
    float* __restrict__ x_residual,                         // [N × M]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int expert_id = blockIdx.y;
    const int tid = threadIdx.x;

    const int start = expert_offsets[expert_id];
    const int end = expert_offsets[expert_id + 1];
    if (start >= end) return;

    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups_per_row = (K + 127) / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 72;

    // Stage row into LDS (cooperative 32-thread load).
    __shared__ unsigned char lds_weights[MAX_LDS_DOWN_BYTES];
    const int row_bytes = groups_per_row * 72;
    const int n_words = row_bytes >> 2;
    unsigned int* lds_w32 = reinterpret_cast<unsigned int*>(lds_weights);
    const unsigned int* row_w32 = reinterpret_cast<const unsigned int*>(row_ptr);
    for (int w = tid; w < n_words; w += 32) {
        lds_w32[w] = row_w32[w];
    }
    __syncthreads();

    const float pes = per_expert_scale[expert_id];

    for (int slot = start; slot < end; slot++) {
        int packed = expert_token_list[slot];
        int token_idx = packed / K_TOP;
        int krank = packed - token_idx * K_TOP;
        const float* x =
            hidden_batch + ((long long)token_idx * K_TOP + krank) * K;

        float acc = 0.0f;
        for (int g = 0; g < groups_per_row; g++) {
            const unsigned char* gp = lds_weights + g * 72;
            float scale = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            const unsigned char* nibbles = gp + 8;
            const int byte_off = tid * 2;
            const unsigned char b0 = nibbles[byte_off];
            const unsigned char b1 = nibbles[byte_off + 1];
            const float v0 = scale * (float)(b0 & 0xF) + zero;
            const float v1 = scale * (float)(b0 >> 4)  + zero;
            const float v2 = scale * (float)(b1 & 0xF) + zero;
            const float v3 = scale * (float)(b1 >> 4)  + zero;

            const int base_idx = g * 128 + tid * 4;
            if (base_idx + 3 < K) {
                acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
                     + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
            } else {
                if (base_idx + 0 < K) acc += v0 * x[base_idx + 0];
                if (base_idx + 1 < K) acc += v1 * x[base_idx + 1];
                if (base_idx + 2 < K) acc += v2 * x[base_idx + 2];
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);

        if (tid == 0) {
            const float scale =
                topk_weights[token_idx * K_TOP + krank] * pes;
            atomicAdd(&x_residual[(long long)token_idx * M + row], scale * acc);
        }
    }
}
