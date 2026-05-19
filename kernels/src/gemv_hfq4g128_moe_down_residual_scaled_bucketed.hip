#include <hip/hip_runtime.h>

// Routing-bucketed MoE down + scaled residual (HFQ4G128). Each block processes
// ONE output row for ONE expert across ALL tokens routed to that expert,
// holding the weight tile in registers for reuse across the bucket.
//
// Mirrors `gemv_hfq4g128_moe_down_residual_scaled_k8_indexed_batched` but
// dispatched bucketed: grid (M, n_exp) instead of (M, K_TOP, N).
//
// Shapes:
//   expert_ptrs       : [n_exp]
//   expert_offsets    : [n_exp + 1]
//   expert_token_list : [N × K_TOP] packed (token_idx * K_TOP + krank), sorted by expert
//   topk_weights      : [N × K_TOP]
//   per_expert_scale  : [n_exp]
//   hidden_batch      : [N × K_TOP × K]
//   x_residual        : [N × M]    atomicAdd into existing residual
//
// Grid: (M, n_exp). Block = 32 threads.

#define MAX_DOWN_GROUPS 16

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

    // Load weight tile for this (row, expert) once.
    float sc_arr[MAX_DOWN_GROUPS];
    float zp_arr[MAX_DOWN_GROUPS];
    unsigned char nb0_arr[MAX_DOWN_GROUPS];
    unsigned char nb1_arr[MAX_DOWN_GROUPS];
    int valid_arr[MAX_DOWN_GROUPS];  // how many of {v0,v1,v2,v3} live (0..4)

    #pragma unroll
    for (int g = 0; g < MAX_DOWN_GROUPS; g++) {
        if (g < groups_per_row) {
            const char* gptr = row_ptr + g * 72;
            sc_arr[g] = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
            zp_arr[g] = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
            const unsigned char* nibbles = (const unsigned char*)(gptr + 8);
            const int byte_off = tid * 2;
            nb0_arr[g] = nibbles[byte_off];
            nb1_arr[g] = nibbles[byte_off + 1];
            const int base_idx = g * 128 + tid * 4;
            int v = 0;
            if (base_idx + 0 < K) v++;
            if (base_idx + 1 < K) v++;
            if (base_idx + 2 < K) v++;
            if (base_idx + 3 < K) v++;
            valid_arr[g] = v;
        }
    }

    const float pes = per_expert_scale[expert_id];

    for (int slot = start; slot < end; slot++) {
        int packed = expert_token_list[slot];
        int token_idx = packed / K_TOP;
        int krank = packed - token_idx * K_TOP;
        const float* x =
            hidden_batch + ((long long)token_idx * K_TOP + krank) * K;

        float acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < MAX_DOWN_GROUPS; g++) {
            if (g < groups_per_row) {
                const int base_idx = g * 128 + tid * 4;
                const float v0 = sc_arr[g] * (float)(nb0_arr[g] & 0xF) + zp_arr[g];
                const float v1 = sc_arr[g] * (float)(nb0_arr[g] >> 4)  + zp_arr[g];
                const float v2 = sc_arr[g] * (float)(nb1_arr[g] & 0xF) + zp_arr[g];
                const float v3 = sc_arr[g] * (float)(nb1_arr[g] >> 4)  + zp_arr[g];
                const int v = valid_arr[g];
                if (v == 4) {
                    acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
                         + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
                } else {
                    if (v >= 1) acc += v0 * x[base_idx + 0];
                    if (v >= 2) acc += v1 * x[base_idx + 1];
                    if (v >= 3) acc += v2 * x[base_idx + 2];
                }
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
