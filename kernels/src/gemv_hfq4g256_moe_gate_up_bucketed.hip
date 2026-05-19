#include <hip/hip_runtime.h>

// Routing-bucketed MoE gate_up GEMV (HFQ4G256). Each block processes ONE
// output row for ONE expert across ALL tokens routed to that expert, holding
// the weight tile in registers for reuse across the bucket.
//
// Shapes:
//   expert_ptrs       : [n_exp]            device pointer table
//   expert_offsets    : [n_exp + 1]        bucket boundaries (from moe_bucket_build)
//   expert_token_list : [N × K_TOP]        packed token_idx * K_TOP + krank, sorted by expert
//   x                 : [N × K]            FWHT-rotated post-RMSNorm
//   y_gate            : [N × K_TOP × MI]   MI = M / 2
//   y_up              : [N × K_TOP × MI]
//
// Grid: (M, n_exp). Block = 32 threads (one wavefront on gfx1201).
//
// Weight-tile budget per lane: groups_per_row * (4 + 4 + 4) bytes
//   = 11 groups × 12 B = 132 B at Gemma 4 (K=2816).
// Hardcoded ceiling at MAX_GROUPS=16 (K up to 4096).

#define MAX_GROUPS 16

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_bucketed(
    const unsigned long long* __restrict__ expert_ptrs,    // [n_exp]
    const int* __restrict__ expert_offsets,                 // [n_exp + 1]
    const int* __restrict__ expert_token_list,              // [N × K_TOP]
    const float* __restrict__ x,                            // [N × K]
    float* __restrict__ y_gate,                             // [N × K_TOP × MI]
    float* __restrict__ y_up,                               // [N × K_TOP × MI]
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
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;

    // Load weight tile for this (row, expert) once.
    float sc_arr[MAX_GROUPS];
    float zp_arr[MAX_GROUPS];
    unsigned int pk_arr[MAX_GROUPS];

    #pragma unroll
    for (int g = 0; g < MAX_GROUPS; g++) {
        if (g < groups_per_row) {
            const char* gp = row_ptr + g * 136;
            sc_arr[g] = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            zp_arr[g] = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            pk_arr[g] = *(const unsigned int*)(gp + 8 + boff);
        }
    }

    const int mi = M >> 1;

    for (int slot = start; slot < end; slot++) {
        int packed = expert_token_list[slot];
        int token_idx = packed / K_TOP;
        int krank = packed - token_idx * K_TOP;
        const float* x_row = x + (long long)token_idx * K;

        float acc = 0.0f;
        #pragma unroll
        for (int g = 0; g < MAX_GROUPS; g++) {
            if (g < groups_per_row) {
                int base = g * 256 + tid * 8;
                unsigned int pk = pk_arr[g];
                float sc = sc_arr[g];
                float zp = zp_arr[g];
                acc += (sc * (float)((pk) & 0xFu)        + zp) * x_row[base]
                     + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[base + 1]
                     + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[base + 2]
                     + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[base + 3]
                     + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[base + 4]
                     + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[base + 5]
                     + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[base + 6]
                     + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[base + 7];
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);

        if (tid == 0) {
            long long base_out =
                (long long)token_idx * K_TOP * mi + (long long)krank * mi;
            if (row < mi) {
                y_gate[base_out + row] = acc;
            } else {
                y_up[base_out + (row - mi)] = acc;
            }
        }
    }
}
