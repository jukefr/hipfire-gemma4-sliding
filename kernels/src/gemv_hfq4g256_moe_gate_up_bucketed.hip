#include <hip/hip_runtime.h>

// Routing-bucketed MoE gate_up GEMV (HFQ4G256), LDS-staged weight tile.
//
// Each block processes ONE output row for ONE expert across ALL tokens
// routed to that expert. The weight row is loaded once into LDS at the
// start of the block, then re-read from LDS in the inner loop for every
// token in the bucket. This trades a one-time global → LDS load per row
// for N_bucket re-uses, while keeping VGPR pressure low so wave
// occupancy stays at __launch_bounds__(32, 16).
//
// Earlier register-tile version held sc/zp/pk arrays in ~48 VGPRs per
// lane, which dropped occupancy below the indexed_batched baseline and
// regressed prefill by 5.7%. LDS-staged version moves those 48 floats
// off the register budget at the cost of one set of LDS reads per inner
// iteration (still ~10x faster than re-reading from global / L2).
//
// Shapes:
//   expert_ptrs       : [n_exp]              device pointer table
//   expert_offsets    : [n_exp + 1]          bucket boundaries (moe_bucket_build)
//   expert_token_list : [N × K_TOP]          token_idx * K_TOP + krank, sorted by expert
//   x                 : [N × K]              FWHT-rotated post-RMSNorm
//   y_gate            : [N × K_TOP × MI]     MI = M / 2
//   y_up              : [N × K_TOP × MI]
//
// Grid: (M, n_exp). Block: 32 threads.
// LDS budget per block: groups_per_row * 136 bytes (1496 B for Gemma 4 K=2816).
// Bounded by MAX_LDS_BYTES = 2176 B (K up to 4096).

#define MAX_LDS_BYTES 2176

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

    // Stage the entire weight row into LDS. 32 threads cooperate so each
    // moves groups_per_row * 136 / 32 bytes. groups_per_row*136 must fit
    // in MAX_LDS_BYTES.
    __shared__ unsigned char lds_weights[MAX_LDS_BYTES];
    const int row_bytes = groups_per_row * 136;
    // uint32 stride = 4 bytes per thread per "word"; row_bytes/4 words total
    const int n_words = row_bytes >> 2;
    unsigned int* lds_w32 = reinterpret_cast<unsigned int*>(lds_weights);
    const unsigned int* row_w32 = reinterpret_cast<const unsigned int*>(row_ptr);
    for (int w = tid; w < n_words; w += 32) {
        lds_w32[w] = row_w32[w];
    }
    __syncthreads();

    const int mi = M >> 1;
    const int boff = tid * 4;  // byte offset into packed nibbles for this lane

    for (int slot = start; slot < end; slot++) {
        int packed = expert_token_list[slot];
        int token_idx = packed / K_TOP;
        int krank = packed - token_idx * K_TOP;
        const float* x_row = x + (long long)token_idx * K;

        float acc = 0.0f;
        // Iterate groups, reading from LDS. groups_per_row is at most 16
        // (K up to 4096). Use a runtime loop — the compiler can't unroll
        // since K is dynamic, but LDS reads are fast enough that the
        // loop overhead is small.
        for (int g = 0; g < groups_per_row; g++) {
            const unsigned char* gp = lds_weights + g * 136;
            float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp));
            float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
            unsigned int pk = *(const unsigned int*)(gp + 8 + boff);

            int base = g * 256 + tid * 8;
            acc += (sc * (float)((pk) & 0xFu)        + zp) * x_row[base]
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[base + 1]
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[base + 2]
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[base + 3]
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[base + 4]
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[base + 5]
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[base + 6]
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[base + 7];
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
