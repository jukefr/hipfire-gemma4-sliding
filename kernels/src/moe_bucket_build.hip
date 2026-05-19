#include <hip/hip_runtime.h>

// Build routing buckets for token-batched MoE GEMM.
//
// Inputs:
//   topk_indices: [N × K_TOP] int — expert index per (token, k-rank)
//   N, K_TOP, N_EXP scalars
//
// Outputs:
//   expert_offsets:    [N_EXP + 1] int — prefix-sum boundaries; tokens for
//                                        expert e live at indices
//                                        [expert_offsets[e], expert_offsets[e+1])
//   expert_token_list: [N × K_TOP]  int — packed (token_idx, krank) entries
//                                        sorted by expert. Encoded as
//                                        token_idx * K_TOP + krank, so the
//                                        consumer can recover both via
//                                        div/mod.
//
// One block, n_exp shared counts. n_exp is bounded by MAX_N_EXP (256) which
// covers Gemma 4 (128 experts) with headroom.

#define MAX_N_EXP 256

__launch_bounds__(256, 1)
extern "C" __global__ void moe_bucket_build(
    const int* __restrict__ topk_indices,  // [N × K_TOP]
    int* __restrict__ expert_offsets,       // [N_EXP + 1]
    int* __restrict__ expert_token_list,    // [N × K_TOP]
    int N, int K_TOP, int N_EXP
) {
    __shared__ int counts[MAX_N_EXP + 1];

    const int tid = threadIdx.x;
    const int total = N * K_TOP;

    // Phase 0: zero counts.
    for (int e = tid; e <= N_EXP; e += blockDim.x) {
        counts[e] = 0;
    }
    __syncthreads();

    // Phase 1: histogram.
    for (int i = tid; i < total; i += blockDim.x) {
        int e = topk_indices[i];
        if (e >= 0 && e < N_EXP) {
            atomicAdd(&counts[e], 1);
        }
    }
    __syncthreads();

    // Phase 2: exclusive prefix-sum on counts → start offsets.
    if (tid == 0) {
        int sum = 0;
        for (int e = 0; e < N_EXP; e++) {
            int c = counts[e];
            counts[e] = sum;
            sum += c;
        }
        counts[N_EXP] = sum;  // total
    }
    __syncthreads();

    // Phase 3: publish offsets to global.
    for (int e = tid; e <= N_EXP; e += blockDim.x) {
        expert_offsets[e] = counts[e];
    }
    __syncthreads();

    // Phase 4: scatter. Re-read topk_indices, allocate a slot per (token, krank)
    // by atomicAdd-ing into the start offset for that expert.
    for (int i = tid; i < total; i += blockDim.x) {
        int e = topk_indices[i];
        if (e >= 0 && e < N_EXP) {
            int pos = atomicAdd(&counts[e], 1);
            // Encode (token_idx, krank) as a single int. The kernel that
            // reads this list decodes via /K_TOP and %K_TOP.
            expert_token_list[pos] = i;
        }
    }
}
