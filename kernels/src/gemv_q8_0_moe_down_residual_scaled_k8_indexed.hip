#include <hip/hip_runtime.h>

// Fused Q8_0 indexed MoE down GEMV with scaled residual.
//
// Replaces the 8-iteration per-expert CPU loop in `apply_moe_branch` for
// Gemma 4 MoE layers (down_proj is Q8_0 because mi=704 isn't 256-aligned,
// so the quantizer fallback chain lands on Q8_0). The kernel reads:
//   - expert_id      from `topk_indices[krank]` (device)
//   - weight base    from `expert_ptrs[expert_id]`
//   - topk weight    from `topk_weights[krank]`
//   - per-expert sc  from `per_expert_scale[expert_id]`
// — all on device, so the whole MoE branch becomes hipGraph-capture-safe
// and the 60 D2H syncs per token disappear from the prefill path.
//
// Q8_0 row layout (matches `gemv_q8_0`): blocks_per_row = K/32, each
// block is 34 bytes = 2-byte f16 scale + 32 signed-int8 quants. Row
// stride = blocks_per_row * 34. Expert pool is contiguous rows: each
// expert at byte offset i * M * blocks_per_row * 34.
//
// Grid: [M, K_TOP, 1]. Block: [32, 1, 1]. One block computes one
// (output_row, krank) pair; atomicAdd accumulates into x_residual[row].

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_q8_0_moe_down_residual_scaled_k8_indexed(
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
    const unsigned char* __restrict__ A =
        reinterpret_cast<const unsigned char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x = hidden_batch + (long long)krank * K;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    // Mirror the unrolled 8-block inner loop from `gemv_q8_0` so we get
    // the same throughput as the per-expert path it replaces.
    const int outer_iters = blocks_per_row / 8;
    for (int oi = 0; oi < outer_iters; oi++) {
        const unsigned char* base = row_data + oi * 8 * 34;
        const float* xb = x + oi * 256;
        #pragma unroll
        for (int sub = 0; sub < 8; sub++) {
            const unsigned char* block = base + sub * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + tid];
            sum += d * (float)qval * xb[sub * 32 + tid];
        }
    }
    // Tail blocks (K not multiple of 256). For Gemma 4 mi=704, blocks_per_row=22,
    // outer_iters=2 covers 16, tail handles last 6.
    for (int bi = outer_iters * 8; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + tid];
        sum += d * (float)qval * x[bi * 32 + tid];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down(sum, offset);

    if (tid == 0) {
        const float scale = topk_weights[krank] * per_expert_scale[expert_id];
        atomicAdd(&x_residual[row], scale * sum);
    }
}
