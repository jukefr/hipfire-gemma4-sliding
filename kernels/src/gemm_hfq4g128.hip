#include <hip/hip_runtime.h>

// HFQ4-G128 batched GEMM. Operates on `ceil(K/128)` groups per row — the
// quantizer pads partial trailing groups with `min_val` so storage is always
// `ceil(K/128) * 72` bytes.
//
// 2026-05-19: ported the partial-trailing-group fix from `gemv_hfq4g128`.
// The original `K / 128` floor here silently dropped the last partial group
// (e.g. K=2112 for Gemma 4 dense down_proj: floor=16 vs needed 17 groups),
// producing dot products missing 64 input elements per row. That manifested
// end-to-end as the v2 batched-prefill structural-loop attractor traced in
// docs/investigations/2026-05-19-gemma4-128k-branch-evidence-debt.md.
// Per-token `gemv_hfq4g128` got this fix when Gemma 4 landed; the batched
// kernel was missed because the prefill ladder validated qualitatively
// ("Capital of France?" → "Paris") at d2bb8573 where the down_proj fell
// through to the per-token GEMV loop, and the 521161f8 commit that routed
// HFQ4G128 through `gemm_hfq4g128` shipped without re-running coherence.

#define BATCH_TILE 8

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_hfq4g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K, int batch_size
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int batch_start = blockIdx.y * BATCH_TILE;
    const int tid = threadIdx.x;

    const int groups_per_row = (K + 127) / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    int local_bs = batch_size - batch_start;
    if (local_bs > BATCH_TILE) local_bs = BATCH_TILE;
    if (local_bs <= 0) return;

    float acc[BATCH_TILE];
    for (int b = 0; b < BATCH_TILE; b++) acc[b] = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_k = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        float w0 = scale * (float)(b0 & 0xF) + zero;
        float w1 = scale * (float)(b0 >> 4)  + zero;
        float w2 = scale * (float)(b1 & 0xF) + zero;
        float w3 = scale * (float)(b1 >> 4)  + zero;

        // Fast path: all 4 indices in range. Branchless for the common case.
        if (base_k + 3 < K) {
            for (int b = 0; b < local_bs; b++) {
                const float* xb = x + (batch_start + b) * K;
                acc[b] += w0 * xb[base_k]     + w1 * xb[base_k + 1]
                        + w2 * xb[base_k + 2] + w3 * xb[base_k + 3];
            }
        } else {
            // Partial last group: per-element bounds check. Mirrors the
            // gemv_hfq4g128 fix exactly.
            for (int b = 0; b < local_bs; b++) {
                const float* xb = x + (batch_start + b) * K;
                if (base_k + 0 < K) acc[b] += w0 * xb[base_k + 0];
                if (base_k + 1 < K) acc[b] += w1 * xb[base_k + 1];
                if (base_k + 2 < K) acc[b] += w2 * xb[base_k + 2];
                if (base_k + 3 < K) acc[b] += w3 * xb[base_k + 3];
            }
        }
    }

    for (int b = 0; b < local_bs; b++) {
        float val = acc[b];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down(val, offset);
        if (tid == 0) y[(batch_start + b) * M + row] = val;
    }
}
