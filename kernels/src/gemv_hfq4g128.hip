#include <hip/hip_runtime.h>

// HFQ4-G128 GEMV. Operates on `ceil(K/128)` groups per row — the quantizer
// pads partial trailing groups with `min_val` so storage is always
// `ceil(K/128) * 72` bytes. The original kernel did `K / 128` (integer
// divide, floor) and silently dropped the partial last group's weights:
// e.g. K=704 (Gemma 4 26B-A4B moe_intermediate_size) or K=2112 (dense
// intermediate_size) would compute floor(K/128) × 128 valid
// multiplications and entirely skip the trailing real-data indices,
// producing systematically wrong dot products. Two changes:
//   1. `groups_per_row = (K + 127) / 128` (ceil) so the partial group's
//      weight bytes get read.
//   2. Per-thread bounds check on `base_idx + {0,1,2,3} < K` so the four
//      multiplications in that thread only contribute when the index is
//      in the valid range. Threads outside the valid range still
//      participate in `__shfl_down` (their acc=0 contribution).
__launch_bounds__(32, 20)
extern "C" __global__ void gemv_hfq4g128(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = (K + 127) / 128;
    const int row_bytes = groups_per_row * 72;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 72;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        // 128 weights / 32 threads = 4 weights per thread
        int base_idx = g * 128 + tid * 4;
        int byte_off = tid * 2;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];

        float v0 = scale * (float)(b0 & 0xF) + zero;
        float v1 = scale * (float)(b0 >> 4)  + zero;
        float v2 = scale * (float)(b1 & 0xF) + zero;
        float v3 = scale * (float)(b1 >> 4)  + zero;

        // Fast path: all 4 indices in range. Branchless for the common case.
        if (base_idx + 3 < K) {
            acc += v0 * x[base_idx]     + v1 * x[base_idx + 1]
                 + v2 * x[base_idx + 2] + v3 * x[base_idx + 3];
        } else {
            // Partial last group: per-element bounds check.
            if (base_idx + 0 < K) acc += v0 * x[base_idx + 0];
            if (base_idx + 1 < K) acc += v1 * x[base_idx + 1];
            if (base_idx + 2 < K) acc += v2 * x[base_idx + 2];
            if (base_idx + 3 < K) acc += v3 * x[base_idx + 3];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}
