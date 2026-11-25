// direct_tr12.hip — AOT gfx1151 bank-conflict-free 3D transpose [B, R, C] -> [B, C, R].
// Replaces Triton _direct_tr12_kernel with zero JIT latency and full coalesced streaming BW.
#include <hip/hip_runtime.h>
#include <stdint.h>

// 32x32 tile with 32x33 padded LDS -> zero bank conflicts, 256 threads (8 waves, 4 rows/thread)
extern "C" __global__ void __launch_bounds__(256) direct_tr12_b16(
    const uint16_t* __restrict__ X, uint16_t* __restrict__ Y,
    int R, int C,
    int64_t sxb, int64_t sxr, int64_t sxc,
    int64_t syb, int64_t syc, int64_t syr) {
  __shared__ uint16_t tile[32][33];
  const int b = blockIdx.z;
  const int c0 = blockIdx.x * 32;
  const int r0 = blockIdx.y * 32;
  const int tx = threadIdx.x;  // 0..31
  const int ty = threadIdx.y;  // 0..7

  const uint16_t* xb = X + (int64_t)b * sxb;
  uint16_t* yb = Y + (int64_t)b * syb;

  const int c_in = c0 + tx;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const int r_in = r0 + ty + i;
    if (r_in < R && c_in < C) {
      tile[ty + i][tx] = xb[(int64_t)r_in * sxr + (int64_t)c_in * sxc];
    }
  }
  __syncthreads();

  const int r_out = r0 + tx;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const int c_out = c0 + ty + i;
    if (c_out < C && r_out < R) {
      yb[(int64_t)c_out * syc + (int64_t)r_out * syr] = tile[tx][ty + i];
    }
  }
}

extern "C" __global__ void __launch_bounds__(256) direct_tr12_b32(
    const uint32_t* __restrict__ X, uint32_t* __restrict__ Y,
    int R, int C,
    int64_t sxb, int64_t sxr, int64_t sxc,
    int64_t syb, int64_t syc, int64_t syr) {
  __shared__ uint32_t tile[32][33];
  const int b = blockIdx.z;
  const int c0 = blockIdx.x * 32;
  const int r0 = blockIdx.y * 32;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const uint32_t* xb = X + (int64_t)b * sxb;
  uint32_t* yb = Y + (int64_t)b * syb;

  const int c_in = c0 + tx;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const int r_in = r0 + ty + i;
    if (r_in < R && c_in < C) {
      tile[ty + i][tx] = xb[(int64_t)r_in * sxr + (int64_t)c_in * sxc];
    }
  }
  __syncthreads();

  const int r_out = r0 + tx;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    const int c_out = c0 + ty + i;
    if (c_out < C && r_out < R) {
      yb[(int64_t)c_out * syc + (int64_t)r_out * syr] = tile[tx][ty + i];
    }
  }
}
