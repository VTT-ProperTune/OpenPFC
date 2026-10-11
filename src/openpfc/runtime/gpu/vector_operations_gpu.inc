// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <openpfc/runtime/gpu/vector_operations.hpp>
namespace pfc::solvers::gpu::detail {
__global__ void combine_kernel(double *y, double a, const double *x, double b,
                               std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (a == 0 ? 0 : a * x[i]) + (b == 0 ? 0 : b * y[i]);
}
__global__ void dot_kernel(const double *x, const double *y, std::size_t n,
                           double *out) {
  __shared__ double sums[256];
  double sum = 0;
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(gridDim.x) * blockDim.x)
    sum += x[i] * y[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sums[0];
}
__global__ void sum_kernel(double *out, int n) {
  __shared__ double sums[256];
  double sum = 0;
  for (int i = threadIdx.x; i < n; i += 256) sum += out[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[0] = sums[0];
}
__global__ void component_mean(const double *x, std::size_t n, int components,
                               double *out) {
  __shared__ double sums[256];
  double sum = 0;
  for (auto i = std::size_t(threadIdx.x) * components + blockIdx.x; i < n;
       i += 256 * components)
    sum += x[i];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sums[0] / (n / components);
}
__global__ void mean_correction(double *y, std::size_t n, int components,
                                const double *mean, bool subtract) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] += (subtract ? -1 : 1) * mean[i % components];
}
#if defined(OPENPFC_GPU_API_CUDA)
#define OP_NAME(name) name##_cuda
#else
#define OP_NAME(name) name##_hip
#endif
void OP_NAME(combine)(double *y, double a, const double *x, double b,
                      std::size_t n) {
  if (!n) return;
  combine_kernel<<<(n + 255) / 256, 256>>>(y, a, x, b, n);
  check(pfc::gpuGetLastError());
}
double OP_NAME(dot)(const double *x, const double *y, std::size_t n, double *out) {
  if (!n) return 0;
  int blocks = std::min<std::size_t>(1024, (n + 255) / 256);
  dot_kernel<<<blocks, 256>>>(x, y, n, out);
  sum_kernel<<<1, 256>>>(out, blocks);
  check(pfc::gpuGetLastError());
  double result;
  check(pfc::gpuMemcpy(&result, out, sizeof(double), pfc::gpuMemcpyDeviceToHost));
  return result;
}
void OP_NAME(mean)(double *y, const double *x, std::size_t n, int components,
                   double *out, bool subtract) {
  if (!n) return;
  component_mean<<<components, 256>>>(x, n, components, out);
  mean_correction<<<(n + 255) / 256, 256>>>(y, n, components, out, subtract);
  check(pfc::gpuGetLastError());
}
#undef OP_NAME
} // namespace pfc::solvers::gpu::detail
