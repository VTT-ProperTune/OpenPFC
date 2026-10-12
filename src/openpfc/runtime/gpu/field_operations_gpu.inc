// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file field_operations_gpu.inc
 * @brief Owned-cell dot of `pfc::solvers::FieldOperations` (included from .cu/.hip).
 *
 * Halo cells are not summed. A null mask weights every owned cell by 1.
 */

#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/field_operations_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::solvers::gpu::fops {
namespace {

inline void check(const char *what) {
  ::pfc::gpuError_t e = ::pfc::gpuGetLastError();
  if (e != ::pfc::gpuSuccess)
    throw std::runtime_error(std::string("FieldOperations ") + what + ": " +
                             ::pfc::gpuGetErrorString(e));
}

__global__ void owned_dot_kernel(Geometry g, const double *x, const double *y,
                                 const double *m, double *out) {
  __shared__ double sums[256];
  const std::size_t n =
      std::size_t(g.n[0]) * std::size_t(g.n[1]) * std::size_t(g.n[2]);
  double s = 0.0;
  for (std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < n;
       t += std::size_t(gridDim.x) * blockDim.x) {
    const int i = static_cast<int>(t % static_cast<std::size_t>(g.n[0]));
    const int j = static_cast<int>((t / static_cast<std::size_t>(g.n[0])) %
                                   static_cast<std::size_t>(g.n[1]));
    const int k = static_cast<int>(t / (static_cast<std::size_t>(g.n[0]) * g.n[1]));
    const std::size_t q = static_cast<std::size_t>(i + g.hw) +
                          static_cast<std::size_t>(g.pitch[0]) *
                              (static_cast<std::size_t>(j + g.hw) +
                               static_cast<std::size_t>(g.pitch[1]) *
                                   static_cast<std::size_t>(k + g.hw));
    const double w = m ? m[q] : 1.0;
    if (w != 0.0) s += w * x[q] * y[q];
  }
  sums[threadIdx.x] = s;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = sums[0];
}

__global__ void sum_kernel(double *p, int n) {
  __shared__ double sums[256];
  double s = 0.0;
  for (int i = threadIdx.x; i < n; i += 256) s += p[i];
  sums[threadIdx.x] = s;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) p[0] = sums[0];
}

} // namespace

#if defined(OPENPFC_GPU_API_CUDA)
#define OPENPFC_FOPS_NAME(name) name##_cuda
#else
#define OPENPFC_FOPS_NAME(name) name##_hip
#endif

void OPENPFC_FOPS_NAME(owned_dot)(const Geometry &g, const double *x,
                                  const double *y, const double *m, double *out,
                                  int blocks) {
  owned_dot_kernel<<<blocks, 256>>>(g, x, y, m, out);
  check("owned_dot");
}

double OPENPFC_FOPS_NAME(sum)(double *p, int n) {
  sum_kernel<<<1, 256>>>(p, n);
  check("sum");
  double r = 0.0;
  if (::pfc::gpuMemcpy(&r, p, sizeof(double), ::pfc::gpuMemcpyDeviceToHost) !=
      ::pfc::gpuSuccess)
    throw std::runtime_error("FieldOperations sum: device to host copy failed");
  return r;
}

#undef OPENPFC_FOPS_NAME

} // namespace pfc::solvers::gpu::fops
