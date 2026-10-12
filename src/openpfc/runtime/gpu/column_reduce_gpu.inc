// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file column_reduce_gpu.inc
 * @brief Single-source x-column maximum (included from vendor .cu/.hip).
 *
 * Do not include this from a host TU.
 *
 * @see runtime/gpu/modifier_rule_gpu.hpp
 */

#include <cfloat>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/modifier_rule_gpu.hpp>

namespace pfc::sim::gpu {
namespace {

// One thread per x column; consecutive threads read consecutive x, so each
// (y, z) row is a coalesced read.
__global__ void column_max_x_kernel(PointwiseGeometry g, const double *field,
                                    double *out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= g.nx) return;
  double m = -DBL_MAX;
  const std::size_t rows = static_cast<std::size_t>(g.ny) * g.nz;
  for (std::size_t r = 0; r < rows; ++r) {
    const double v = field[r * g.nx + i];
    m = v > m ? v : m;
  }
  out[i] = m;
}

void launch_column_max_x(const PointwiseGeometry &g, const double *field,
                         double *out_host) {
  // A small reused device buffer: this runs once per boundary-condition step.
  static thread_local double *buffer = nullptr;
  static thread_local int capacity = 0;
  ::pfc::gpuError_t err = ::pfc::gpuSuccess;
  if (capacity < g.nx) {
    if (buffer != nullptr) (void)::pfc::gpuFree(buffer);
    err = ::pfc::gpuMalloc(reinterpret_cast<void **>(&buffer),
                           sizeof(double) * static_cast<std::size_t>(g.nx));
    capacity = err == ::pfc::gpuSuccess ? g.nx : 0;
  }
  if (err == ::pfc::gpuSuccess) {
    constexpr int threads = 128;
    column_max_x_kernel<<<(g.nx + threads - 1) / threads, threads>>>(g, field,
                                                                     buffer);
    err = ::pfc::gpuGetLastError();
  }
  if (err == ::pfc::gpuSuccess) {
    err = ::pfc::gpuMemcpy(out_host, buffer,
                           sizeof(double) * static_cast<std::size_t>(g.nx),
                           ::pfc::gpuMemcpyDeviceToHost);
  }
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string("column_max_x: ") +
                             ::pfc::gpuGetErrorString(err));
  }
}

} // namespace

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPENPFC_GPU_API_CUDA)
void column_max_x_cuda_impl(const PointwiseGeometry &g, const double *field,
                            double *out_host) {
  launch_column_max_x(g, field, out_host);
}
#endif

#if defined(OpenPFC_ENABLE_HIP) && defined(OPENPFC_GPU_API_HIP)
void column_max_x_hip_impl(const PointwiseGeometry &g, const double *field,
                           double *out_host) {
  launch_column_max_x(g, field, out_host);
}
#endif

} // namespace pfc::sim::gpu
