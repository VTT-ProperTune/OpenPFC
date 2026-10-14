// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file reflection_ghosts_gpu.inc
 * @brief Single-source even/odd reflection kernel.
 *
 * Included from the vendor translation units. Do not include from a host TU.
 */

#include <stdexcept>
#include <string>

#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::communication::detail {

#if defined(OPENPFC_GPU_API_CUDA)
constexpr const char *kReflectionGpuKind = "cuda";
#elif defined(OPENPFC_GPU_API_HIP)
constexpr const char *kReflectionGpuKind = "hip";
#endif

template <typename T>
__global__ void reflection_ghosts_kernel(T *pad, ReflectionGeometry<T> geom,
                                        ReflectionVisitPlan plan) {
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= plan.total) {
    return;
  }
  int offset = tid;
  int slab_index = 0;
  for (; slab_index < plan.n; ++slab_index) {
    if (offset < plan.slabs[slab_index].count) {
      break;
    }
    offset -= plan.slabs[slab_index].count;
  }
  if (slab_index >= plan.n) {
    return;
  }
  const ReflectionSlab slab = plan.slabs[slab_index];
  const int dx = slab.end[0] - slab.begin[0];
  const int dy = slab.end[1] - slab.begin[1];
  if (dx <= 0 || dy <= 0) {
    return;
  }
  const int i = slab.begin[0] + (offset % dx);
  const int tiled = offset / dx;
  const int j = slab.begin[1] + (tiled % dy);
  const int k = slab.begin[2] + (tiled / dy);
  const ReflectionSample<T> sample = reflection_sample(geom, i, j, k);
  if (!sample.write) {
    return;
  }
  const int nxp = geom.nloc[0] + 2 * geom.halo;
  const int npy = geom.nloc[1] + 2 * geom.halo;
  const int si = sample.si + geom.halo;
  const int sj = sample.sj + geom.halo;
  const int sk = sample.sk + geom.halo;
  const int src = si + sj * nxp + sk * nxp * npy;
  const int dst = (i + geom.halo) + (j + geom.halo) * nxp +
                  (k + geom.halo) * nxp * npy;
  pad[dst] = sample.scale * pad[src] + sample.offset;
}

template <typename T>
void launch_reflection_ghosts(T *pad, ReflectionGeometry<T> geom) {
  if (geom.halo < 1) {
    return;
  }
  const ReflectionVisitPlan plan = plan_reflection_visits(geom);
  if (plan.total <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (plan.total + threads - 1) / threads;
  reflection_ghosts_kernel<<<blocks, threads>>>(pad, geom, plan);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(
        std::string(kReflectionGpuKind) +
        " reflection ghost kernel failed: " + ::pfc::gpuGetErrorString(err));
  }
  err = ::pfc::gpuDeviceSynchronize();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(
        std::string(kReflectionGpuKind) +
        " reflection ghost sync failed: " + ::pfc::gpuGetErrorString(err));
  }
}

#if defined(OpenPFC_ENABLE_CUDA) && defined(OPENPFC_GPU_API_CUDA)
void reflection_ghosts_cuda_impl(double *pad, ReflectionGeometry<double> geom) {
  launch_reflection_ghosts(pad, geom);
}
void reflection_ghosts_cuda_impl(float *pad, ReflectionGeometry<float> geom) {
  launch_reflection_ghosts(pad, geom);
}
#endif

#if defined(OpenPFC_ENABLE_HIP) && defined(OPENPFC_GPU_API_HIP)
void reflection_ghosts_hip_impl(double *pad, ReflectionGeometry<double> geom) {
  launch_reflection_ghosts(pad, geom);
}
void reflection_ghosts_hip_impl(float *pad, ReflectionGeometry<float> geom) {
  launch_reflection_ghosts(pad, geom);
}
#endif

} // namespace pfc::communication::detail
