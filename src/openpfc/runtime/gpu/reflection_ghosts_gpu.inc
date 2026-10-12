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
__global__ void reflection_ghosts_kernel(T *pad, ReflectionGeometry<T> geom) {
  const int nxp = geom.nloc[0] + 2 * geom.halo;
  const int npy = geom.nloc[1] + 2 * geom.halo;
  const int npz = geom.nloc[2] + 2 * geom.halo;
  const int n = nxp * npy * npz;
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) {
    return;
  }
  const int i = (tid % nxp) - geom.halo;
  const int t2 = tid / nxp;
  const int j = (t2 % npy) - geom.halo;
  const int k = (t2 / npy) - geom.halo;
  const ReflectionSample<T> sample = reflection_sample(geom, i, j, k);
  if (!sample.write) {
    return;
  }
  const int si = sample.si + geom.halo;
  const int sj = sample.sj + geom.halo;
  const int sk = sample.sk + geom.halo;
  const int src = si + sj * nxp + sk * nxp * npy;
  pad[tid] = sample.scale * pad[src] + sample.offset;
}

template <typename T>
void launch_reflection_ghosts(T *pad, ReflectionGeometry<T> geom) {
  if (geom.halo < 1) {
    return;
  }
  const int nxp = geom.nloc[0] + 2 * geom.halo;
  const int npy = geom.nloc[1] + 2 * geom.halo;
  const int npz = geom.nloc[2] + 2 * geom.halo;
  const int n = nxp * npy * npz;
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  reflection_ghosts_kernel<<<blocks, threads>>>(pad, geom);
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
