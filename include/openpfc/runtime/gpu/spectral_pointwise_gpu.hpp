// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_pointwise_gpu.hpp
 * @brief Device evaluation of a spectral-ETD pointwise functor (CUDA + HIP).
 *
 * @details
 * `spectral_pointwise_apply<F>` is *declared* for every translation unit and
 * *defined* only where a device compiler is active (`__CUDACC__` /
 * `__HIPCC__`). A host TU that instantiates
 * `SpectralETDSystem<Physics, CUDASpace / HIPSpace>` therefore links against
 * an explicit instantiation that must live in exactly one `.cu` / `.hip` TU
 * of the consumer:
 *
 * @code
 * // apps/foo/src/gpu/foo_pointwise.inc  (stamped into foo_pointwise.cu/.hip)
 * #include <foo/foo_pointwise.hpp>
 * #include <openpfc/runtime/gpu/spectral_pointwise_gpu.hpp>
 * OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(foo::FooPointwise)
 * @endcode
 *
 * Forgetting the instantiation fails closed at link time with the functor's
 * name in the missing symbol. This keeps the app's session / JSON headers out
 * of the device compiler: only the tiny functor header is device-compiled.
 *
 * The kernel mirrors `pfc::sim::for_each_spectral_cell` exactly (same
 * coordinate formula, same optional outputs) so host and device paths agree.
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>

#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::sim::gpu {

/**
 * @brief Launch the pointwise nonlinearity over an owned box on device data.
 *
 * All pointers are device memory. `psi_mf`, `p_star`, `fe_out` may be null
 * (`fe_out` is written only when `F` models `HasFreeEnergyDensity`).
 * Synchronizes @p stream before returning.
 */
template <class F>
void spectral_pointwise_apply(const PointwiseGeometry &g, double t,
                              const double *psi, const double *psi_mf,
                              const double *p_star, double *n_out, double *fe_out,
                              F f, gpuStream_t stream);

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

template <class F>
__global__ void spectral_pointwise_kernel(PointwiseGeometry g, double t,
                                          const double *psi, const double *psi_mf,
                                          const double *p_star, double *n_out,
                                          double *fe_out, F f) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= g.volume()) {
    return;
  }
  SpectralCell c;
  c.psi = psi[idx];
  c.psi_mf = (psi_mf != nullptr) ? psi_mf[idx] : 0.0;
  c.p_star = (p_star != nullptr) ? p_star[idx] : 0.0;
  g.coords(idx, c.x, c.y, c.z);
  c.t = t;
  n_out[idx] = f.nonlinearity(c);
  if constexpr (HasFreeEnergyDensity<F>) {
    if (fe_out != nullptr) {
      fe_out[idx] = f.free_energy_density(c);
    }
  }
}

template <class F>
void spectral_pointwise_apply(const PointwiseGeometry &g, double t,
                              const double *psi, const double *psi_mf,
                              const double *p_star, double *n_out, double *fe_out,
                              F f, gpuStream_t stream) {
  static_assert(SpectralPointwise<F>,
                "spectral_pointwise_apply: F must be a trivially copyable "
                "functor with OPENPFC_HD double nonlinearity(const SpectralCell&)");
  const std::size_t n = g.volume();
  if (n == 0) {
    return;
  }
  constexpr unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((n + threads - 1) / static_cast<std::size_t>(threads));
  GPU_LAUNCH_KERNEL(spectral_pointwise_kernel<F>, blocks, threads,
                    (g, t, psi, psi_mf, p_star, n_out, fe_out, f), stream);
  GPU_CHECK(::pfc::gpuStreamSynchronize(stream));
}

/// Explicitly instantiate the device launcher for functor type @p F.
#define OPENPFC_INSTANTIATE_SPECTRAL_POINTWISE(F)                                   \
  template void ::pfc::sim::gpu::spectral_pointwise_apply<F>(                       \
      const ::pfc::sim::PointwiseGeometry &, double, const double *,                \
      const double *, const double *, double *, double *, F, ::pfc::gpuStream_t);

#endif // device compiler

} // namespace pfc::sim::gpu

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
