// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file modifier_rule_gpu.hpp
 * @brief Device building blocks for boundary conditions on device fields.
 *
 * @details
 * - `modifier_rule_apply<R>` evaluates a pointwise rule
 *   `value = rule(x, y, z, current)` on every owned cell of a device field and
 *   writes the cells whose value changes. Like `spectral_pointwise_apply`, it
 *   is declared everywhere and defined only under a device compiler; the
 *   consumer instantiates it for its rule in one `.cu` / `.hip` TU:
 *
 *   @code
 *   #include <foo/foo_bc_rules.hpp>
 *   #include <openpfc/runtime/gpu/modifier_rule_gpu.hpp>
 *   OPENPFC_INSTANTIATE_MODIFIER_RULE(foo::FooBandRule)
 *   @endcode
 *
 *   `R` must be trivially copyable and callable on the device. Coordinates
 *   follow `PointwiseGeometry::coords`, the same formula as the host
 *   `field::apply_inplace`, so host and device rules see the same points.
 *
 * - `column_max_x` reduces a device field to the maximum over y and z of each
 *   local x column and returns those `nx` values on the host: the only data a
 *   front tracker needs, instead of the whole field.
 *
 * Both synchronise before returning, as boundary conditions run between
 * default-stream transforms.
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/spectral_pointwise.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>

namespace pfc::sim::gpu {

/// Geometry of the owned @p box of @p domain, for the launchers below.
inline PointwiseGeometry pointwise_geometry(const Domain &domain, const Box3i &box) {
  const Real3 o = pfc::domain::lower(domain);
  const Real3 &s = pfc::domain::get_spacing(domain);
  return PointwiseGeometry{.nx = box.size[0],
                           .ny = box.size[1],
                           .nz = box.size[2],
                           .low_x = box.low[0],
                           .low_y = box.low[1],
                           .low_z = box.low[2],
                           .origin_x = o[0],
                           .origin_y = o[1],
                           .origin_z = o[2],
                           .dx = s[0],
                           .dy = s[1],
                           .dz = s[2]};
}

#if defined(OpenPFC_ENABLE_CUDA)
void column_max_x_cuda_impl(const PointwiseGeometry &g, const double *field,
                            double *out_host);
#endif
#if defined(OpenPFC_ENABLE_HIP)
void column_max_x_hip_impl(const PointwiseGeometry &g, const double *field,
                           double *out_host);
#endif

/// Max over y and z of each local x column of a device field (host result).
inline std::vector<double> column_max_x(const PointwiseGeometry &g,
                                        const double *field) {
  std::vector<double> out(static_cast<std::size_t>(g.nx));
  if (g.volume() == 0) return out;
#if defined(OpenPFC_ENABLE_CUDA)
  column_max_x_cuda_impl(g, field, out.data());
#else
  column_max_x_hip_impl(g, field, out.data());
#endif
  return out;
}

/// Apply `rule(x, y, z, current)` to every owned cell of a device field.
template <class R>
void modifier_rule_apply(const PointwiseGeometry &g, double *field, R rule);

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

template <class R>
__global__ void modifier_rule_kernel(PointwiseGeometry g, double *field, R rule) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= g.volume()) return;
  double x, y, z;
  g.coords(idx, x, y, z);
  const double current = field[idx];
  const double value = rule(x, y, z, current);
  if (value != current) field[idx] = value;
}

template <class R>
void modifier_rule_apply(const PointwiseGeometry &g, double *field, R rule) {
  static_assert(std::is_trivially_copyable_v<R>,
                "modifier_rule_apply: the rule is passed by value to a kernel");
  const std::size_t n = g.volume();
  if (n == 0) return;
  constexpr int threads = 256;
  const auto blocks = static_cast<unsigned>((n + threads - 1) / threads);
  modifier_rule_kernel<<<blocks, threads>>>(g, field, rule);
  ::pfc::gpuError_t err = ::pfc::gpuGetLastError();
  if (err == ::pfc::gpuSuccess) err = ::pfc::gpuDeviceSynchronize();
  if (err != ::pfc::gpuSuccess) {
    throw std::runtime_error(std::string("modifier_rule_apply: ") +
                             ::pfc::gpuGetErrorString(err));
  }
}

#endif // device compiler

} // namespace pfc::sim::gpu

/// Explicit instantiation of the device rule launcher for `R` (one TU).
#define OPENPFC_INSTANTIATE_MODIFIER_RULE(R)                                        \
  template void pfc::sim::gpu::modifier_rule_apply<R>(                              \
      const pfc::sim::PointwiseGeometry &, double *, R);

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
