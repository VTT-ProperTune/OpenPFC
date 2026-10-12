// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file reflection_ghosts_gpu.hpp
 * @brief Device even/odd reflection for padded fields.
 *
 * Same `ReflectionWalls` as the host fill. The kernel reads and writes the
 * device buffer in place. Host data is uploaded only when the device copy is
 * stale. The call does not copy the field back to the host.
 *
 * @see kernel/decomposition/reflection_ghosts.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <type_traits>

#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::communication::detail {

#if defined(OpenPFC_ENABLE_CUDA)
void reflection_ghosts_cuda_impl(double *pad, ReflectionGeometry<double> geom);
void reflection_ghosts_cuda_impl(float *pad, ReflectionGeometry<float> geom);
#endif

#if defined(OpenPFC_ENABLE_HIP)
void reflection_ghosts_hip_impl(double *pad, ReflectionGeometry<double> geom);
void reflection_ghosts_hip_impl(float *pad, ReflectionGeometry<float> geom);
#endif

} // namespace pfc::communication::detail

namespace pfc::communication {

#if defined(OpenPFC_ENABLE_CUDA)
template <typename T>
void apply_reflection_ghosts(
    pfc::data::Field<T, CUDASpace> &u, const ReflectionWalls<T> &walls,
    ReflectionExtent extent = ReflectionExtent::PhysicalFaces) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "device reflection supports float and double");
  if (u.storage_halo() < 1) {
    return;
  }
  const detail::ReflectionGeometry<T> geom =
      detail::build_reflection_geometry(u, walls, extent);
  detail::validate_reflection_geometry(geom);
  u.sync_to_device();
  detail::reflection_ghosts_cuda_impl(u.data(), geom);
  u.note_device_write();
}
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename T>
void apply_reflection_ghosts(
    pfc::data::Field<T, HIPSpace> &u, const ReflectionWalls<T> &walls,
    ReflectionExtent extent = ReflectionExtent::PhysicalFaces) {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "device reflection supports float and double");
  if (u.storage_halo() < 1) {
    return;
  }
  const detail::ReflectionGeometry<T> geom =
      detail::build_reflection_geometry(u, walls, extent);
  detail::validate_reflection_geometry(geom);
  u.sync_to_device();
  detail::reflection_ghosts_hip_impl(u.data(), geom);
  u.note_device_write();
}
#endif

} // namespace pfc::communication

#endif
