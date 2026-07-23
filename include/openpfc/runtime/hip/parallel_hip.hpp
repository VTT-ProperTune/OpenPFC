// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file parallel_hip.hpp
 * @brief HIP execution space support for parallel_for and fence (runtime/hip)
 *
 * Include after openpfc/kernel/execution/parallel.hpp when using HIP policy.
 */

#pragma once

#if defined(OpenPFC_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <openpfc/kernel/execution/parallel.hpp>
#include <openpfc/kernel/execution/policy.hpp>
#include <openpfc/runtime/hip/execution_space_hip.hpp>
#include <openpfc/runtime/hip/hip_check.hpp>
#include <string>

namespace pfc {
namespace detail {

// NOTE (audit 4.2 / PB): mirror of the Cuda case -- the HIP parallel_for used to
// run the functor in a serial *host* loop, silently dereferencing device memory
// on the host when paired with a device-space View. Fail closed at compile time
// until a real device kernel launch exists (M3). `sizeof(Functor) == 0` is a
// dependent false that fires only on instantiation.
// TODO: not tested at runtime (device kernel launch is deferred to M3).
template <typename Functor, typename IndexType>
void parallel_for_impl_hip(const RangePolicy<HIP, IndexType> &, const Functor &) {
  static_assert(sizeof(Functor) == 0,
                "pfc::parallel_for on the HIP execution space is not "
                "implemented yet (it would otherwise run on the host over "
                "device memory). Use DataBuffer + the runtime device kernels, "
                "or run on a host execution space. See audit 4.2 / M3.");
}

template <typename Functor, typename IndexType>
void parallel_for_impl_hip(const MDRangePolicy<HIP, Rank<3>, IndexType> &,
                           const Functor &) {
  static_assert(sizeof(Functor) == 0,
                "pfc::parallel_for on the HIP execution space is not "
                "implemented yet (it would otherwise run on the host over "
                "device memory). Use DataBuffer + the runtime device kernels, "
                "or run on a host execution space. See audit 4.2 / M3.");
}

} // namespace detail

template <typename IndexType, typename Functor>
void parallel_for(const RangePolicy<HIP, IndexType> &policy,
                  const Functor &functor) {
  if (policy.size() == 0) return;
  detail::parallel_for_impl_hip(policy, functor);
}

template <typename IndexType, typename Functor>
void parallel_for(const std::string &name, const RangePolicy<HIP, IndexType> &policy,
                  const Functor &functor) {
  (void)name;
  parallel_for(policy, functor);
}

template <typename IndexType, typename Functor>
void parallel_for(const MDRangePolicy<HIP, Rank<3>, IndexType> &policy,
                  const Functor &functor) {
  detail::parallel_for_impl_hip(policy, functor);
}

template <typename IndexType, typename Functor>
void parallel_for(const std::string &name,
                  const MDRangePolicy<HIP, Rank<3>, IndexType> &policy,
                  const Functor &functor) {
  (void)name;
  parallel_for(policy, functor);
}

inline void fence(const HIP &) {
  hip::detail::hip_check(hipDeviceSynchronize(), "fence(HIP)");
}

} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
