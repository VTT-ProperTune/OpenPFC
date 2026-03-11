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
#include <string>

namespace pfc {
namespace detail {

template <typename Functor, typename IndexType>
void parallel_for_impl_hip(const RangePolicy<HIP, IndexType> &policy,
                           const Functor &functor) {
  for (IndexType i = policy.begin(); i != policy.end(); ++i) {
    functor(i);
  }
}

template <typename Functor, typename IndexType>
void parallel_for_impl_hip(const MDRangePolicy<HIP, Rank<3>, IndexType> &policy,
                           const Functor &functor) {
  for (IndexType i = policy.start(0); i != policy.end(0); ++i) {
    for (IndexType j = policy.start(1); j != policy.end(1); ++j) {
      for (IndexType k = policy.start(2); k != policy.end(2); ++k) {
        functor(i, j, k);
      }
    }
  }
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

inline void fence(const HIP &) { hipDeviceSynchronize(); }

} // namespace pfc

#endif // OpenPFC_ENABLE_HIP
