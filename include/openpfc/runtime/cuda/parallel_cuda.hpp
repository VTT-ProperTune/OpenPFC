// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file parallel_cuda.hpp
 * @brief CUDA execution space support for parallel_for and fence (runtime/cuda)
 *
 * Include after openpfc/kernel/execution/parallel.hpp when using Cuda policy.
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA)

#include <cuda_runtime.h>
#include <openpfc/kernel/execution/parallel.hpp>
#include <openpfc/kernel/execution/policy.hpp>
#include <openpfc/runtime/cuda/cuda_check.hpp>
#include <openpfc/runtime/cuda/execution_space_cuda.hpp>
#include <string>

namespace pfc {
namespace detail {

// NOTE (audit 4.2 / PB): the Cuda parallel_for previously ran the functor in a
// serial *host* loop. Paired with a device-space View (whose operator()
// dereferences a device pointer), that silently produced a host dereference of
// device memory -- a segfault/garbage trap with no diagnostic. Until a real
// device kernel launch is implemented (M3), fail closed at compile time: any
// attempt to instantiate a device parallel_for is a hard error instead of a
// silent wrong-backend execution. `sizeof(Functor) == 0` is always false but
// dependent, so it fires only on instantiation.
// TODO: not tested at runtime (device kernel launch is deferred to M3).
template <typename Functor, typename IndexType>
void parallel_for_impl_cuda(const RangePolicy<Cuda, IndexType> &, const Functor &) {
  static_assert(sizeof(Functor) == 0,
                "pfc::parallel_for on the Cuda execution space is not "
                "implemented yet (it would otherwise run on the host over "
                "device memory). Use DataBuffer + the runtime device kernels, "
                "or run on a host execution space. See audit 4.2 / M3.");
}

template <typename Functor, typename IndexType>
void parallel_for_impl_cuda(const MDRangePolicy<Cuda, Rank<3>, IndexType> &,
                            const Functor &) {
  static_assert(sizeof(Functor) == 0,
                "pfc::parallel_for on the Cuda execution space is not "
                "implemented yet (it would otherwise run on the host over "
                "device memory). Use DataBuffer + the runtime device kernels, "
                "or run on a host execution space. See audit 4.2 / M3.");
}

} // namespace detail

template <typename IndexType, typename Functor>
void parallel_for(const RangePolicy<Cuda, IndexType> &policy,
                  const Functor &functor) {
  if (policy.size() == 0) return;
  detail::parallel_for_impl_cuda(policy, functor);
}

template <typename IndexType, typename Functor>
void parallel_for(const std::string &name,
                  const RangePolicy<Cuda, IndexType> &policy,
                  const Functor &functor) {
  (void)name;
  parallel_for(policy, functor);
}

template <typename IndexType, typename Functor>
void parallel_for(const MDRangePolicy<Cuda, Rank<3>, IndexType> &policy,
                  const Functor &functor) {
  detail::parallel_for_impl_cuda(policy, functor);
}

template <typename IndexType, typename Functor>
void parallel_for(const std::string &name,
                  const MDRangePolicy<Cuda, Rank<3>, IndexType> &policy,
                  const Functor &functor) {
  (void)name;
  parallel_for(policy, functor);
}

inline void fence(const Cuda &) {
  cuda::detail::cuda_check(cudaDeviceSynchronize(), "fence(Cuda)");
}

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA
