// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file parallel_gpu.hpp
 * @brief Single-source GPU `parallel_for` / `fence` for CUDA and HIP (M3).
 *
 * Device `parallel_for` stays fail-closed (`static_assert`) until a real
 * kernel launch exists. `fence` synchronizes the matching device. Vendor
 * headers `parallel_cuda.hpp` / `parallel_hip.hpp` are thin includes of
 * this file so existing call sites keep compiling.
 *
 * Per-tag fence calls the native runtime (not `gpu_api.hpp`) so a CUDA+HIP
 * co-enabled translation unit can fence both spaces.
 *
 * @see kernel/execution/parallel.hpp
 * @see runtime/gpu/gpu_check.hpp
 */

#pragma once

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <openpfc/kernel/execution/parallel.hpp>
#include <openpfc/kernel/execution/policy.hpp>
#include <openpfc/runtime/gpu/execution_space_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_check.hpp>
#include <string>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace pfc {
namespace detail {

#if defined(OpenPFC_ENABLE_CUDA)
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
#endif

#if defined(OpenPFC_ENABLE_HIP)
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
#endif

} // namespace detail

#if defined(OpenPFC_ENABLE_CUDA)
template <typename IndexType, typename Functor>
void parallel_for(const RangePolicy<Cuda, IndexType> &policy,
                  const Functor &functor) {
  if (policy.size() == 0) {
    return;
  }
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
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <typename IndexType, typename Functor>
void parallel_for(const RangePolicy<HIP, IndexType> &policy,
                  const Functor &functor) {
  if (policy.size() == 0) {
    return;
  }
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
#endif

} // namespace pfc

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
