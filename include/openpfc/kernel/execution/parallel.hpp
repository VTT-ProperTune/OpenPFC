// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file parallel.hpp
 * @brief Kokkos-compatible parallel_for and fence
 *
 * @details
 * parallel_for dispatches work according to an execution policy. fence
 * synchronizes outstanding work. Names and semantics match Kokkos.
 *
 * - parallel_for(policy, functor) / parallel_for(name, policy, functor)
 * - fence() / fence(execution_space_instance)
 *
 * Serial: single-threaded loop. OpenMP: optional multi-threaded when
 * _OPENMP defined. Cuda/HIP: fallback to host execution until device
 * kernel launch is implemented.
 *
 * @see policy.hpp for RangePolicy, MDRangePolicy
 * @see execution_space.hpp for execution spaces
 *
 * @author OpenPFC Development Team
 * @date 2025
 */

#pragma once

#include <openpfc/kernel/execution/execution_space.hpp>
#include <openpfc/kernel/execution/policy.hpp>
#include <string>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace pfc {

namespace detail {

// RangePolicy + Serial
template <typename Functor, typename IndexType>
void parallel_for_impl_serial(const RangePolicy<Serial, IndexType> &policy,
                              const Functor &functor) {
  for (IndexType i = policy.begin(); i != policy.end(); ++i) {
    functor(i);
  }
}

// RangePolicy + OpenMP
#if defined(_OPENMP)
template <typename Functor, typename IndexType>
void parallel_for_impl_omp(const RangePolicy<OpenMP, IndexType> &policy,
                           const Functor &functor) {
#pragma omp parallel for
  for (IndexType i = policy.begin(); i != policy.end(); ++i) {
    functor(i);
  }
}
#else
template <typename Functor, typename IndexType>
void parallel_for_impl_omp(const RangePolicy<OpenMP, IndexType> &policy,
                           const Functor &functor) {
  parallel_for_impl_serial(
      RangePolicy<Serial, IndexType>(policy.begin(), policy.end()), functor);
}
#endif

// RangePolicy + Cuda/HIP: fallback to host execution (device kernel TBD)
#if defined(OpenPFC_ENABLE_CUDA)
template <typename Functor, typename IndexType>
void parallel_for_impl_cuda(const RangePolicy<Cuda, IndexType> &policy,
                            const Functor &functor) {
  (void)policy;
  (void)functor;
  // TODO: launch CUDA kernel with functor; for now run on host
  for (IndexType i = policy.begin(); i != policy.end(); ++i) {
    functor(i);
  }
}
#endif
#if defined(OpenPFC_ENABLE_HIP)
template <typename Functor, typename IndexType>
void parallel_for_impl_hip(const RangePolicy<HIP, IndexType> &policy,
                           const Functor &functor) {
  for (IndexType i = policy.begin(); i != policy.end(); ++i) {
    functor(i);
  }
}
#endif

// MDRangePolicy 3D + Serial
template <typename Functor, typename IndexType>
void parallel_for_impl_serial(
    const MDRangePolicy<Serial, Rank<3>, IndexType> &policy,
    const Functor &functor) {
  for (IndexType i = policy.start(0); i != policy.end(0); ++i) {
    for (IndexType j = policy.start(1); j != policy.end(1); ++j) {
      for (IndexType k = policy.start(2); k != policy.end(2); ++k) {
        functor(i, j, k);
      }
    }
  }
}

#if defined(_OPENMP)
template <typename Functor, typename IndexType>
void parallel_for_impl_omp(const MDRangePolicy<OpenMP, Rank<3>, IndexType> &policy,
                           const Functor &functor) {
  const IndexType n0 = policy.end(0) - policy.start(0);
  const IndexType n1 = policy.end(1) - policy.start(1);
  const IndexType n2 = policy.end(2) - policy.start(2);
#pragma omp parallel for collapse(3)
  for (IndexType i = policy.start(0); i != policy.end(0); ++i) {
    for (IndexType j = policy.start(1); j != policy.end(1); ++j) {
      for (IndexType k = policy.start(2); k != policy.end(2); ++k) {
        functor(i, j, k);
      }
    }
  }
}
#else
template <typename Functor, typename IndexType>
void parallel_for_impl_omp(const MDRangePolicy<OpenMP, Rank<3>, IndexType> &policy,
                           const Functor &functor) {
  parallel_for_impl_serial(
      MDRangePolicy<Serial, Rank<3>, IndexType>(policy.start(), policy.end()),
      functor);
}
#endif

#if defined(OpenPFC_ENABLE_CUDA)
template <typename Functor, typename IndexType>
void parallel_for_impl_cuda(const MDRangePolicy<Cuda, Rank<3>, IndexType> &policy,
                            const Functor &functor) {
  for (IndexType i = policy.start(0); i != policy.end(0); ++i) {
    for (IndexType j = policy.start(1); j != policy.end(1); ++j) {
      for (IndexType k = policy.start(2); k != policy.end(2); ++k) {
        functor(i, j, k);
      }
    }
  }
}
#endif
#if defined(OpenPFC_ENABLE_HIP)
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
#endif

} // namespace detail

/**
 * @brief Dispatch parallel_for by policy type (Kokkos-compatible)
 *
 * Functor must have operator()(index_type i) for RangePolicy, or
 * operator()(i, j, k) for MDRangePolicy<..., Rank<3>>.
 */
template <typename ExecutionSpace, typename IndexType, typename Functor>
void parallel_for(const RangePolicy<ExecutionSpace, IndexType> &policy,
                  const Functor &functor) {
  if (policy.size() == 0) return;
  if constexpr (std::is_same_v<ExecutionSpace, Serial>) {
    detail::parallel_for_impl_serial(policy, functor);
  } else if constexpr (std::is_same_v<ExecutionSpace, OpenMP>) {
    detail::parallel_for_impl_omp(policy, functor);
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<ExecutionSpace, Cuda>) {
    detail::parallel_for_impl_cuda(policy, functor);
  }
#endif
#if defined(OpenPFC_ENABLE_HIP)
  else if constexpr (std::is_same_v<ExecutionSpace, HIP>) {
    detail::parallel_for_impl_hip(policy, functor);
  }
#endif
  else {
    static_assert(sizeof(ExecutionSpace) == 0, "Unknown execution space");
  }
}

/** @brief parallel_for with name (for profiling/debugging, Kokkos-compatible) */
template <typename ExecutionSpace, typename IndexType, typename Functor>
void parallel_for(const std::string &name,
                  const RangePolicy<ExecutionSpace, IndexType> &policy,
                  const Functor &functor) {
  (void)name;
  parallel_for(policy, functor);
}

/** @brief MDRangePolicy 3D */
template <typename ExecutionSpace, typename IndexType, typename Functor>
void parallel_for(const MDRangePolicy<ExecutionSpace, Rank<3>, IndexType> &policy,
                  const Functor &functor) {
  if constexpr (std::is_same_v<ExecutionSpace, Serial>) {
    detail::parallel_for_impl_serial(policy, functor);
  } else if constexpr (std::is_same_v<ExecutionSpace, OpenMP>) {
    detail::parallel_for_impl_omp(policy, functor);
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<ExecutionSpace, Cuda>) {
    detail::parallel_for_impl_cuda(policy, functor);
  }
#endif
#if defined(OpenPFC_ENABLE_HIP)
  else if constexpr (std::is_same_v<ExecutionSpace, HIP>) {
    detail::parallel_for_impl_hip(policy, functor);
  }
#endif
  else {
    static_assert(sizeof(ExecutionSpace) == 0, "Unknown execution space");
  }
}

template <typename ExecutionSpace, typename IndexType, typename Functor>
void parallel_for(const std::string &name,
                  const MDRangePolicy<ExecutionSpace, Rank<3>, IndexType> &policy,
                  const Functor &functor) {
  (void)name;
  parallel_for(policy, functor);
}

/**
 * @brief Fence: block until all outstanding work on the default space completes
 * (Kokkos-compatible). Serial: no-op; Cuda: cudaDeviceSynchronize; HIP:
 * hipDeviceSynchronize.
 */
inline void fence() {
#if defined(OpenPFC_ENABLE_CUDA)
  cudaDeviceSynchronize();
#elif defined(OpenPFC_ENABLE_HIP)
  hipDeviceSynchronize();
#endif
  (void)0;
}

/**
 * @brief Fence for a specific execution space instance
 */
template <typename ExecutionSpace> void fence(const ExecutionSpace &) {
  if constexpr (std::is_same_v<ExecutionSpace, Serial>) {
    (void)0;
  }
#if defined(OpenPFC_ENABLE_CUDA)
  else if constexpr (std::is_same_v<ExecutionSpace, Cuda>) {
    cudaDeviceSynchronize();
  }
#endif
#if defined(OpenPFC_ENABLE_HIP)
  else if constexpr (std::is_same_v<ExecutionSpace, HIP>) {
    hipDeviceSynchronize();
  }
#endif
  else {
    (void)0;
  }
}

} // namespace pfc
