// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file session_gpu_stack_factory.hpp
 * @brief Construct GPU stacks from `SessionSelection` (M10).
 *
 * @details
 * Lives in runtime because `GPUSpectralStack` / `FDGPUStack` and device FFT
 * factories are runtime. CPU stacks are `session_stack_factory.hpp` in
 * kernel. Stacks are non-copyable; these factories return prvalues.
 *
 * GPU app binaries may omit JSON `backend`; call
 * `apply_omitted_gpu_backend` before the factory so the default cpu token
 * is replaced by this MemorySpace. An explicit `"backend": "cpu"` still
 * fails closed.
 */

#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/session_selection.hpp>

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_CUDA_SPECTRAL) ||        \
    defined(OpenPFC_ENABLE_HIP) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>
#endif

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/gpu_spectral_stack.hpp>
#endif
#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/runtime/gpu/fd_gpu_stack.hpp>
#endif

namespace pfc::sim {

template <class MemorySpace> struct gpu_session_backend;

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
template <> struct gpu_session_backend<CUDASpace> {
  static constexpr SimulationBackend value = SimulationBackend::Cuda;
};
#endif

#if defined(OpenPFC_ENABLE_HIP) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)
template <> struct gpu_session_backend<HIPSpace> {
  static constexpr SimulationBackend value = SimulationBackend::Hip;
};
#endif

template <class MemorySpace>
inline void apply_omitted_gpu_backend(SessionSelection &s,
                                      bool backend_key_present) {
  if (!backend_key_present) {
    s.backend = gpu_session_backend<MemorySpace>::value;
  }
}

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL) || defined(OpenPFC_ENABLE_HIP_SPECTRAL)

template <class MemorySpace>
[[nodiscard]] inline stacks::GPUSpectralStack<MemorySpace>
make_gpu_spectral_stack(const SessionSelection &s, pfc::Domain domain, int rank,
                        int nproc, MPI_Comm comm,
                        const heffte::plan_options &options) {
  require_session_for_stack(s, SimulationMethod::Spectral,
                            gpu_session_backend<MemorySpace>::value);
  return stacks::GPUSpectralStack<MemorySpace>(std::move(domain), rank, nproc, comm,
                                               options);
}

template <class MemorySpace>
[[nodiscard]] inline stacks::GPUSpectralStack<MemorySpace>
make_gpu_spectral_stack(const SessionSelection &s, pfc::Domain domain, int rank,
                        int nproc, MPI_Comm comm = MPI_COMM_WORLD) {
  return make_gpu_spectral_stack<MemorySpace>(
      s, std::move(domain), rank, nproc, comm,
      stacks::gpu_fft_for<MemorySpace>::default_plan_options());
}

#endif

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

template <class MemorySpace>
[[nodiscard]] inline stacks::FDGPUStack<MemorySpace>
make_fd_gpu_stack(const SessionSelection &s, pfc::Domain domain, int rank, int nproc,
                  MPI_Comm comm = MPI_COMM_WORLD,
                  comm::HaloExchangeOptions opt = {}) {
  require_session_for_stack(s, SimulationMethod::Fd,
                            gpu_session_backend<MemorySpace>::value);
  return stacks::FDGPUStack<MemorySpace>(std::move(domain),
                                         halo_width_from_fd_order(s.fd_order), rank,
                                         nproc, comm, opt);
}

#endif

#if defined(OpenPFC_ENABLE_CUDA_SPECTRAL)
template <> struct stack_builder<stacks::GPUSpectralStack<CUDASpace>> {
  static constexpr const char *name = "GPUSpectralStack<CUDASpace>";
  [[nodiscard]] static stacks::GPUSpectralStack<CUDASpace>
  make(const SessionSelection &s, pfc::Domain domain, int rank, int nproc,
       MPI_Comm comm) {
    return make_gpu_spectral_stack<CUDASpace>(s, std::move(domain), rank, nproc,
                                              comm);
  }
};
#endif

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
template <> struct stack_builder<stacks::GPUSpectralStack<HIPSpace>> {
  static constexpr const char *name = "GPUSpectralStack<HIPSpace>";
  [[nodiscard]] static stacks::GPUSpectralStack<HIPSpace>
  make(const SessionSelection &s, pfc::Domain domain, int rank, int nproc,
       MPI_Comm comm) {
    return make_gpu_spectral_stack<HIPSpace>(s, std::move(domain), rank, nproc,
                                             comm);
  }
};
#endif

#if defined(OpenPFC_ENABLE_CUDA)
template <> struct stack_builder<stacks::FDGPUStack<CUDASpace>> {
  static constexpr const char *name = "FDGPUStack<CUDASpace>";
  [[nodiscard]] static stacks::FDGPUStack<CUDASpace> make(const SessionSelection &s,
                                                          pfc::Domain domain,
                                                          int rank, int nproc,
                                                          MPI_Comm comm) {
    return make_fd_gpu_stack<CUDASpace>(s, std::move(domain), rank, nproc, comm);
  }
};
#endif

#if defined(OpenPFC_ENABLE_HIP)
template <> struct stack_builder<stacks::FDGPUStack<HIPSpace>> {
  static constexpr const char *name = "FDGPUStack<HIPSpace>";
  [[nodiscard]] static stacks::FDGPUStack<HIPSpace> make(const SessionSelection &s,
                                                         pfc::Domain domain,
                                                         int rank, int nproc,
                                                         MPI_Comm comm) {
    return make_fd_gpu_stack<HIPSpace>(s, std::move(domain), rank, nproc, comm);
  }
};
#endif

} // namespace pfc::sim
