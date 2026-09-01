// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file session_stack_factory.hpp
 * @brief Construct CPU stacks from `SessionSelection` (M10).
 *
 * @details
 * Stacks are non-copyable / non-movable; these factories return prvalues so
 * C++17 elision initializes the caller's member in place. GPU stacks live
 * in runtime (`GPUSpectralStack` / `FDGPUStack`) and are not built here.
 *
 * FD CPU halo width is `fd_order / 2`. `make_fd_cpu_stack` is unpadded
 * `SparseExchange` (heat3d / wave2d). `make_fd_padded_cpu_stack` is the
 * Kobayashi-style storage halo.
 */

#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/simulation/session_selection.hpp>
#include <openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>
#ifdef OpenPFC_ENABLE_HEFFTE
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#endif

namespace pfc::sim {

#ifdef OpenPFC_ENABLE_HEFFTE
[[nodiscard]] inline stacks::SpectralCPUStack
make_spectral_cpu_stack(const SessionSelection &s, pfc::Domain domain, int rank,
                        int nproc, MPI_Comm comm = MPI_COMM_WORLD) {
  require_session_for_stack(s, SimulationMethod::Spectral, SimulationBackend::Cpu);
  return stacks::SpectralCPUStack(std::move(domain), rank, nproc, comm);
}
#endif

[[nodiscard]] inline stacks::FDCPUStack
make_fd_cpu_stack(const SessionSelection &s, pfc::Domain domain, int rank, int nproc,
                  MPI_Comm comm = MPI_COMM_WORLD) {
  require_session_for_stack(s, SimulationMethod::Fd, SimulationBackend::Cpu);
  return stacks::FDCPUStack(std::move(domain), s.fd_order, rank, nproc, comm);
}

[[nodiscard]] inline stacks::FDPaddedCPUStack
make_fd_padded_cpu_stack(const SessionSelection &s, pfc::Domain domain, int rank,
                         int nproc, MPI_Comm comm = MPI_COMM_WORLD,
                         comm::HaloExchangeOptions opt = {}) {
  require_session_for_stack(s, SimulationMethod::Fd, SimulationBackend::Cpu);
  return stacks::FDPaddedCPUStack(std::move(domain),
                                  halo_width_from_fd_order(s.fd_order), rank, nproc,
                                  comm, opt);
}

#ifdef OpenPFC_ENABLE_HEFFTE
template <> struct stack_builder<stacks::SpectralCPUStack> {
  static constexpr const char *name = "SpectralCPUStack";
  [[nodiscard]] static stacks::SpectralCPUStack make(const SessionSelection &s,
                                                     pfc::Domain domain, int rank,
                                                     int nproc, MPI_Comm comm) {
    return make_spectral_cpu_stack(s, std::move(domain), rank, nproc, comm);
  }
};
#endif

template <> struct stack_builder<stacks::FDCPUStack> {
  static constexpr const char *name = "FDCPUStack";
  [[nodiscard]] static stacks::FDCPUStack make(const SessionSelection &s,
                                               pfc::Domain domain, int rank,
                                               int nproc, MPI_Comm comm) {
    return make_fd_cpu_stack(s, std::move(domain), rank, nproc, comm);
  }
};

template <> struct stack_builder<stacks::FDPaddedCPUStack> {
  static constexpr const char *name = "FDPaddedCPUStack";
  [[nodiscard]] static stacks::FDPaddedCPUStack make(const SessionSelection &s,
                                                     pfc::Domain domain, int rank,
                                                     int nproc, MPI_Comm comm) {
    return make_fd_padded_cpu_stack(s, std::move(domain), rank, nproc, comm);
  }
};

} // namespace pfc::sim
