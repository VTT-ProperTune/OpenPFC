// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_gpu_stack.hpp
 * @brief Device counterpart of the padded FD + `HaloExchange` bundle.
 *
 * @details
 * Lives in runtime because device `HaloExchange` and `FDGradientDevice` are
 * runtime. Bundles
 *
 *     Domain → Decomposition → Field<double, MemorySpace> → HaloExchange
 *
 * Storage halo equals the constructor `halo_width` (Kobayashi uses 1 or 2).
 * Extra fields on the same subdomain use `make_field()`; extra halo groups
 * (state vs aux) use `make_exchange()`.
 *
 * Non-copyable / non-movable: sub-objects would dangle after a move.
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/runtime/gpu/comm_halo_exchange_gpu.hpp>
#include <openpfc/runtime/gpu/fd_gradient_device_gpu.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::sim::stacks {

template <class MemorySpace> class FDGPUStack {
public:
  using field_type = data::Field<double, MemorySpace>;
  using exchange_type = comm::HaloExchange<MemorySpace, double>;

  FDGPUStack(const FDGPUStack &) = delete;
  FDGPUStack &operator=(const FDGPUStack &) = delete;
  FDGPUStack(FDGPUStack &&) = delete;
  FDGPUStack &operator=(FDGPUStack &&) = delete;

  /**
   * @param domain      Global Cartesian domain.
   * @param halo_width  Storage/iteration halo (typically `fd_order/2`).
   * @param rank        Caller MPI rank on `comm`.
   * @param nproc       Size of `comm`.
   * @param comm        Communicator for halo exchange.
   * @param opt         Halo knobs (`Axes2D()` for nz=1 slabs).
   */
  explicit FDGPUStack(pfc::Domain domain, int halo_width, int rank, int nproc,
                      MPI_Comm comm = MPI_COMM_WORLD,
                      comm::HaloExchangeOptions opt = {})
      : m_domain(std::move(domain)),
        m_decomp(pfc::decomposition::create(m_domain, nproc)),
        m_u(m_domain, pfc::decomposition::local_box(m_decomp, rank), halo_width),
        m_exchanger(m_u, m_decomp, rank, comm, opt), m_opt(opt),
        m_halo_width(halo_width), m_rank(rank), m_nproc(nproc), m_comm(comm) {
    if (halo_width < 1) {
      throw std::invalid_argument(
          "FDGPUStack: halo_width must be >= 1 (padded device Field)");
    }
  }

  [[nodiscard]] field_type make_field() const {
    return field_type(m_domain, pfc::decomposition::local_box(m_decomp, m_rank),
                      m_halo_width);
  }

  [[nodiscard]] exchange_type make_exchange(std::vector<field_type *> fields,
                                            comm::HaloExchangeOptions opt) const {
    return exchange_type(std::move(fields), m_decomp, m_rank, m_comm, opt);
  }

  void exchange_halos() { m_exchanger.exchange(); }

  template <class G> [[nodiscard]] auto gradient(int fd_order = 2) const {
    return pfc::gpu::create<G>(m_u, fd_order);
  }

  [[nodiscard]] const pfc::Domain &domain() const noexcept { return m_domain; }
  [[nodiscard]] pfc::decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] field_type &u() noexcept { return m_u; }
  [[nodiscard]] const field_type &u() const noexcept { return m_u; }

  [[nodiscard]] exchange_type &exchanger() noexcept { return m_exchanger; }
  [[nodiscard]] const exchange_type &exchanger() const noexcept {
    return m_exchanger;
  }

  [[nodiscard]] int halo_width() const noexcept { return m_halo_width; }
  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int nproc() const noexcept { return m_nproc; }
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  pfc::Domain m_domain{};
  pfc::decomposition::Decomposition m_decomp;
  field_type m_u;
  exchange_type m_exchanger;
  comm::HaloExchangeOptions m_opt{};
  int m_halo_width{1};
  int m_rank{0};
  int m_nproc{1};
  MPI_Comm m_comm{MPI_COMM_WORLD};
};

#if defined(OpenPFC_ENABLE_CUDA)
using CUDAFDStack = FDGPUStack<CUDASpace>;
#endif
#if defined(OpenPFC_ENABLE_HIP)
using HIPFDStack = FDGPUStack<HIPSpace>;
#endif

} // namespace pfc::sim::stacks

#endif
