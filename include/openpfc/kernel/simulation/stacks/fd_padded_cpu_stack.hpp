// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_padded_cpu_stack.hpp
 * @brief Host padded FD + `HaloExchange` bundle (Kobayashi-style storage halo).
 *
 * @details
 * Distinct from `FDCPUStack`, which is unpadded `Field` + face-halo
 * `SparseExchange` (heat3d / wave2d). This stack is the CPU twin of
 * `FDGPUStack`:
 *
 *     Domain → Decomposition → Field<double, HostSpace> → HaloExchange
 *
 * Storage halo equals the constructor `halo_width` (Kobayashi uses 1).
 * Extra fields on the same subdomain use `make_field()`; extra halo groups
 * (state vs aux) use `make_exchange()`.
 *
 * Non-copyable / non-movable: sub-objects would dangle after a move.
 */

#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/fd_gradient.hpp>

namespace pfc::sim::stacks {

class FDPaddedCPUStack {
public:
  using field_type = data::Field<double, HostSpace>;
  using exchange_type = comm::HaloExchange<HostSpace, double>;

  FDPaddedCPUStack(const FDPaddedCPUStack &) = delete;
  FDPaddedCPUStack &operator=(const FDPaddedCPUStack &) = delete;
  FDPaddedCPUStack(FDPaddedCPUStack &&) = delete;
  FDPaddedCPUStack &operator=(FDPaddedCPUStack &&) = delete;

  /**
   * @param domain      Global Cartesian domain.
   * @param halo_width  Storage/iteration halo (typically `fd_order/2`).
   * @param rank        Caller MPI rank on `comm`.
   * @param nproc       Size of `comm`.
   * @param comm        Communicator for halo exchange.
   * @param opt         Halo knobs (`Axes2D()` for nz=1 slabs).
   */
  explicit FDPaddedCPUStack(pfc::Domain domain, int halo_width, int rank, int nproc,
                            MPI_Comm comm = MPI_COMM_WORLD,
                            comm::HaloExchangeOptions opt = {})
      : m_domain(std::move(domain)),
        m_decomp(pfc::decomposition::create(m_domain, nproc)),
        m_u(m_domain, pfc::decomposition::local_box(m_decomp, rank),
            require_padded_halo(halo_width)),
        m_exchanger(m_u, m_decomp, rank, comm, opt), m_opt(opt),
        m_halo_width(halo_width), m_rank(rank), m_nproc(nproc), m_comm(comm) {}

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
    return pfc::field::create<G>(m_u, fd_order);
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
  [[nodiscard]] static int require_padded_halo(int halo_width) {
    if (halo_width < 1) {
      throw std::invalid_argument(
          "FDPaddedCPUStack: halo_width must be >= 1 (padded host Field)");
    }
    return halo_width;
  }

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

} // namespace pfc::sim::stacks
