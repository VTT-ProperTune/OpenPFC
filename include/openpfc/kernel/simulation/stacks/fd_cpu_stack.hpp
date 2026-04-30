// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file fd_cpu_stack.hpp
 * @brief One-shot bundle of `World + Decomposition + LocalField + face_halos
 *        + SeparatedFaceHaloExchanger` for finite-difference CPU solvers.
 *
 * @details
 * Programmatic counterpart to `pfc::sim::stacks::SpectralCpuStack` for
 * the explicit-FD path. The members are declared in dependency order so
 * cross-references stay valid for the lifetime of the stack:
 *
 *     m_world  →  m_decomp  →  m_u  →  m_face_halos  →  m_exchanger
 *
 *  - `pfc::decomposition::Decomposition` stores `const World&` to its
 *    constructor argument; `m_world` is initialised first so its
 *    destruction order is correct.
 *  - `pfc::SeparatedFaceHaloExchanger<double>` stores
 *    `const Decomposition&`; `m_decomp` is initialised before it.
 *  - `m_u` is sized to the local subdomain with a halo of
 *    `halo_width = fd_order / 2` (the standard central-difference halo).
 *
 * `exchange_halos()` is the one-line wrapper every FD time-stepping
 * loop calls each step. Keeping it on the stack means the application
 * does not have to pass the field, its size, and the halo buffer
 * triple to `m_exchanger.exchange_halos(...)` every iteration.
 *
 * The class is **non-copyable, non-movable** for the same reason as
 * `pfc::ui::SpectralCpuStack` and `pfc::sim::stacks::SpectralCpuStack`:
 * the internal references would dangle the moment the source bundle is
 * destroyed. Construct in place, take references.
 *
 * @see openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp
 */

#include <array>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/kernel/decomposition/separated_halo_exchange.hpp>
#include <openpfc/kernel/field/local_field.hpp>

namespace pfc::sim::stacks {

/**
 * @brief Programmatic FD periodic CPU stack: World + Decomposition +
 *        halo-aware LocalField + face-halo buffers + halo exchanger.
 */
class FdCpuStack {
public:
  /**
   * @param size      Global grid size `{Nx, Ny, Nz}`.
   * @param origin    World origin in physical coordinates.
   * @param spacing   Grid spacing in physical coordinates.
   * @param fd_order  Even FD order (2, 4, …, 20). Halo width is
   *                  `fd_order / 2`.
   * @param rank      Caller's MPI rank on `comm`.
   * @param nproc     Total number of ranks on `comm`.
   * @param comm      MPI communicator passed to the halo exchanger.
   */
  FdCpuStack(const pfc::GridSize &size, const pfc::PhysicalOrigin &origin,
             const pfc::GridSpacing &spacing, int fd_order, int rank, int nproc,
             MPI_Comm comm = MPI_COMM_WORLD)
      : m_world(pfc::world::create(size, origin, spacing)),
        m_decomp(pfc::decomposition::create(m_world, nproc)),
        m_u(pfc::field::LocalField<double>::from_subdomain(m_decomp, rank,
                                                           fd_order / 2)),
        m_face_halos(
            pfc::halo::allocate_face_halos<double>(m_decomp, rank, fd_order / 2)),
        m_exchanger(m_decomp, rank, fd_order / 2, comm), m_fd_order(fd_order),
        m_rank(rank), m_nproc(nproc), m_comm(comm) {}

  FdCpuStack(const FdCpuStack &) = delete;
  FdCpuStack &operator=(const FdCpuStack &) = delete;
  FdCpuStack(FdCpuStack &&) = delete;
  FdCpuStack &operator=(FdCpuStack &&) = delete;

  /**
   * @brief Synchronise the face-halo region of `m_u` with neighbouring
   *        ranks. Call once per time step before reading derivatives at
   *        the subdomain interior boundary.
   */
  void exchange_halos() {
    m_exchanger.exchange_halos(m_u.data(), m_u.size(), m_face_halos);
  }

  [[nodiscard]] pfc::World &world() noexcept { return m_world; }
  [[nodiscard]] const pfc::World &world() const noexcept { return m_world; }

  [[nodiscard]] pfc::decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] pfc::field::LocalField<double> &u() noexcept { return m_u; }
  [[nodiscard]] const pfc::field::LocalField<double> &u() const noexcept {
    return m_u;
  }

  [[nodiscard]] pfc::SeparatedFaceHaloExchanger<double> &exchanger() noexcept {
    return m_exchanger;
  }
  [[nodiscard]] const pfc::SeparatedFaceHaloExchanger<double> &
  exchanger() const noexcept {
    return m_exchanger;
  }

  [[nodiscard]] int fd_order() const noexcept { return m_fd_order; }
  [[nodiscard]] int halo_width() const noexcept { return m_fd_order / 2; }
  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int nproc() const noexcept { return m_nproc; }
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  pfc::World m_world;
  pfc::decomposition::Decomposition m_decomp;
  pfc::field::LocalField<double> m_u;
  std::array<std::vector<double>, 6> m_face_halos;
  pfc::SeparatedFaceHaloExchanger<double> m_exchanger;
  int m_fd_order{2};
  int m_rank{0};
  int m_nproc{1};
  MPI_Comm m_comm{MPI_COMM_WORLD};
};

} // namespace pfc::sim::stacks
