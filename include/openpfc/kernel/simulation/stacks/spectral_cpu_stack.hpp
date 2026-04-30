// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_cpu_stack.hpp
 * @brief One-shot bundle of `World + Decomposition + CpuFft + LocalField` for
 *        spectral CPU solvers driven programmatically (no JSON / `App`).
 *
 * @details
 * Mirrors `pfc::ui::SpectralCpuStack` (which is JSON-driven and lives in
 * the frontend), but takes plain grid parameters so applications and
 * examples can build the same OpenPFC primitive set in **one statement**.
 *
 * The members are stored in a strict declaration order so that internal
 * cross-references stay valid for the lifetime of the stack:
 *
 *     m_world  →  m_decomp  →  m_fft  →  m_u
 *
 *  - `pfc::decomposition::Decomposition` stores `const World&` to its
 *    constructor argument. Putting `m_world` first guarantees it is
 *    initialised before — and destroyed after — `m_decomp`.
 *  - `pfc::fft::CpuFft` internally caches a `Decomposition` (which still
 *    references the same `m_world`).
 *  - `pfc::field::LocalField<double>` is sized to the FFT's local
 *    real-space inbox via `LocalField::from_inbox(world, fft.get_inbox_bounds())`.
 *
 * The class is **non-copyable, non-movable** for the same reason as
 * `pfc::ui::SpectralCpuStack`: a copy or move of the bundle would leave
 * its sub-objects pointing into the source's storage and dangle the
 * moment the source is destroyed. Construct in place, take references.
 *
 * @see openpfc/frontend/ui/spectral_cpu_stack.hpp — JSON-driven sibling.
 * @see openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp — FD analogue.
 */

#include <mpi.h>

#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/field/local_field.hpp>

namespace pfc::sim::stacks {

/**
 * @brief Programmatic spectral CPU stack: World + Decomposition + CpuFft +
 *        LocalField sized to the FFT inbox.
 */
class SpectralCpuStack {
public:
  /**
   * @param size    Global grid size `{Nx, Ny, Nz}`.
   * @param origin  World origin in physical coordinates.
   * @param spacing Grid spacing in physical coordinates.
   * @param rank    Caller's MPI rank on `comm`.
   * @param nproc   Total number of ranks on `comm` (used by
   *                `decomposition::create`).
   * @param comm    MPI communicator passed to the FFT.
   */
  SpectralCpuStack(const pfc::GridSize &size, const pfc::PhysicalOrigin &origin,
                   const pfc::GridSpacing &spacing, int rank, int nproc,
                   MPI_Comm comm = MPI_COMM_WORLD)
      : m_world(pfc::world::create(size, origin, spacing)),
        m_decomp(pfc::decomposition::create(m_world, nproc)),
        m_fft(pfc::fft::create(m_decomp, comm)),
        m_u(pfc::field::LocalField<double>::from_inbox(
            pfc::decomposition::get_world(m_decomp), m_fft.get_inbox_bounds())),
        m_rank(rank), m_nproc(nproc), m_comm(comm) {}

  SpectralCpuStack(const SpectralCpuStack &) = delete;
  SpectralCpuStack &operator=(const SpectralCpuStack &) = delete;
  SpectralCpuStack(SpectralCpuStack &&) = delete;
  SpectralCpuStack &operator=(SpectralCpuStack &&) = delete;

  [[nodiscard]] pfc::World &world() noexcept { return m_world; }
  [[nodiscard]] const pfc::World &world() const noexcept { return m_world; }

  [[nodiscard]] pfc::decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] pfc::fft::CpuFft &fft() noexcept { return m_fft; }
  [[nodiscard]] const pfc::fft::CpuFft &fft() const noexcept { return m_fft; }

  [[nodiscard]] pfc::field::LocalField<double> &u() noexcept { return m_u; }
  [[nodiscard]] const pfc::field::LocalField<double> &u() const noexcept {
    return m_u;
  }

  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int nproc() const noexcept { return m_nproc; }
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  pfc::World m_world;
  pfc::decomposition::Decomposition m_decomp;
  pfc::fft::CpuFft m_fft;
  pfc::field::LocalField<double> m_u;
  int m_rank{0};
  int m_nproc{1};
  MPI_Comm m_comm{MPI_COMM_WORLD};
};

} // namespace pfc::sim::stacks
