// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file spectral_cpu_stack.hpp
 * @brief One-shot bundle of `Domain + Decomposition + CPUFFT + Field` for
 *        spectral CPU solvers driven programmatically (no JSON / `App`).
 *
 * @details
 * Programmatic (and JSON via `make_simulation_session`) spectral CPU
 * stack: Domain + Decomposition + host FFT + Field. JSON HeFFTe plan
 * options overlay through the optional `heffte::plan_options` constructor.
 *
 * The members are stored in a strict declaration order so that internal
 * cross-references stay valid for the lifetime of the stack:
 *
 *     m_geometry  →  m_decomp  →  m_fft  →  m_u
 *
 *  - Geometry (size/spacing/origin/periodicity) is extracted from the
 *    Domain and stored internally.
 *  - Decomposition is created directly from Domain.
 *  - The selected transform implements `pfc::fft::IHostFFT`.
 *  - `pfc::data::Field<double>` is sized to the FFT's local real-space
 *    inbox via `pfc::data::field_from_inbox(domain, fft.get_inbox_bounds())`.
 *
 * The class is **non-copyable, non-movable**: a copy or move of the
 * bundle would leave its sub-objects pointing into the source's storage
 * and dangle the moment the source is destroyed. Construct in place,
 * take references.
 *
 * @see openpfc/kernel/simulation/stacks/fd_cpu_stack.hpp — FD analogue.
 * @see openpfc/frontend/ui/from_json_simulation_session.hpp — JSON factory.
 */

#include <memory>
#include <stdexcept>
#include <utility>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/decomposition/brick_split.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/local_fft.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/field/scaled_field.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/simulation/du_field.hpp>

namespace pfc::sim::stacks {

/**
 * @brief Internal geometry storage extracted from Domain.
 */
struct SpectralGeometry {
  pfc::Int3 size{1, 1, 1};
  pfc::Real3 spacing{1.0, 1.0, 1.0};
  pfc::Real3 origin{0.0, 0.0, 0.0};
  pfc::Bool3 periodic{true, true, true};
};

/**
 * @brief Programmatic spectral CPU stack: Domain + Decomposition + CPUFFT +
 *        Field sized to the FFT inbox.
 */
class SpectralCPUStack {
public:
  /**
   * @param domain  The global Cartesian simulation domain.
   * @param rank    Caller's MPI rank on `comm`.
   * @param nproc   Total number of ranks on `comm` (used by
   *                `decomposition::create`).
   * @param comm    MPI communicator passed to the FFT.
   */
  explicit SpectralCPUStack(pfc::Domain domain, int rank, int nproc, MPI_Comm comm,
                            const heffte::plan_options &options)
      : SpectralCPUStack(std::move(domain), rank, nproc, comm,
                         pfc::fft::Implementation::Heffte, options) {}

  explicit SpectralCPUStack(pfc::Domain domain, int rank, int nproc,
                            MPI_Comm comm = MPI_COMM_WORLD)
      : SpectralCPUStack(std::move(domain), rank, nproc, comm,
                         heffte::default_options<heffte::backend::fftw>()) {}

  /// Local execution requires one actual communicator rank owning the domain.
  SpectralCPUStack(pfc::Domain domain, int rank, int nproc, MPI_Comm comm,
                   pfc::fft::Implementation implementation,
                   const heffte::plan_options &options =
                       heffte::default_options<heffte::backend::fftw>())
      : m_geometry({domain.size, domain.spacing, domain.origin, domain.periodic}),
        m_decomp(make_decomposition(domain, rank, nproc, comm, implementation)),
        m_fft(make_fft(domain, m_decomp, rank, comm, implementation, options)),
        m_u(pfc::data::field_from_inbox<double>(domain, m_fft->get_inbox_bounds())),
        m_implementation(implementation), m_rank(rank), m_nproc(nproc),
        m_comm(comm) {}

  SpectralCPUStack(const SpectralCPUStack &) = delete;
  SpectralCPUStack &operator=(const SpectralCPUStack &) = delete;
  SpectralCPUStack(SpectralCPUStack &&) = delete;
  SpectralCPUStack &operator=(SpectralCPUStack &&) = delete;

  [[nodiscard]] const SpectralGeometry &geometry() const noexcept {
    return m_geometry;
  }

  [[nodiscard]] pfc::decomposition::Decomposition &decomposition() noexcept {
    return m_decomp;
  }
  [[nodiscard]] const pfc::decomposition::Decomposition &
  decomposition() const noexcept {
    return m_decomp;
  }

  [[nodiscard]] pfc::fft::IHostFFT &fft() noexcept { return *m_fft; }
  [[nodiscard]] const pfc::fft::IHostFFT &fft() const noexcept { return *m_fft; }

  [[nodiscard]] pfc::fft::Implementation fft_choice() const noexcept {
    return m_implementation;
  }

  [[nodiscard]] pfc::data::Field<double> &u() noexcept { return m_u; }
  [[nodiscard]] const pfc::data::Field<double> &u() const noexcept { return m_u; }

  /**
   * @brief Build a compact-driver residual field for the spectral stack.
   *
   * Returns a `pfc::sim::DuField<G, pfc::field::SpectralGradient<G>>`
   * bound to `m_u` and the cached FFT plan. There is no halo exchange to
   * hide here, so the prepare callable is a no-op; the spectral
   * evaluator's own `prepare()` runs the forward FFT plus one inverse
   * FFT per declared derivative inside `apply(...)`.
   *
   * Usage mirrors the FD stack:
   *
   *     auto& u  = stack.u();
   *     auto  du = stack.du<MyGrads>();
   *     du.apply([](const G& g) { ... });
   *     u += dt * du;
   *
   * Returned by value (move). Captures `this` only indirectly through
   * the evaluator's reference to `m_u.vec()`; must not outlive the stack.
   *
   * @tparam G  Model-owned per-point grads aggregate.
   */
  template <class G> [[nodiscard]] auto du() {
    auto eval = pfc::field::create<G>(m_u, *m_fft);
    using EvalT = decltype(eval);
    return pfc::sim::DuField<G, EvalT>(m_u.size(), std::move(eval), []() {});
  }

  [[nodiscard]] int rank() const noexcept { return m_rank; }
  [[nodiscard]] int nproc() const noexcept { return m_nproc; }
  [[nodiscard]] MPI_Comm mpi_comm() const noexcept { return m_comm; }

private:
  static pfc::decomposition::Decomposition
  make_decomposition(const pfc::Domain &domain, int rank, int nproc, MPI_Comm comm,
                     pfc::fft::Implementation implementation) {
    if (implementation == pfc::fft::Implementation::Local) {
      if (comm == MPI_COMM_NULL)
        throw std::invalid_argument(
            "SpectralCPUStack: local FFT needs a valid communicator");
      int comm_size = 0, comm_rank = -1;
      MPI_Comm_size(comm, &comm_size);
      MPI_Comm_rank(comm, &comm_rank);
      if (nproc != 1 || rank != 0 || comm_size != 1 || comm_rank != 0)
        throw std::invalid_argument("SpectralCPUStack: local FFT needs exactly one "
                                    "rank and matching communicator rank/size");
    } else if (implementation != pfc::fft::Implementation::Heffte) {
      throw std::invalid_argument("SpectralCPUStack: invalid FFT choice");
    }
    return pfc::decomposition::create(
        domain, pfc::decomposition::spectral_fft_proc_grid(domain.size, nproc));
  }

  static std::unique_ptr<pfc::fft::IHostFFT>
  make_fft(const pfc::Domain &domain,
           const pfc::decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
           pfc::fft::Implementation implementation,
           const heffte::plan_options &options) {
    if (implementation == pfc::fft::Implementation::Local) {
      if (!(pfc::decomposition::local_box(decomp, rank) ==
            pfc::domain::index_box(domain)))
        throw std::invalid_argument(
            "SpectralCPUStack: local FFT needs the whole domain");
      return std::make_unique<pfc::fft::LocalFFT>(domain.size);
    }
    return std::unique_ptr<pfc::fft::IHostFFT>(
        new pfc::fft::CPUFFT(pfc::fft::create(decomp, rank, comm, 0, options)));
  }

  SpectralGeometry m_geometry;
  pfc::decomposition::Decomposition m_decomp;
  std::unique_ptr<pfc::fft::IHostFFT> m_fft;
  pfc::data::Field<double> m_u;
  pfc::fft::Implementation m_implementation;
  int m_rank{0};
  int m_nproc{1};
  MPI_Comm m_comm{MPI_COMM_WORLD};
};

} // namespace pfc::sim::stacks
