// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file reporting.hpp
 * @brief Heat-equation reference solution and run-end metadata reporter.
 *
 * @details
 * Heat-specific verification + output. The pieces here are intentionally
 * **not** in OpenPFC: the reference solution is the fundamental Gaussian
 * for the linear heat operator, and the printed format / labels are
 * specific to this application.
 *
 *  - `analytic_gaussian` — closed-form solution on \f$\mathbb{R}^3\f$.
 *  - `fd_extra_metadata` — `fd_order=` + OpenMP info string.
 *  - `report` — RMS-against-analytic reduction + rank-0 three-line output
 *    (`method` / `timing` / `l2_error`).
 *
 * The visitor parameter on `report` lets each solver pick which set of
 * cells to score (FD's `for_each_interior` vs spectral's
 * `for_each_owned`) without `report` itself knowing about `LocalField`.
 */

#include <cmath>
#include <cstddef>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <heat3d/cli.hpp>
#include <heat3d/heat_model.hpp>

namespace heat3d {

/**
 * @brief Fundamental Gaussian solution on \f$\mathbb{R}^3\f$ for the heat
 *        equation with IC \f$u(x,0)=\exp(-|x|^2/(4D))\f$:
 *        \f$u(x,t)=(1+t)^{-3/2}\exp(-|x|^2/(4D(1+t)))\f$.
 *
 * @param r2 \f$|x|^2 = x^2 + y^2 + z^2\f$ (caller already squared it).
 * @param t  Time.
 * @param D  Diffusion coefficient.
 */
inline double analytic_gaussian(double r2, double t, double D) noexcept {
  const double s = 1.0 + t;
  return std::pow(s, -1.5) * std::exp(-r2 / (4.0 * D * s));
}

/**
 * @brief Build the FD-specific extra metadata string: `fd_order=...` plus
 *        OpenMP thread info (when `_OPENMP` is defined).
 */
inline std::string fd_extra_metadata(const RunConfig &cfg) {
  std::ostringstream os;
  os << "fd_order=" << cfg.fd_order;
#if defined(_OPENMP)
  os << " omp_max_threads=" << omp_get_max_threads()
     << " omp_get_num_procs()=" << omp_get_num_procs();
#else
  os << " omp_max_threads=1";
#endif
  return os.str();
}

/**
 * @brief Run-end RMS-against-analytic reduction + rank-0 reporting,
 *        shared by all three solvers.
 *
 * Iterates rank-local owned cells (the visitor decides which set: FD
 * `for_each_interior`, point-wise spectral `for_each_owned`, or a manual
 * loop over an FFT inbox), accumulates `(u - u_exact)^2`, MPI-reduces to
 * rank 0, and prints the canonical three-line `method / timing / l2`
 * summary.
 *
 * @tparam Visitor   Callable `void(cb)` that calls
 *                   `cb(double x, double y, double z, double u_val)`
 *                   once per rank-local owned cell.
 * @param method_tag      Value after `method=` in the metadata line.
 * @param extra_metadata  Appended after `mpi_ranks=...` (e.g.
 *                        `fd_order=`, OpenMP info); pass an empty string
 *                        for none.
 * @param max_elapsed     Wall-clock max of the time-stepping loop across
 *                        ranks (already reduced by the caller).
 * @param l2_note         Parenthetical context appended to the L2 line.
 */
template <class Visitor>
void report(int rank, int nproc, const RunConfig &cfg, const char *method_tag,
            const std::string &extra_metadata, double max_elapsed,
            const char *l2_note, Visitor &&visit_owned_cells) {
  double sum_err2 = 0.0;
  const double t_final = static_cast<double>(cfg.n_steps) * cfg.dt;
  visit_owned_cells([&](double x, double y, double z, double u_val) {
    const double r2 = x * x + y * y + z * z;
    const double uex = analytic_gaussian(r2, t_final, kD);
    const double e = u_val - uex;
    sum_err2 += e * e;
  });
  double g_err2 = 0.0;
  MPI_Reduce(&sum_err2, &g_err2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "heat3d method=" << method_tag << " N=" << cfg.N
              << " n_steps=" << cfg.n_steps << " dt=" << cfg.dt << " D=" << kD
              << " mpi_ranks=" << nproc;
    if (!extra_metadata.empty()) std::cout << " " << extra_metadata;
    std::cout << "\n";
    std::cout << "timing_s=" << max_elapsed << " avg_step_time_s="
              << (max_elapsed / static_cast<double>(cfg.n_steps))
              << " (MPI_MAX across ranks)\n";
    const double rms =
        std::sqrt(g_err2 / static_cast<double>(cfg.N * cfg.N * cfg.N));
    std::cout << "l2_error_vs_R3_analytic_rms=" << rms << " " << l2_note << "\n";
  }
}

} // namespace heat3d
