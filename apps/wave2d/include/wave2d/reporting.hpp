// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file reporting.hpp
 * @brief The `global_rms_u_interior` observable and its rank-0 report line.
 *
 * @details
 * "Interior" here means the *global* interior: the cells left after trimming a
 * `margin`-wide shell off each **global** axis that is thick enough to carry
 * one. Two properties follow, and both are the point of this file.
 *
 * 1. A degenerate axis is never trimmed. `wave2d` is an `nz == 1` slab, so
 *    trimming `[margin, 1 - margin)` in z leaves nothing and the reduction
 *    silently summed zero cells — the app advertised an observable that was
 *    identically `0` for every configuration. z is a dimension the model does
 *    not have; excluding it from the exclusion is what "the x/y interior of a
 *    2-D slab" means.
 * 2. Trimming in *global* index space, not per-rank, makes the number
 *    independent of the rank count. Trimming each rank's owned box would eat a
 *    shell at every internal subdomain seam, so the same physics would report
 *    a different RMS on 1 rank and on 4.
 *
 * `InteriorStats` carries the visited cell count alongside the sum so an empty
 * reduction is reportable rather than indistinguishable from a field that
 * legitimately summed to zero: `report()` prints `interior_cells=` and returns
 * `false` when nothing was visited.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mpi.h>
#include <string>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc_apps/mpi_report.hpp>
#include <wave2d/cli.hpp>
#include <wave2d/wave_model.hpp>

namespace wave2d {

/// Sum of `u^2` over the visited cells, and how many cells that was.
struct InteriorStats {
  double sum_sq = 0.0;
  std::int64_t count = 0;
};

/**
 * @brief Cells to trim off both ends of one global axis.
 *
 * `margin` if the axis can spare two of them and still leave a cell standing,
 * otherwise `0`. An `nz == 1` slab therefore keeps its single z layer.
 */
[[nodiscard]] inline int interior_margin(int global_extent, int margin) noexcept {
  return (global_extent > 2 * margin) ? margin : 0;
}

/**
 * @brief This rank's contribution to the global-interior sum of `u^2`.
 *
 * @param u      Owned field; ghost cells are never read.
 * @param margin Shell width to trim off each non-degenerate global axis
 *               (the FD stencil half-width, in practice).
 */
template <class T>
[[nodiscard]] inline InteriorStats
interior_stats(const pfc::data::Field<T, pfc::HostSpace> &u, int margin) {
  const auto gsize = pfc::domain::get_size(u.domain());
  const auto lo = u.box().low;
  const auto sz = u.local_size();

  std::array<int, 3> begin{};
  std::array<int, 3> end{};
  for (int d = 0; d < 3; ++d) {
    const int m = interior_margin(gsize[d], margin);
    // Intersect the global interior [m, gsize-m) with this rank's owned box.
    begin[d] = std::max(0, m - lo[d]);
    end[d] = std::min(sz[d], gsize[d] - m - lo[d]);
  }

  InteriorStats s;
  for (int k = begin[2]; k < end[2]; ++k) {
    for (int j = begin[1]; j < end[1]; ++j) {
      for (int i = begin[0]; i < end[0]; ++i) {
        const double val = static_cast<double>(u(i, j, k));
        s.sum_sq += val * val;
        ++s.count;
      }
    }
  }
  return s;
}

/**
 * @brief Print the rank-0 run summary.
 *
 * @param local Interior reduction for this rank (see `interior_stats`).
 * @return `false` if the interior was empty on every rank, i.e. the observable
 *         is not defined for this configuration. Callers should treat that as
 *         a run failure — a printed `0` that means "nothing was summed" is the
 *         defect this signature exists to prevent.
 */
[[nodiscard]] inline bool report(int rank, int nproc, const RunConfig &cfg,
                                 const char *method_tag,
                                 const std::string &extra_metadata,
                                 double max_elapsed, const char *note,
                                 const InteriorStats &local) {
  const double g_sum = pfc::apps::reduce_sum(local.sum_sq, MPI_COMM_WORLD);
  std::int64_t g_count = 0;
  MPI_Allreduce(&local.count, &g_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    const double dx = 1.0;
    const double cfl = kC * cfg.dt / dx;
    std::cout << "wave2d method=" << method_tag << " Nx=" << cfg.Nx
              << " Ny=" << cfg.Ny << " n_steps=" << cfg.n_steps << " dt=" << cfg.dt
              << " c=" << kC << " mpi_ranks=" << nproc;
    if (!extra_metadata.empty()) std::cout << " " << extra_metadata;
    std::cout << "\n";
    pfc::apps::print_timing_line(std::cout, max_elapsed, cfg.n_steps);
    if (g_count == 0) {
      std::cerr << "wave2d: interior reduction visited no cells; "
                   "global_rms_u_interior is undefined for this configuration\n";
      std::cout << "global_rms_u_interior=undefined interior_cells=0"
                << " cfl_c_dt_dx=" << cfl << " " << note << "\n";
    } else {
      const double rms_u = std::sqrt(g_sum / static_cast<double>(g_count));
      std::cout << "global_rms_u_interior=" << rms_u
                << " interior_cells=" << g_count << " cfl_c_dt_dx=" << cfl << " "
                << note << "\n";
    }
  }
  return g_count > 0;
}

inline std::string fd_extra_metadata(const RunConfig &cfg) {
  std::string s = "fd_order=" + std::to_string(cfg.fd_order);
  s += (cfg.y_bc == YBoundaryKind::Dirichlet) ? " y_bc=dirichlet" : " y_bc=neumann";
  return s;
}

} // namespace wave2d
