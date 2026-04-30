// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <iostream>
#include <mpi.h>
#include <string>

#include <wave2d/cli.hpp>
#include <wave2d/wave_model.hpp>

namespace wave2d {

template <class Visitor>
void report(int rank, int nproc, const RunConfig &cfg, const char *method_tag,
            const std::string &extra_metadata, double max_elapsed, const char *note,
            Visitor &&visit_interior) {
  double sum_u2 = 0.0;
  visit_interior([&](double /*x*/, double /*y*/, double /*z*/, double u_val) {
    sum_u2 += u_val * u_val;
  });
  double g_sum = 0.0;
  MPI_Reduce(&sum_u2, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    const double dx = 1.0;
    const double cfl = kC * cfg.dt / dx;
    std::cout << "wave2d method=" << method_tag << " Nx=" << cfg.Nx
              << " Ny=" << cfg.Ny << " n_steps=" << cfg.n_steps << " dt=" << cfg.dt
              << " c=" << kC << " mpi_ranks=" << nproc;
    if (!extra_metadata.empty()) std::cout << " " << extra_metadata;
    std::cout << "\n";
    std::cout << "timing_s=" << max_elapsed << " avg_step_time_s="
              << (max_elapsed / static_cast<double>(cfg.n_steps))
              << " (MPI_MAX across ranks)\n";
    const double rms_u = std::sqrt(g_sum / static_cast<double>(cfg.Nx * cfg.Ny));
    std::cout << "global_rms_u_interior=" << rms_u << " cfl_c_dt_dx=" << cfl << " "
              << note << "\n";
  }
}

inline std::string fd_extra_metadata(const RunConfig &cfg) {
  std::string s = "fd_order=" + std::to_string(cfg.fd_order);
  s += (cfg.y_bc == YBoundaryKind::Dirichlet) ? " y_bc=dirichlet" : " y_bc=neumann";
  return s;
}

} // namespace wave2d
