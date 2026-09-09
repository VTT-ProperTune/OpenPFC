// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_fd_convergence_study.cpp
 * @brief Order-of-accuracy sweep for the compact FD Laplacian
 *        (`pfc::gradient::FDGradient<HeatGrads>`), single mode, N x order.
 *
 * @details
 * Runs `heat3d::convergence::run_case(order, N)` (see
 * `heat3d/convergence_study.hpp` for how this has zero time-discretization
 * error by construction, not merely small) for every `(fd_order, N)` pair
 * in the fixed sweep below, prints a human-readable table (design vs
 * observed order, plus the eigenvalue self-check), and writes the raw
 * per-case rows as a CSV that `docs/report/figures/make_figures.py` turns
 * into a log-log figure.
 *
 * Single MPI rank only (the sweep is a numerical-accuracy measurement, not
 * a scaling benchmark; see `heat3d::convergence::run_case`).
 *
 * Usage:
 *   heat3d_fd_convergence_study [output.csv]
 *
 * `output.csv` defaults to `docs/report/data/heat3d_fd_order_convergence.csv`
 * resolved against the current working directory — run this from the repo
 * root (or pass an absolute path).
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include <heat3d/convergence_study.hpp>

namespace {
using heat3d::convergence::ConvergenceCase;
} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  if (nproc != 1) {
    if (rank == 0) {
      std::cerr << "heat3d_fd_convergence_study: single-rank only (got nproc="
                << nproc << "). Run with `srun -n 1` / `mpirun -n 1`.\n";
    }
    MPI_Finalize();
    return 1;
  }

  const std::string out_path =
      (argc > 1) ? argv[1] : "docs/report/data/heat3d_fd_order_convergence.csv";

  const std::vector<int> orders = {2, 4, 6, 8, 10, 12};
  const std::vector<int> grid_sizes = {16, 24, 32, 48, 64};

  std::vector<ConvergenceCase> cases;
  cases.reserve(orders.size() * grid_sizes.size());

  std::cout << std::scientific << std::setprecision(6);
  std::cout << "heat3d FD order-convergence study: mode=" << heat3d::convergence::kMode
            << " L=" << heat3d::convergence::kDomainLength
            << " D=" << heat3d::kD << " t_final=" << heat3d::convergence::kFinalTime
            << " (exact time evolution -- no dt)\n";
  std::cout << std::left << std::setw(10) << "order" << std::setw(6) << "N"
            << std::setw(14) << "dx" << std::setw(16) << "eigenvalue" << std::setw(14)
            << "eig_resid" << std::setw(16) << "l2_error" << "observed_order\n";

  for (int order : orders) {
    ConvergenceCase prev;
    bool have_prev = false;
    for (int N : grid_sizes) {
      const ConvergenceCase c = heat3d::convergence::run_case(order, N);
      cases.push_back(c);

      std::cout << std::left << std::setw(10) << c.fd_order << std::setw(6) << c.N
                << std::setw(14) << c.dx << std::setw(16) << c.eigenvalue
                << std::setw(14) << c.eigenvalue_residual << std::setw(16) << c.l2_error;
      if (have_prev) {
        std::cout << heat3d::convergence::observed_order(prev, c);
      } else {
        std::cout << "n/a";
      }
      std::cout << "\n";

      prev = c;
      have_prev = true;
    }
  }

  std::ofstream csv(out_path);
  if (!csv) {
    std::cerr << "heat3d_fd_convergence_study: could not open '" << out_path
               << "' for writing\n";
    MPI_Finalize();
    return 1;
  }
  csv << "# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd\n";
  csv << "# SPDX-License-Identifier: AGPL-3.0-or-later\n";
  csv << "fd_order,design_order,N,dx,eigenvalue,eigenvalue_residual,l2_error,observed_order\n";
  csv << std::scientific << std::setprecision(10);

  for (int order : orders) {
    ConvergenceCase prev;
    bool have_prev = false;
    for (const ConvergenceCase &c : cases) {
      if (c.fd_order != order) continue;
      csv << c.fd_order << "," << c.fd_order << "," << c.N << "," << c.dx << ","
          << c.eigenvalue << "," << c.eigenvalue_residual << "," << c.l2_error << ",";
      if (have_prev) {
        csv << heat3d::convergence::observed_order(prev, c);
      }
      csv << "\n";
      prev = c;
      have_prev = true;
    }
  }
  csv.close();

  if (rank == 0) {
    std::cout << "wrote " << out_path << "\n";
  }

  MPI_Finalize();
  return 0;
}
