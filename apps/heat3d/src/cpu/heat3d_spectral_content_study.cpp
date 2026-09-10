// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file heat3d_spectral_content_study.cpp
 * @brief Where the spectral Laplacian is the cheaper route to a given
 *        accuracy, and where a finite-difference stencil is — as a
 *        function of the field's own spectral content.
 *
 * @details
 * ## Why
 *
 * The method comparison in `docs/report/16_scalability.qmd` had two of its
 * three axes measured (cost per step at equal grid, strong scaling at equal
 * grid) and the third — accuracy — measured only for a **single Fourier
 * mode**, by `heat3d_fd_convergence_study`. One smooth mode is the best
 * case a high-order stencil can be given, so the chapter could not turn
 * three measurements into a recommendation and said so. This driver
 * supplies the missing axis for fields of *arbitrary, controlled* spectral
 * content, which closes it.
 *
 * See `heat3d/spectral_content_study.hpp` for the derivation: because the
 * heat equation is linear on a periodic box, every method's error is a
 * closed-form sum over the initial spectrum, so the accuracy map needs no
 * simulation. What it does need is checking, which is the second half of
 * this driver.
 *
 * ## What it writes
 *
 * Three CSVs under `docs/report/data/` (override the directory with
 * `--data-dir`):
 *
 *  - `heat3d_spectral_content_map.csv` — the semi-analytic \f$L^2\f$ error
 *    of each FD order against the content fraction \f$f\f$ of Nyquist.
 *    The spectral path is exact here and carries no row.
 *  - `heat3d_spectral_content_crossover.csv` — for a ladder of accuracy
 *    targets, the coarsest grid each method may use (as \f$f^*\f$) and the
 *    resulting equal-accuracy cost relative to the spectral path, using
 *    the measured per-step costs in `heat3d_method_cost.csv` and an
 *    \f$N^3\f$ cost model at a fixed step count.
 *  - `heat3d_spectral_content_validation.csv` — real RK4 runs of the
 *    shipped FD stack at points on the map, measured against the exact
 *    solution, next to what the map predicted.
 *
 * ## Usage
 *
 *     heat3d_spectral_content_study [--data-dir DIR] [--no-validate]
 *
 * `DIR` defaults to `docs/report/data` resolved against the current
 * working directory — run this from the repository root, or pass an
 * absolute path. Single MPI rank only (an accuracy measurement, not a
 * scaling benchmark); the whole thing is a couple of minutes on a login
 * node and needs no allocation.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include <heat3d/spectral_content_study.hpp>

namespace sc = heat3d::spectral_content;

namespace {

/// Per-step cost of one spatial operator at the reference grid, as measured
/// in `docs/report/data/heat3d_method_cost.csv`.
struct MethodCost {
  /// FD order; 0 denotes the spectral path, matching the CSV's convention.
  int fd_order{0};
  double wall_step_ms{0.0};
};

/// Read `heat3d_method_cost.csv`. Parsing the committed measurement rather
/// than restating it keeps one source of truth for the numbers the
/// crossover depends on; a stale copy here would be invisible.
std::vector<MethodCost> read_method_cost(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<MethodCost> out;
  std::string line;
  bool header_seen = false;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!header_seen) { // "method,fd_order,job,wall_step_ms"
      header_seen = true;
      continue;
    }
    std::istringstream fields(line);
    std::string method, order, job, wall;
    std::getline(fields, method, ',');
    std::getline(fields, order, ',');
    std::getline(fields, job, ',');
    std::getline(fields, wall, ',');
    out.push_back({std::stoi(order), std::stod(wall)});
  }
  if (out.empty()) throw std::runtime_error("no rows parsed from " + path);
  return out;
}

/// Content fractions swept for the map. Stops at 0.9 rather than 1.0: at
/// f = 1 the content edge sits exactly on Nyquist, where the field is on
/// the boundary of being representable at all and the number stops being
/// about the operator.
const std::vector<double> kFractions = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
                                        0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
                                        0.70, 0.75, 0.80, 0.85, 0.90};

const std::vector<int> kOrders = {2, 4, 6, 8, 12};

const std::vector<double> kTargets = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10};

/// Validation points. Two per order, one either side of the crossover
/// fraction for that order, so the check spans the regime where the map
/// says FD wins and the regime where it says spectral does. `n_steps` is
/// sized so RK4's temporal error sits far below the spatial error being
/// measured — the driver halves it on a subset and prints both.
struct ValidationPoint {
  int fd_order;
  int N;
  double f;
  int n_steps;
};
const std::vector<ValidationPoint> kValidationPoints = {
    {2, 64, 0.30, 200},  {2, 64, 0.60, 200},  {4, 64, 0.30, 200},
    {4, 64, 0.60, 200},  {6, 64, 0.40, 400},  {8, 64, 0.30, 400},
    {8, 64, 0.60, 400},  {12, 64, 0.40, 800}, {12, 64, 0.60, 800},
    // One off-grid point: a different N at a fraction already covered, to
    // show the map's N-independence is a property of the runs too.
    {4, 96, 0.30, 200},
};

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1) {
    if (rank == 0) {
      std::cerr << "heat3d_spectral_content_study: single-rank only (got nproc="
                << nproc << "). Run with `srun -n 1` / `mpirun -n 1`.\n";
    }
    MPI_Finalize();
    return 1;
  }

  std::string data_dir = "docs/report/data";
  bool validate = true;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--data-dir" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (arg == "--no-validate") {
      validate = false;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--data-dir DIR] [--no-validate]\n";
      MPI_Finalize();
      return 1;
    }
  }

  std::vector<MethodCost> costs;
  try {
    costs = read_method_cost(data_dir + "/heat3d_method_cost.csv");
  } catch (const std::exception &e) {
    std::cerr << "heat3d_spectral_content_study: " << e.what() << "\n";
    MPI_Finalize();
    return 1;
  }
  double cost_spectral = 0.0;
  std::map<int, double> cost_fd;
  for (const MethodCost &c : costs) {
    if (c.fd_order == 0) {
      cost_spectral = c.wall_step_ms;
    } else {
      cost_fd[c.fd_order] = c.wall_step_ms;
    }
  }
  if (cost_spectral <= 0.0) {
    std::cerr << "heat3d_spectral_content_study: no spectral row in the cost CSV\n";
    MPI_Finalize();
    return 1;
  }

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "heat3d spectral-content accuracy map: L=2*pi, D=" << heat3d::kD
            << ", tau = D*t*k_c^2 = " << sc::kDiffusionTime
            << ", content threshold = " << sc::kContentThreshold << "\n"
            << "f is the fraction of Nyquist at which the initial amplitude "
               "spectrum is down to that threshold.\n"
            << "The spectral path's spatial error is identically zero and is "
               "not tabulated.\n\n";

  // ---- 1. the error map -------------------------------------------------
  std::cout << std::left << std::setw(8) << "f";
  for (int order : kOrders) std::cout << std::setw(15) << ("FD-" + std::to_string(order));
  std::cout << "\n";
  std::vector<std::vector<double>> map_rows;
  map_rows.reserve(kFractions.size());
  for (double f : kFractions) {
    std::vector<double> row;
    row.reserve(kOrders.size());
    std::cout << std::left << std::setw(8) << std::fixed << std::setprecision(2) << f
              << std::scientific << std::setprecision(4);
    for (int order : kOrders) {
      const double e = sc::predict_l2_error(order, f, sc::kMapGrid);
      row.push_back(e);
      std::cout << std::setw(15) << e;
    }
    std::cout << "\n";
    map_rows.push_back(std::move(row));
  }

  {
    std::ofstream csv(data_dir + "/heat3d_spectral_content_map.csv");
    if (!csv) {
      std::cerr << "cannot write the map CSV into " << data_dir << "\n";
      MPI_Finalize();
      return 1;
    }
    csv << "# Semi-analytic L2 error of the central-FD Laplacian against the "
           "exact solution\n"
        << "# of the linear heat equation, for a truncated-Gaussian field whose "
           "amplitude\n"
        << "# spectrum is down to " << sc::kContentThreshold
        << " of its peak at a fraction f of Nyquist.\n"
        << "# From heat3d_spectral_content_study; L=2*pi, D=" << heat3d::kD
        << ", tau = D*t*k_c^2 = " << sc::kDiffusionTime << ", mode cube N="
        << sc::kMapGrid << ".\n"
        << "# The spectral operator's spatial error is identically zero, so it "
           "has no rows here.\n"
        << "fd_order,content_fraction,N,tau,l2_error\n";
    csv << std::scientific << std::setprecision(10);
    for (std::size_t oi = 0; oi < kOrders.size(); ++oi) {
      for (std::size_t fi = 0; fi < kFractions.size(); ++fi) {
        csv << kOrders[oi] << "," << kFractions[fi] << "," << sc::kMapGrid << ","
            << sc::kDiffusionTime << "," << map_rows[fi][oi] << "\n";
      }
    }
  }

  // ---- 2. equal-accuracy cost and the crossover -------------------------
  std::cout << "\nCrossover fraction per order -- FD is the cheaper route to a "
               "fixed accuracy\n"
               "iff it still meets that accuracy with content above this "
               "fraction of Nyquist:\n";
  for (int order : kOrders) {
    if (!cost_fd.count(order)) continue;
    std::cout << "  FD-" << order << ": " << std::fixed << std::setprecision(4)
              << sc::crossover_fraction(cost_fd[order], cost_spectral)
              << "  (cost " << std::setprecision(2) << cost_fd[order] << " ms vs "
              << cost_spectral << " ms)\n";
  }
  std::cout << std::scientific << std::setprecision(4) << "\n"
            << std::left << std::setw(12) << "target L2";
  for (int order : kOrders)
    std::cout << std::setw(22) << ("FD-" + std::to_string(order));
  std::cout << "cheapest\n";

  struct CrossoverRow {
    double target;
    int fd_order;
    double f_star;
    double relative_cost;
    bool beats_spectral;
  };
  std::vector<CrossoverRow> crossover_rows;
  for (double eps : kTargets) {
    std::cout << std::left << std::setw(12) << eps;
    std::string best = "spectral";
    double best_cost = 1.0;
    for (int order : kOrders) {
      if (!cost_fd.count(order)) continue;
      const double fs = sc::content_fraction_at(order, eps);
      const double rel =
          sc::equal_accuracy_cost_ratio(fs, cost_fd[order], cost_spectral);
      crossover_rows.push_back(
          {eps, order, fs, rel,
           sc::fd_cheaper_than_spectral(fs, cost_fd[order], cost_spectral)});
      std::ostringstream cell;
      cell << std::fixed << std::setprecision(3) << "f*=" << fs << " x"
           << std::scientific << std::setprecision(2) << rel;
      std::cout << std::setw(22) << cell.str();
      if (rel < best_cost) {
        best_cost = rel;
        best = "FD-" + std::to_string(order);
      }
    }
    std::cout << best << "\n";
  }

  {
    std::ofstream csv(data_dir + "/heat3d_spectral_content_crossover.csv");
    if (!csv) {
      std::cerr << "cannot write the crossover CSV into " << data_dir << "\n";
      MPI_Finalize();
      return 1;
    }
    csv << "# Equal-accuracy cost of each spatial operator, from "
           "heat3d_spectral_content_study.\n"
        << "# content_fraction is the coarsest grid the method may use and "
           "still meet target_l2,\n"
        << "# expressed as the fraction of Nyquist the field's content then "
           "reaches (see the map CSV).\n"
        << "# relative_cost = (wall_step_ms / content_fraction^3) / "
           "spectral wall_step_ms: an N^3 cost\n"
        << "# model at a FIXED step count, using the measured per-step costs "
           "in heat3d_method_cost.csv\n"
        << "# (N=1024, 8 GCDs). That model is generous to FD, whose explicit "
           "dt also falls as dx^2.\n"
        << "# The spectral operator is exact, so its content_fraction is 1 "
           "(representation only)\n"
        << "# and its relative_cost is 1 at every target.\n"
        << "target_l2,method,fd_order,content_fraction,wall_step_ms,relative_cost,"
           "beats_spectral\n";
    csv << std::scientific << std::setprecision(10);
    for (double eps : kTargets) {
      csv << eps << ",spectral,0,1.0," << cost_spectral << ",1.0,\n";
    }
    for (const CrossoverRow &r : crossover_rows) {
      csv << r.target << ",fd," << r.fd_order << "," << r.f_star << ","
          << cost_fd[r.fd_order] << "," << r.relative_cost << ","
          << (r.beats_spectral ? "yes" : "no") << "\n";
    }
  }

  // ---- 3. validation against real runs ----------------------------------
  if (validate) {
    std::cout << "\nValidation: real RK4 runs of the shipped FD stack "
                 "(pfc::data::Field + HaloExchange +\n"
                 "FDGradient<HeatGrads>) against the exact solution of the same "
                 "sampled field.\n"
              << std::left << std::setw(8) << "order" << std::setw(6) << "N"
              << std::setw(7) << "f" << std::setw(8) << "steps" << std::setw(15)
              << "predicted" << std::setw(15) << "measured" << std::setw(12)
              << "meas/pred" << "half-dt meas/pred\n";

    std::vector<sc::ValidationCase> cases;
    for (const ValidationPoint &p : kValidationPoints) {
      const sc::ValidationCase c =
          sc::run_validation(p.fd_order, p.N, p.f, sc::kDiffusionTime, p.n_steps);
      const sc::ValidationCase c2 = sc::run_validation(
          p.fd_order, p.N, p.f, sc::kDiffusionTime, 2 * p.n_steps);
      cases.push_back(c);
      std::cout << std::left << std::setw(8) << c.fd_order << std::setw(6) << c.N
                << std::fixed << std::setprecision(2) << std::setw(7) << c.f
                << std::setw(8) << c.n_steps << std::scientific
                << std::setprecision(5) << std::setw(15) << c.predicted_l2
                << std::setw(15) << c.measured_l2 << std::fixed
                << std::setprecision(6) << std::setw(12) << c.ratio << c2.ratio
                << "\n";
      // Keep the halved-step run in the CSV as its own row: it is the
      // evidence that the temporal error is not what is being measured.
      cases.push_back(c2);
    }

    std::ofstream csv(data_dir + "/heat3d_spectral_content_validation.csv");
    if (!csv) {
      std::cerr << "cannot write the validation CSV into " << data_dir << "\n";
      MPI_Finalize();
      return 1;
    }
    csv << "# Validation of the semi-analytic map above against real runs of "
           "the shipped FD\n"
        << "# stack, from heat3d_spectral_content_study. Classical RK4 in time "
           "so the O(dt)\n"
        << "# error of the driver's forward Euler does not swamp a spatial "
           "error of 1e-8;\n"
        << "# each point appears twice, at n_steps and 2*n_steps, and the two "
           "agreeing is the\n"
        << "# evidence that dt is not what is being measured.\n"
        << "# measured_l2 and predicted_l2 are both RMS errors relative to the "
           "initial RMS.\n"
        << "fd_order,N,content_fraction,tau,n_steps,dt,t_final,predicted_l2,"
           "measured_l2,ratio\n";
    csv << std::scientific << std::setprecision(10);
    for (const sc::ValidationCase &c : cases) {
      csv << c.fd_order << "," << c.N << "," << c.f << "," << c.tau << ","
          << c.n_steps << "," << c.dt << "," << c.t_final << "," << c.predicted_l2
          << "," << c.measured_l2 << "," << c.ratio << "\n";
    }
  }

  std::cout << "\nwrote " << data_dir << "/heat3d_spectral_content_map.csv\n"
            << "wrote " << data_dir << "/heat3d_spectral_content_crossover.csv\n";
  if (validate) {
    std::cout << "wrote " << data_dir
              << "/heat3d_spectral_content_validation.csv\n";
  }

  MPI_Finalize();
  return 0;
}
