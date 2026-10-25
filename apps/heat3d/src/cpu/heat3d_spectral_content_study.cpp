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
 * The method comparison for this heat application had two of its
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
 * Three CSVs under `out/report/data/` (override the directory with
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
 *  - `heat3d_spectral_family_{definitions,map,diagnostics,selection}.csv`
 *    — occupancy-matched Gaussian / top-hat / exponential families, using
 *    the same admitted CPU/GPU cost tables (`--families-only` skips the
 *    Gaussian RK4 block).
 *  - `heat3d_spectral_family_heldout.csv` — predeclared held-out RK4 of
 *    the shipped FD stack on those families (`--held-out-families`).
 *
 * ## Usage
 *
 *     heat3d_spectral_content_study [--data-dir DIR] [--no-validate]
 *     heat3d_spectral_content_study [--data-dir DIR] [--held-out-only]
 *     heat3d_spectral_content_study [--data-dir DIR] [--held-out-families]
 *     heat3d_spectral_content_study [--data-dir DIR] [--families-only]
 *
 * `DIR` defaults to `out/report/data` resolved against the current
 * working directory — run this from the repository root, or pass an
 * absolute path. Single MPI rank only (an accuracy measurement, not a
 * scaling benchmark). The Gaussian map is a couple of minutes; the
 * family held-out cosine-sum ICs belong on a compute node.
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>

#include <heat3d/spectral_content_study.hpp>
#include <openpfc/kernel/field/periodic_spectra.hpp>

namespace sc = heat3d::spectral_content;
namespace sp = pfc::field::spectra;

namespace {

/// Per-step cost of one spatial operator at the reference grid, as measured
/// in `out/report/data/heat3d_method_cost.csv`.
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

const std::vector<double> kTargets = {1e-2, 1e-3, 1e-4, 1e-5,
                                      1e-6, 1e-7, 1e-8, 1e-10};

/// Family comparison uses the manuscript loose/tight pair plus the interior
/// ladder, omitting 1e-7 and 1e-10 which do not change the declared rankings.
const std::vector<double> kFamilyTargets = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8};

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
    {2, 64, 0.30, 200},
    {2, 64, 0.60, 200},
    {4, 64, 0.30, 200},
    {4, 64, 0.60, 200},
    {6, 64, 0.40, 400},
    {8, 64, 0.30, 400},
    {8, 64, 0.60, 400},
    {12, 64, 0.40, 800},
    {12, 64, 0.60, 800},
    // One off-grid point: a different N at a fraction already covered, to
    // show the map's N-independence is a property of the runs too.
    {4, 96, 0.30, 200},
};

/// Protocol held-out pair (Paper A linear slice): orders 2 and 12 at
/// f = 0.3 and 0.6 on N = 128, not in kValidationPoints.
const std::vector<ValidationPoint> kHeldOutPoints = {
    {2, 128, 0.30, 200},
    {2, 128, 0.60, 200},
    {12, 128, 0.30, 800},
    {12, 128, 0.60, 800},
};

/// Predeclared non-Gaussian referee gate (). Same orders,
/// occupancies, step counts and 1e-6 residual bound as kHeldOutPoints.
/// N=32 is held out from the family Parseval map (N=64) and from the
/// N=16 --validate-families smoke; cosine-sum ICs forbid N=128.
const std::vector<ValidationPoint> kFamilyHeldOutPoints = {
    {2, 32, 0.30, 200},
    {2, 32, 0.60, 200},
    {12, 32, 0.30, 800},
    {12, 32, 0.60, 800},
};

void write_family_study(const std::string &data_dir,
                        const std::map<int, double> &cost_gpu,
                        double cost_gpu_spectral,
                        const std::map<int, double> &cost_cpu,
                        double cost_cpu_spectral) {
  std::ofstream def(data_dir + "/heat3d_spectral_family_definitions.csv");
  if (!def) throw std::runtime_error("cannot write family definitions CSV");
  def << "# Occupancy-matched periodic families. f is the fraction of Nyquist "
         "at which the amplitude prescription reaches "
      << sp::kContentThreshold << " of peak (top-hat: support cutoff).\n"
      << "family_id,name,geometry,amplitude_at_nu_eq_1,occupancy_threshold\n";
  for (const sp::SpectrumFamily &fam : sp::builtin_families()) {
    def << fam.id << ",\"" << fam.name << "\","
        << (fam.geometry == sp::WeightGeometry::IsotropicRadial ? "isotropic"
                                                                : "separable")
        << "," << std::scientific << std::setprecision(10) << fam.amplitude(1.0)
        << "," << sp::kContentThreshold << "\n";
  }
  def.flush();

  std::ofstream map(data_dir + "/heat3d_spectral_family_map.csv");
  std::ofstream diag(data_dir + "/heat3d_spectral_family_diagnostics.csv");
  std::ofstream sel(data_dir + "/heat3d_spectral_family_selection.csv");
  if (!map || !diag || !sel)
    throw std::runtime_error("cannot write family study CSVs");
  map << "# Parseval spatial L2 error vs occupancy f for each family. "
         "fd_order 0 is the spectral operator (identically zero).\n"
      << "family_id,fd_order,content_fraction,N,dim,tau,l2_error\n";
  diag << "# Explanatory diagnostics at matched f; not a fitted replacement "
          "for occupancy.\n"
       << "family_id,content_fraction,N,dim,fd_order,energy_high_k,"
          "energy_near_cutoff,moment2_over_nyquist2,moment4_over_nyquist4,"
          "weighted_d2_defect\n";
  sel << "# Equal-accuracy ranking using admitted Heat3D per-step costs "
         "(N^3, fixed step count). row_kind=order is one stencil; "
         "row_kind=cheapest is the winner (fd_order 0 = spectral). "
         "attainable=no means the FD target is not met on the admitted "
         "interval; those rows are not ranked.\n"
      << "row_kind,family_id,hardware,target_l2,fd_order,content_fraction,"
         "relative_cost,beats_spectral,attainable\n";
  map << std::scientific << std::setprecision(10);
  diag << std::scientific << std::setprecision(10);
  sel << std::scientific << std::setprecision(10);

  constexpr int kFamilyGrid = 64;
  constexpr int kDim = 3;
  std::cout << "\nSpectral-family Parseval maps (N=" << kFamilyGrid
            << ", dim=" << kDim << ")\n";
  for (const sp::SpectrumFamily &fam : sp::builtin_families()) {
    std::cout << "  family " << fam.id << "\n";
    for (double f : kFractions) {
      const auto d = sp::diagnose_spectrum(fam, f, kFamilyGrid, 2, kDim);
      diag << fam.id << "," << f << "," << kFamilyGrid << "," << kDim << ",2,"
           << d.energy_high_k << "," << d.energy_near_cutoff << ","
           << d.moment2_over_nyquist2 << "," << d.moment4_over_nyquist4 << ","
           << d.weighted_d2_defect << "\n";
      map << fam.id << ",0," << f << "," << kFamilyGrid << "," << kDim << ","
          << sp::kDiffusionTime << ",0\n";
      for (int order : kOrders) {
        const double e = sp::predict_heat_l2_error(fam, order, f, kFamilyGrid,
                                                   sp::kDiffusionTime, kDim);
        map << fam.id << "," << order << "," << f << "," << kFamilyGrid << ","
            << kDim << "," << sp::kDiffusionTime << "," << e << "\n";
      }
    }
    auto emit_hw = [&](const char *hw, const std::map<int, double> &cost,
                       double cost_spec) {
      if (!(cost_spec > 0.0)) return;
      std::vector<std::pair<int, double>> pairs;
      for (int order : kOrders) {
        auto it = cost.find(order);
        if (it == cost.end()) continue;
        pairs.emplace_back(order, it->second);
      }
      for (double eps : kFamilyTargets) {
        const auto best = sp::pick_equal_accuracy(fam, eps, pairs, cost_spec,
                                                  sp::kDiffusionTime, kDim);
        for (const auto &[order, c] : pairs) {
          const auto match =
              sp::content_fraction_at(fam, order, eps, sp::kDiffusionTime, kDim);
          sel << "order," << fam.id << "," << hw << "," << eps << "," << order
              << ",";
          if (match.attainable) {
            const double rel = sp::equal_accuracy_cost_ratio(match.f, c, cost_spec);
            sel << match.f << "," << rel << ","
                << (sp::fd_cheaper_than_spectral(match.f, c, cost_spec) ? "yes"
                                                                        : "no")
                << ",yes\n";
          } else {
            sel << ",,no,no\n";
          }
        }
        sel << "cheapest," << fam.id << "," << hw << "," << eps << ","
            << best.fd_order << "," << best.f_star << "," << best.relative_cost
            << "," << (best.fd_order != 0 ? "yes" : "no") << ","
            << (best.attainable ? "yes" : "no") << "\n";
      }
    };
    emit_hw("lumi-g", cost_gpu, cost_gpu_spectral);
    emit_hw("lumi-c", cost_cpu, cost_cpu_spectral);
    map.flush();
    diag.flush();
    sel.flush();
  }
  std::cout << "wrote " << data_dir << "/heat3d_spectral_family_definitions.csv\n"
            << "wrote " << data_dir << "/heat3d_spectral_family_map.csv\n"
            << "wrote " << data_dir << "/heat3d_spectral_family_diagnostics.csv\n"
            << "wrote " << data_dir << "/heat3d_spectral_family_selection.csv\n";
}

void write_family_heldout(const std::string &data_dir) {
  std::cout << "Held-out family RK4 (N=32, orders 2 and 12, f=0.3 and 0.6)\n"
            << "relative |ratio-1| bound " << sc::kFamilyHeldOutRatioTol
            << " (frozen; not retuned)\n";
  std::vector<sc::ValidationCase> cases;
  double max_abs_rel = 0.0;
  for (const sp::SpectrumFamily &fam : sp::builtin_families()) {
    std::cout << "  family " << fam.id << "\n";
    for (const ValidationPoint &p : kFamilyHeldOutPoints) {
      const sc::ValidationCase c = sc::run_validation(
          p.fd_order, p.N, p.f, sc::kDiffusionTime, p.n_steps, fam);
      const sc::ValidationCase c2 = sc::run_validation(
          p.fd_order, p.N, p.f, sc::kDiffusionTime, 2 * p.n_steps, fam);
      cases.push_back(c);
      cases.push_back(c2);
      const double r1 = std::abs(c.ratio - 1.0);
      const double r2 = std::abs(c2.ratio - 1.0);
      max_abs_rel = std::max(max_abs_rel, std::max(r1, r2));
      std::cout << "    order=" << c.fd_order << " f=" << c.f << " ratio=" << c.ratio
                << " half-dt ratio=" << c2.ratio << "\n";
    }
  }
  const std::string path = data_dir + "/heat3d_spectral_family_heldout.csv";
  std::ofstream csv(path);
  if (!csv) throw std::runtime_error("cannot write " + path);
  csv << "# Predeclared held-out RK4 of the shipped FD stack against Parseval "
         "for every builtin occupancy-matched family. Same orders, f, n/"
         "2n steps and 1e-6 |ratio-1| bound as the Gaussian N=128 protocol; "
         "N=32 because non-Gaussian ICs are cosine sums. Do not retune.\n"
      << "family_id,fd_order,N,content_fraction,tau,n_steps,dt,t_final,"
         "predicted_l2,measured_l2,ratio\n";
  csv << std::scientific << std::setprecision(10);
  for (const sc::ValidationCase &c : cases) {
    csv << c.family_id << "," << c.fd_order << "," << c.N << "," << c.f << ","
        << c.tau << "," << c.n_steps << "," << c.dt << "," << c.t_final << ","
        << c.predicted_l2 << "," << c.measured_l2 << "," << c.ratio << "\n";
  }
  std::cout << "wrote " << path << "\n"
            << "max |ratio-1| = " << std::scientific << std::setprecision(4)
            << max_abs_rel
            << (sc::family_heldout_passes(max_abs_rel) ? " PASS\n" : " FAIL\n");
  sc::require_family_heldout_ok(max_abs_rel);
}

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

  std::string data_dir = "out/report/data";
  bool validate = true;
  bool held_out_only = false;
  bool held_out_families = false;
  bool run_families = true;
  bool families_only = false;
  bool validate_families = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--data-dir" && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (arg == "--no-validate") {
      validate = false;
    } else if (arg == "--held-out-only") {
      held_out_only = true;
    } else if (arg == "--held-out-families") {
      held_out_families = true;
    } else if (arg == "--no-families") {
      run_families = false;
    } else if (arg == "--families-only") {
      families_only = true;
      validate = false;
    } else if (arg == "--validate-families") {
      validate_families = true;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--data-dir DIR] [--no-validate|--held-out-only|"
                   "--held-out-families]"
                   " [--no-families|--families-only] [--validate-families]\n";
      MPI_Finalize();
      return 1;
    }
  }

  if (held_out_only) {
    std::cout << "Held-out N=128 RK4 (orders 2 and 12, f=0.3 and 0.6)\n";
    std::vector<sc::ValidationCase> cases;
    for (const ValidationPoint &p : kHeldOutPoints) {
      const sc::ValidationCase c =
          sc::run_validation(p.fd_order, p.N, p.f, sc::kDiffusionTime, p.n_steps);
      const sc::ValidationCase c2 = sc::run_validation(
          p.fd_order, p.N, p.f, sc::kDiffusionTime, 2 * p.n_steps);
      cases.push_back(c);
      cases.push_back(c2);
      std::cout << "order=" << c.fd_order << " N=" << c.N << " f=" << c.f
                << " ratio=" << c.ratio << " half-dt ratio=" << c2.ratio << "\n";
    }
    const std::string path = data_dir + "/heat3d_spectral_content_heldout.csv";
    std::ofstream csv(path);
    if (!csv) {
      std::cerr << "cannot write " << path << "\n";
      MPI_Finalize();
      return 1;
    }
    csv << "# Held-out N=128 RK4 residuals, Paper A linear protocol.\n"
        << "fd_order,N,content_fraction,tau,n_steps,dt,t_final,predicted_l2,"
           "measured_l2,ratio\n";
    csv << std::scientific << std::setprecision(10);
    for (const sc::ValidationCase &c : cases) {
      csv << c.fd_order << "," << c.N << "," << c.f << "," << c.tau << ","
          << c.n_steps << "," << c.dt << "," << c.t_final << "," << c.predicted_l2
          << "," << c.measured_l2 << "," << c.ratio << "\n";
    }
    std::cout << "wrote " << path << "\n";
    MPI_Finalize();
    return 0;
  }

  if (held_out_families) {
    try {
      write_family_heldout(data_dir);
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      MPI_Finalize();
      return 1;
    }
    MPI_Finalize();
    return 0;
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

  double cost_cpu_spectral = 0.0;
  std::map<int, double> cost_cpu;
  try {
    const auto cpu_rows =
        read_method_cost(data_dir + "/heat3d_method_cost_lumi_c.csv");
    for (const MethodCost &c : cpu_rows) {
      if (c.fd_order == 0)
        cost_cpu_spectral = c.wall_step_ms;
      else
        cost_cpu[c.fd_order] = c.wall_step_ms;
    }
  } catch (const std::exception &e) {
    std::cerr << "heat3d_spectral_content_study: CPU cost table skipped ("
              << e.what() << ")\n";
  }

  if (families_only) {
    try {
      write_family_study(data_dir, cost_fd, cost_spectral, cost_cpu,
                         cost_cpu_spectral);
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      MPI_Finalize();
      return 1;
    }
    MPI_Finalize();
    return 0;
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
  for (int order : kOrders)
    std::cout << std::setw(15) << ("FD-" + std::to_string(order));
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
        << ", tau = D*t*k_c^2 = " << sc::kDiffusionTime
        << ", mode cube N=" << sc::kMapGrid << ".\n"
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
              << sc::crossover_fraction(cost_fd[order], cost_spectral) << "  (cost "
              << std::setprecision(2) << cost_fd[order] << " ms vs " << cost_spectral
              << " ms)\n";
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
    bool attainable;
  };
  std::vector<CrossoverRow> crossover_rows;
  for (double eps : kTargets) {
    std::cout << std::left << std::setw(12) << eps;
    std::string best = "spectral";
    double best_cost = 1.0;
    for (int order : kOrders) {
      if (!cost_fd.count(order)) continue;
      const auto match = sc::content_fraction_at(order, eps);
      std::ostringstream cell;
      if (!match.attainable) {
        crossover_rows.push_back({eps, order, 0.0, 0.0, false, false});
        cell << "unattainable";
        std::cout << std::setw(22) << cell.str();
        continue;
      }
      const double rel =
          sc::equal_accuracy_cost_ratio(match.f, cost_fd[order], cost_spectral);
      crossover_rows.push_back(
          {eps, order, match.f, rel,
           sc::fd_cheaper_than_spectral(match.f, cost_fd[order], cost_spectral),
           true});
      cell << std::fixed << std::setprecision(3) << "f*=" << match.f << " x"
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
        << "# and its relative_cost is 1 at every target. attainable=no "
           "marks FD rows whose target is not met on the admitted interval.\n"
        << "target_l2,method,fd_order,content_fraction,wall_step_ms,relative_cost,"
           "beats_spectral,attainable\n";
    csv << std::scientific << std::setprecision(10);
    for (double eps : kTargets) {
      csv << eps << ",spectral,0,1.0," << cost_spectral << ",1.0,,yes\n";
    }
    for (const CrossoverRow &r : crossover_rows) {
      csv << r.target << ",fd," << r.fd_order << ",";
      if (r.attainable) {
        csv << r.f_star << "," << cost_fd[r.fd_order] << "," << r.relative_cost
            << "," << (r.beats_spectral ? "yes" : "no") << ",yes\n";
      } else {
        csv << "," << cost_fd[r.fd_order] << ",,no,no\n";
      }
    }
  }

  if (run_families) {
    try {
      write_family_study(data_dir, cost_fd, cost_spectral, cost_cpu,
                         cost_cpu_spectral);
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      MPI_Finalize();
      return 1;
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

  if (validate_families) {
    std::cout << "\nFamily RK4 checks: N=16, FD-2, f=0.5, 80 steps (and 160)\n";
    std::ofstream csv(data_dir + "/heat3d_spectral_family_validation.csv");
    if (!csv) {
      std::cerr << "cannot write family validation CSV\n";
      MPI_Finalize();
      return 1;
    }
    csv << "# Held-in RK4 of shipped FD-2 against Parseval for every builtin "
           "family. Modest N because non-Gaussian ICs are cosine sums.\n"
        << "family_id,fd_order,N,content_fraction,tau,n_steps,predicted_l2,"
           "measured_l2,ratio\n";
    csv << std::scientific << std::setprecision(10);
    for (const sp::SpectrumFamily &fam : sp::builtin_families()) {
      const sc::ValidationCase c =
          sc::run_validation(2, 16, 0.50, sc::kDiffusionTime, 80, fam);
      const sc::ValidationCase c2 =
          sc::run_validation(2, 16, 0.50, sc::kDiffusionTime, 160, fam);
      std::cout << fam.id << " ratio=" << c.ratio << " half-dt ratio=" << c2.ratio
                << "\n";
      csv << c.family_id << "," << c.fd_order << "," << c.N << "," << c.f << ","
          << c.tau << "," << c.n_steps << "," << c.predicted_l2 << ","
          << c.measured_l2 << "," << c.ratio << "\n";
      csv << c2.family_id << "," << c2.fd_order << "," << c2.N << "," << c2.f << ","
          << c2.tau << "," << c2.n_steps << "," << c2.predicted_l2 << ","
          << c2.measured_l2 << "," << c2.ratio << "\n";
    }
    std::cout << "wrote " << data_dir << "/heat3d_spectral_family_validation.csv\n";
  }

  std::cout << "\nwrote " << data_dir << "/heat3d_spectral_content_map.csv\n"
            << "wrote " << data_dir << "/heat3d_spectral_content_crossover.csv\n";
  if (validate) {
    std::cout << "wrote " << data_dir << "/heat3d_spectral_content_validation.csv\n";
  }

  MPI_Finalize();
  return 0;
}
