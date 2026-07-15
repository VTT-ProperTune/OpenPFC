// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_openmp.cpp
 * @brief CLI driver: Kobayashi FD on one node with OpenMP (periodic wrap, no MPI halos).
 */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <kobayashi/cli.hpp>
#include <kobayashi/openmp_engine.hpp>


#include <kobayashi/verification_utilities.hpp>
namespace {

} // namespace

int main(int argc, char **argv) {
  const auto cfg = kobayashi::parse_or_print_usage_openmp(argc, argv);
  if (!cfg) {
    return EXIT_FAILURE;
  }

  const bool skip_png = std::getenv("OPENPFC_KOBAYASHI_SKIP_PNG") != nullptr;
  const bool quiet = std::getenv("OPENPFC_KOBAYASHI_QUIET") != nullptr;

  std::filesystem::create_directories(cfg->output_dir);

  auto res = kobayashi::openmp_engine::run(*cfg, skip_png, quiet);

  std::cout << "KOBAYASHI_OPENMP_THREADS=" << res.nthreads
            << " (effective omp_get_max_threads for integration loop)\n";

  const int Nx = cfg->Nx;
  const int Ny = cfg->Ny;
  const FieldStats sp = stats_global_ordered(res.phi_xy, Nx, Ny);
  const FieldStats sT = stats_global_ordered(res.tempr_xy, Nx, Ny);
  const double l2_phi = std::sqrt(sp.sumsq);
  const double l2_T = std::sqrt(sT.sumsq);

  std::cout << std::setprecision(17);
  std::cout << "KOBAYASHI_VERIFY"
            << " wall_loop_max_s=" << res.wall_loop_s << " nthreads=" << res.nthreads
            << " Nx=" << Nx << " Ny=" << Ny << " steps=" << cfg->n_steps << " dt=" << cfg->dt
            << " dx=" << cfg->dx << " sum_phi=" << sp.sum << " sumsq_phi=" << sp.sumsq
            << " l2_phi=" << l2_phi << " min_phi=" << sp.min_v << " max_phi=" << sp.max_v
            << " sum_T=" << sT.sum << " sumsq_T=" << sT.sumsq << " l2_T=" << l2_T
            << " min_T=" << sT.min_v << " max_T=" << sT.max_v << "\n";
  std::cout << "KOBAYASHI_VERIFY_HEX"
            << " sum_phi=" << std::hexfloat << sp.sum << std::defaultfloat
            << " sumsq_phi=" << std::hexfloat << sp.sumsq << std::defaultfloat
            << " sum_T=" << std::hexfloat << sT.sum << std::defaultfloat
            << " sumsq_T=" << std::hexfloat << sT.sumsq << "\n";

  return EXIT_SUCCESS;
}
