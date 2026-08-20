// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_manual.cpp
 * @brief Kobayashi phase-field + temperature coupling — manual FD matching the Julia
 *        `kobayashi_v1` script (Biner-style discretisation, explicit Euler).
 *
 * Periodic boundaries in x and y via `HaloExchange` on an nz=1 slab.
 * Fields and halo groups come from `pfc::sim::stacks::FDPaddedCPUStack`.
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>
#include <vector>

#include <kobayashi/cli.hpp>
#include <kobayashi/defaults.hpp>
#include <kobayashi/fd_stencils.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>
#include <openpfc/runtime/common/mpi_main.hpp>

#include <kobayashi/verification_utilities.hpp>

namespace {

using Field = pfc::data::Field<double, pfc::HostSpace>;

void run_kobayashi(const kobayashi::RunConfig &cfg, int rank, int nproc) {
  const double dx = cfg.dx;
  const double dy = dx;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_lap_den = 1.0 / (dx * dy);

  const auto domain = pfc::domain::create(pfc::GridSize({cfg.Nx, cfg.Ny, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, dy, 1.0}));

  constexpr int hw = 1;
  pfc::comm::HaloExchangeOptions state_opt;
  state_opt.directions = pfc::halo::presets::Axes2D();
  pfc::sim::stacks::FDPaddedCPUStack stack(domain, hw, rank, nproc, MPI_COMM_WORLD,
                                           state_opt);
  const auto &decomp = stack.decomposition();

  Field &phi = stack.u();
  Field tempr = stack.make_field();
  Field lap_phi = stack.make_field();
  Field lap_t = stack.make_field();
  Field phidx = stack.make_field();
  Field phidy = stack.make_field();
  Field epsilon = stack.make_field();
  Field epsilon_deriv = stack.make_field();

  const int Nx = cfg.Nx;
  const int Ny = cfg.Ny;
  const int ci = Nx / 2;
  const int cj = Ny / 2;

  phi.for_each_owned([&](int i, int j, int k) {
    (void)k;
    const auto g = phi.global(i, j, 0);
    const int gi = g[0];
    const int gj = g[1];
    const double ddx = static_cast<double>(gi - ci);
    const double ddy = static_cast<double>(gj - cj);
    phi(i, j, 0) = (ddx * ddx + ddy * ddy < kobayashi::kSeed) ? 1.0 : 0.0;
  });
  tempr.for_each_owned([&](int i, int j, int k) { tempr(i, j, k) = 0.0; });

  auto halo_state = stack.make_exchange({&phi, &tempr}, state_opt);
  pfc::comm::HaloExchangeOptions aux_opt;
  aux_opt.exchange_base = 2;
  aux_opt.directions = pfc::halo::presets::Axes2D();
  auto halo_aux =
      stack.make_exchange({&epsilon, &epsilon_deriv, &phidx, &phidy}, aux_opt);

  const bool skip_png = std::getenv("OPENPFC_KOBAYASHI_SKIP_PNG") != nullptr;
  const bool quiet = std::getenv("OPENPFC_KOBAYASHI_QUIET") != nullptr;
  const int nprint_eff = quiet ? 0 : cfg.nprint;

  if (rank == 0) {
    std::filesystem::create_directories(cfg.output_dir);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::cout
        << "KOBAYASHI_MPI_COMM_WORLD_SIZE=" << nproc
        << " (must match srun/mpirun task count; if not, tasks are not sharing one "
           "MPI_COMM_WORLD)\n";
  }

  int filenum = 0;
  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                  filenum);
    if (rank == 0) {
      std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    }
    write_phi_png(rank, decomp, phi, path);
    ++filenum;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop0 = MPI_Wtime();

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {
    halo_state.exchange();

    phi.for_each_owned([&](int i, int j, int k) {
      kobayashi::stage_a_cell(phi, tempr, lap_phi, lap_t, phidx, phidy, epsilon,
                              epsilon_deriv, i, j, k, inv_dx, inv_dy, inv_lap_den);
    });

    halo_aux.exchange();

    phi.for_each_owned([&](int i, int j, int k) {
      kobayashi::stage_b_cell(phi, tempr, lap_phi, lap_t, phidx, phidy, epsilon,
                              epsilon_deriv, i, j, k, inv_dx, inv_dy, cfg.dt);
    });

    if (nprint_eff > 0 && istep % nprint_eff == 0 && rank == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                    filenum);
      if (rank == 0) {
        std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file "
                  << path << "\n";
      }
      write_phi_png(rank, decomp, phi, path);
      ++filenum;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop1 = MPI_Wtime();
  const double wall_local = t_loop1 - t_loop0;
  double wall_max = 0.0;
  MPI_Reduce(&wall_local, &wall_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    if (rank == 0) {
      std::cout << "saving final field to " << path << "\n";
    }
    write_phi_png(rank, decomp, phi, path);
  }

  std::vector<double> loc_phi;
  std::vector<double> loc_T;
  pack_owned_xy0(phi, loc_phi);
  pack_owned_xy0(tempr, loc_T);

  std::vector<double> g_phi;
  std::vector<double> g_T;
  gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, loc_phi, Nx, Ny,
                         g_phi);
  gather_global_xy_rank0(decomp, rank, nproc, MPI_COMM_WORLD, loc_T, Nx, Ny, g_T);

  if (rank == 0) {
    const FieldStats sp = stats_global_ordered(g_phi, Nx, Ny);
    const FieldStats sT = stats_global_ordered(g_T, Nx, Ny);
    const double l2_phi = std::sqrt(sp.sumsq);
    const double l2_T = std::sqrt(sT.sumsq);
    std::cout << std::setprecision(17);
    std::cout << "KOBAYASHI_VERIFY"
              << " wall_loop_max_s=" << wall_max << " nproc=" << nproc
              << " Nx=" << Nx << " Ny=" << Ny << " steps=" << cfg.n_steps
              << " dt=" << cfg.dt << " dx=" << cfg.dx << " sum_phi=" << sp.sum
              << " sumsq_phi=" << sp.sumsq << " l2_phi=" << l2_phi
              << " min_phi=" << sp.min_v << " max_phi=" << sp.max_v
              << " sum_T=" << sT.sum << " sumsq_T=" << sT.sumsq << " l2_T=" << l2_T
              << " min_T=" << sT.min_v << " max_T=" << sT.max_v << "\n";
    std::cout << "KOBAYASHI_VERIFY_HEX"
              << " sum_phi=" << std::hexfloat << sp.sum << std::defaultfloat
              << " sumsq_phi=" << std::hexfloat << sp.sumsq << std::defaultfloat
              << " sum_T=" << std::hexfloat << sT.sum << std::defaultfloat
              << " sumsq_T=" << std::hexfloat << sT.sumsq << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  return pfc::runtime::mpi_main(
      argc, argv, [](int app_argc, char **app_argv, int rank, int nproc) {
        const auto cfg = kobayashi::parse_or_print_usage(app_argc, app_argv, rank);
        if (!cfg) {
          return EXIT_FAILURE;
        }
        run_kobayashi(*cfg, rank, nproc);
        return EXIT_SUCCESS;
      });
}
