// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file kobayashi_fd_openmp_engine.cpp
 * @brief Single-rank `FDPaddedCPUStack` Kobayashi kernel with OpenMP stages.
 */

#include <kobayashi/defaults.hpp>
#include <kobayashi/fd_stencils.hpp>
#include <kobayashi/openmp_engine.hpp>

#include <kobayashi/verification_utilities.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/simulation/stacks/fd_padded_cpu_stack.hpp>

#include <cstdio>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <omp.h>

namespace kobayashi::openmp_engine {

namespace {

void ensure_mpi() {
  int ready = 0;
  MPI_Initialized(&ready);
  if (ready == 0) {
    MPI_Init(nullptr, nullptr);
  }
}

} // namespace

RunResult run(const RunConfigOpenMP &cfg, bool skip_png, bool quiet) {
  ensure_mpi();

  const double dx = cfg.dx;
  const double dy = dx;
  const double inv_dx = 1.0 / dx;
  const double inv_dy = 1.0 / dy;
  const double inv_lap_den = 1.0 / (dx * dy);

  if (cfg.num_threads > 0) {
    omp_set_num_threads(cfg.num_threads);
  }
  const int nthr = omp_get_max_threads();
  const bool use_team = (nthr > 1);

  const auto domain = pfc::domain::create(pfc::GridSize({cfg.Nx, cfg.Ny, 1}),
                                          pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                          pfc::GridSpacing({dx, dy, 1.0}));
  constexpr int hw = 1;
  pfc::comm::HaloExchangeOptions state_opt;
  state_opt.directions = pfc::halo::presets::Axes2D();
  pfc::sim::stacks::FDPaddedCPUStack stack(domain, hw, /*rank=*/0, /*nproc=*/1,
                                           MPI_COMM_WORLD, state_opt);
  const auto &decomp = stack.decomposition();

  using Field = pfc::data::Field<double, pfc::HostSpace>;
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
  const int nx = phi.local_size()[0];
  const int ny = phi.local_size()[1];
  const int ci = Nx / 2;
  const int cj = Ny / 2;

#pragma omp parallel for collapse(2) schedule(static) if (use_team)
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const auto g = phi.global(i, j, 0);
      const double ddx = static_cast<double>(g[0] - ci);
      const double ddy = static_cast<double>(g[1] - cj);
      phi(i, j, 0) = (ddx * ddx + ddy * ddy < kobayashi::kSeed) ? 1.0 : 0.0;
      tempr(i, j, 0) = 0.0;
    }
  }

  auto halo_state = stack.make_exchange({&phi, &tempr}, state_opt);
  pfc::comm::HaloExchangeOptions aux_opt;
  aux_opt.exchange_base = 2;
  aux_opt.directions = pfc::halo::presets::Axes2D();
  auto halo_aux =
      stack.make_exchange({&epsilon, &epsilon_deriv, &phidx, &phidy}, aux_opt);

  int filenum = 0;
  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                  filenum);
    std::cout << "saving step 0/" << cfg.n_steps << " to file " << path << "\n";
    write_phi_png(0, decomp, phi, path);
    ++filenum;
  }

  const int nprint_eff = quiet ? 0 : cfg.nprint;
  const double t_loop0 = omp_get_wtime();

  for (int istep = 1; istep <= cfg.n_steps; ++istep) {
    halo_state.exchange();

#pragma omp parallel for collapse(2) schedule(static) if (use_team)
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        kobayashi::stage_a_cell(phi, tempr, lap_phi, lap_t, phidx, phidy, epsilon,
                                epsilon_deriv, i, j, 0, inv_dx, inv_dy, inv_lap_den);
      }
    }

    halo_aux.exchange();

#pragma omp parallel for collapse(2) schedule(static) if (use_team)
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        kobayashi::stage_b_cell(phi, tempr, lap_phi, lap_t, phidx, phidy, epsilon,
                                epsilon_deriv, i, j, 0, inv_dx, inv_dy, cfg.dt);
      }
    }

    if (nprint_eff > 0 && istep % nprint_eff == 0) {
      std::cout << "step " << istep << "/" << cfg.n_steps << " done\n";
    }

    if (!skip_png && cfg.nsave > 0 && istep % cfg.nsave == 0) {
      char path[4096];
      std::snprintf(path, sizeof(path), "%s/phi_%04d.png", cfg.output_dir.c_str(),
                    filenum);
      std::cout << "saving step " << istep << "/" << cfg.n_steps << " to file "
                << path << "\n";
      write_phi_png(0, decomp, phi, path);
      ++filenum;
    }
  }

  const double t_loop1 = omp_get_wtime();

  if (!skip_png) {
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/phi_final.png", cfg.output_dir.c_str());
    std::cout << "saving final field to " << path << "\n";
    write_phi_png(0, decomp, phi, path);
  }

  RunResult out;
  pack_owned_xy0(phi, out.phi_xy);
  pack_owned_xy0(tempr, out.tempr_xy);
  out.wall_loop_s = t_loop1 - t_loop0;
  out.nthreads = nthr;
  return out;
}

} // namespace kobayashi::openmp_engine
