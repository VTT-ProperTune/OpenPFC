// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <vector>

#include <allen_cahn/common.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const allen_cahn::RunConfig cfg = allen_cahn::parse_args(argc, argv);
  if (cfg.nx_glob < 4 || cfg.ny_glob < 4 || cfg.n_steps < 1) {
    if (rank == 0) {
      std::cerr << "Need nx, ny >= 4 and n_steps >= 1\n";
    }
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  auto world = pfc::world::create(pfc::GridSize({cfg.nx_glob, cfg.ny_glob, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, nproc);

  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  auto local_size = pfc::world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  const double dx = 1.0;
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = inv_dx2;
  const double inv_eps2 = 1.0 / (cfg.epsilon * cfg.epsilon);

  std::vector<double> u(nlocal);
  std::vector<double> lap(nlocal);
  allen_cahn::fill_initial_condition(&u, decomp, rank);

  constexpr int halo_width = allen_cahn::RunConfig::kHaloWidth;
  auto face_halos = pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                                    MPI_COMM_WORLD);

  for (int step = 0; step < cfg.n_steps; ++step) {
    allen_cahn::step_explicit_euler_cpu(&u, &lap, &face_halos, &exchanger, nx, ny,
                                        nz, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                        inv_eps2);
  }

  double sum_u = 0.0;
  for (double v : u) {
    sum_u += v;
  }
  double sum_global = 0.0;
  MPI_Reduce(&sum_u, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Allen–Cahn FD (CPU): grid " << cfg.nx_glob << "x" << cfg.ny_glob
              << "x1, ranks=" << nproc << ", steps=" << cfg.n_steps << "\n";
    std::cout << "Global sum(phi) after stepping: " << sum_global << "\n";
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
