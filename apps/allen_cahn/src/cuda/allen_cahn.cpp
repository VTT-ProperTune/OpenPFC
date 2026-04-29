// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error "allen_cahn_cuda requires CUDA (configure with -DOpenPFC_ENABLE_CUDA=ON)"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <allen_cahn/common.hpp>
#include <allen_cahn/device_step.hpp>
#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

namespace {

void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

} // namespace

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

  std::vector<double> u_host(nlocal);
  allen_cahn::fill_initial_condition(&u_host, decomp, rank);

  constexpr int halo_width = allen_cahn::RunConfig::kHaloWidth;
  auto face_halos_host =
      pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                                    MPI_COMM_WORLD);

  const auto counts = pfc::halo::face_halo_counts(decomp, rank, halo_width);

  double *u_dev = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&u_dev), nlocal * sizeof(double)),
             "cudaMalloc core");
  std::array<double *, 6> face_dev{};
  for (int f = 0; f < 6; ++f) {
    const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
    cuda_check(
        cudaMalloc(reinterpret_cast<void **>(&face_dev[static_cast<std::size_t>(f)]),
                   n * sizeof(double)),
        "cudaMalloc face");
  }

  cuda_check(cudaMemcpy(u_dev, u_host.data(), nlocal * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H2D core");

  for (int step = 0; step < cfg.n_steps; ++step) {
    cuda_check(cudaMemcpy(u_host.data(), u_dev, nlocal * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H core");
    exchanger.exchange_halos(u_host.data(), u_host.size(), face_halos_host);
    for (int f = 0; f < 6; ++f) {
      const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
      if (n == 0) {
        continue;
      }
      cuda_check(cudaMemcpy(face_dev[static_cast<std::size_t>(f)],
                            face_halos_host[static_cast<std::size_t>(f)].data(),
                            n * sizeof(double), cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D face");
    }
    allen_cahn::allen_cahn_step_cuda(u_dev, face_dev[0], face_dev[1], face_dev[2],
                                     face_dev[3], face_dev[4], face_dev[5], nx, ny,
                                     nz, halo_width, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                     inv_eps2);
  }

  cuda_check(cudaMemcpy(u_host.data(), u_dev, nlocal * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy D2H final");

  if (!cfg.png_output.empty()) {
    pfc::io::write_mpi_scalar_field_png_xy(MPI_COMM_WORLD, decomp, rank, u_host,
                                           cfg.png_output);
    if (rank == 0) {
      std::cout << "Wrote PNG: " << cfg.png_output << "\n";
    }
  }

  cudaFree(u_dev);
  for (int f = 0; f < 6; ++f) {
    cudaFree(face_dev[static_cast<std::size_t>(f)]);
  }

  double sum_u = 0.0;
  for (double v : u_host) {
    sum_u += v;
  }
  double sum_global = 0.0;
  MPI_Reduce(&sum_u, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Allen–Cahn FD (CUDA): grid " << cfg.nx_glob << "x" << cfg.ny_glob
              << "x1, ranks=" << nproc << ", steps=" << cfg.n_steps << "\n";
    std::cout << "Global sum(phi) after stepping: " << sum_global << "\n";
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
