// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error                                                                              \
    "test_allen_cahn_cpu_vs_cuda requires CUDA (configure with -DOpenPFC_ENABLE_CUDA=ON)"
#endif

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <allen_cahn/common.hpp>
#include <allen_cahn/device_step.hpp>
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

TEST_CASE("Allen–Cahn CPU vs CUDA agreement (single rank)", "[AllenCahn][CUDA]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  allen_cahn::RunConfig cfg;
  cfg.nx_glob = 32;
  cfg.ny_glob = 32;
  cfg.n_steps = 20;
  cfg.dt = 0.002;
  cfg.M = 1.0;
  cfg.epsilon = 0.5;
  cfg.driving_force = 0.25;

  auto world = pfc::world::create(pfc::GridSize({cfg.nx_glob, cfg.ny_glob, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, 1);

  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  auto local_size = pfc::world::get_size(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  REQUIRE(nz == 1);
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  const double dx = 1.0;
  const double inv_dx2 = 1.0 / (dx * dx);
  const double inv_dy2 = inv_dx2;
  const double inv_eps2 = 1.0 / (cfg.epsilon * cfg.epsilon);

  std::vector<double> u0(nlocal);
  std::vector<double> u_cpu(nlocal);
  std::vector<double> u_gpu_host(nlocal);
  std::vector<double> lap(nlocal);
  allen_cahn::fill_initial_condition(&u0, decomp, rank);
  u_cpu = u0;

  constexpr int halo_width = allen_cahn::RunConfig::kHaloWidth;
  auto face_cpu = pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::SeparatedFaceHaloExchanger<double> exch_cpu(decomp, rank, halo_width,
                                                   MPI_COMM_WORLD);

  for (int step = 0; step < cfg.n_steps; ++step) {
    allen_cahn::step_explicit_euler_cpu(&u_cpu, &lap, &face_cpu, &exch_cpu, nx, ny,
                                        nz, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                        inv_eps2, cfg.driving_force);
  }

  u_gpu_host = u0;

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
  cuda_check(cudaMemcpy(u_dev, u_gpu_host.data(), nlocal * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");

  for (int step = 0; step < cfg.n_steps; ++step) {
    cuda_check(cudaMemcpy(u_gpu_host.data(), u_dev, nlocal * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H core");
    exchanger.exchange_halos(u_gpu_host.data(), u_gpu_host.size(), face_halos_host);
    for (int f = 0; f < 6; ++f) {
      const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
      if (n == 0) {
        continue;
      }
      cuda_check(cudaMemcpy(face_dev[static_cast<std::size_t>(f)],
                            face_halos_host[static_cast<std::size_t>(f)].data(),
                            n * sizeof(double), cudaMemcpyHostToDevice),
                 "cudaMemcpy face");
    }
    allen_cahn::allen_cahn_step_cuda(u_dev, face_dev[0], face_dev[1], face_dev[2],
                                     face_dev[3], face_dev[4], face_dev[5], nx, ny,
                                     nz, halo_width, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                     inv_eps2, cfg.driving_force);
  }

  cuda_check(cudaMemcpy(u_gpu_host.data(), u_dev, nlocal * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy final");

  cudaFree(u_dev);
  for (int f = 0; f < 6; ++f) {
    cudaFree(face_dev[static_cast<std::size_t>(f)]);
  }

  double max_diff = 0.0;
  for (std::size_t i = 0; i < nlocal; ++i) {
    max_diff = std::max(max_diff, std::abs(u_cpu[i] - u_gpu_host[i]));
  }
  REQUIRE(max_diff < 1.0e-9);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  const int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
