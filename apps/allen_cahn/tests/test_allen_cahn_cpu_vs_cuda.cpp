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
#include <cmath>
#include <mpi.h>
#include <vector>

#include <allen_cahn/common.hpp>
#include <allen_cahn/device_step.hpp>
#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>
#include <openpfc/runtime/gpu/comm_sparse_exchange_gpu.hpp>

namespace {

using DevField = pfc::data::Field<double, pfc::CUDASpace>;

} // namespace

TEST_CASE("Allen–Cahn CPU vs CUDA agreement (single rank)", "[AllenCahn][CUDA]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
    SKIP("No CUDA device");
  }
  REQUIRE(pfc::gpu::runtime_mpi_gpu_aware());

  allen_cahn::RunConfig cfg;
  cfg.nx_glob = 32;
  cfg.ny_glob = 32;
  cfg.n_steps = 20;
  cfg.dt = 0.002;
  cfg.M = 1.0;
  cfg.epsilon = 0.5;
  cfg.driving_force = 0.25;

  auto domain = pfc::domain::create(pfc::GridSize({cfg.nx_glob, cfg.ny_glob, 1}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, 1);

  const auto &local_box = pfc::decomposition::local_box(decomp, rank);
  auto local_size = local_box.size;
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
  pfc::comm::SparseExchange<pfc::HostSpace, double> exch_cpu(
      u_cpu.data(), u_cpu.size(), decomp, rank, MPI_COMM_WORLD, halo_width);

  for (int step = 0; step < cfg.n_steps; ++step) {
    allen_cahn::step_explicit_euler_cpu(&u_cpu, &lap, &face_cpu, &exch_cpu, nx, ny,
                                        nz, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                        inv_eps2, cfg.driving_force);
  }

  DevField u_gpu(domain, local_box, /*storage_halo=*/0, /*iteration_halo=*/halo_width);
  u_gpu.with_host_view([&](double *data, std::size_t n) {
    REQUIRE(n == u0.size());
    std::copy(u0.begin(), u0.end(), data);
  });
  u_gpu.sync_to_device();
  pfc::comm::SparseExchange<pfc::CUDASpace, double> exchanger(
      u_gpu, decomp, rank, MPI_COMM_WORLD);

  for (int step = 0; step < cfg.n_steps; ++step) {
    exchanger.exchange();
    const auto faces = exchanger.face_recv_ptrs();
    allen_cahn::allen_cahn_step_cuda(u_gpu.data(), faces[0], faces[1], faces[2],
                                     faces[3], faces[4], faces[5], nx, ny, nz,
                                     halo_width, inv_dx2, inv_dy2, cfg.dt, cfg.M,
                                     inv_eps2, cfg.driving_force);
    u_gpu.note_device_write();
  }

  u_gpu.with_host_view([&](double *data, std::size_t n) {
    u_gpu_host.assign(data, data + n);
  });

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
