// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#if !defined(OpenPFC_ENABLE_CUDA)
#error "test_wave2d_cpu_vs_cuda requires CUDA"
#endif

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_face_layout.hpp>

#include <wave2d/device_step.hpp>
#include <wave2d/wave_model.hpp>
#include <wave2d/wave_step_separated.hpp>

namespace {

void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

} // namespace

TEST_CASE("wave2d CPU vs CUDA (Neumann y, single rank)", "[wave2d][CUDA]") {
  int rank = 0;
  int nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REQUIRE(nproc == 1);

  constexpr int Nx = 24;
  constexpr int Ny = 24;
  constexpr int n_steps = 8;
  const double dt = 0.01;

  auto world = pfc::world::create(pfc::GridSize({Nx, Ny, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, 1);

  const auto &local_world = pfc::decomposition::get_subworld(decomp, rank);
  auto local_size = pfc::world::get_size(local_world);
  const auto lower = pfc::world::get_lower(local_world);
  const int nx = local_size[0];
  const int ny = local_size[1];
  const int nz = local_size[2];
  const std::size_t nlocal = static_cast<std::size_t>(nx) *
                             static_cast<std::size_t>(ny) *
                             static_cast<std::size_t>(nz);

  const double inv_dx2 = 1.0;
  const double inv_dy2 = 1.0;
  constexpr int halo_width = 1;

  std::vector<double> u0(nlocal);
  std::vector<double> v0(nlocal, 0.0);
  const double xc = 0.5 * static_cast<double>(Nx - 1);
  const double yc = 0.5 * static_cast<double>(Ny - 1);
  const double sigma = 3.0;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int gx = lower[0] + ix;
        const int gy = lower[1] + iy;
        const double x = static_cast<double>(gx);
        const double y = static_cast<double>(gy);
        const double dx = x - xc;
        const double dy = y - yc;
        const std::size_t idx =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
            static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny);
        u0[idx] = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
      }
    }
  }

  std::vector<double> u_cpu = u0;
  std::vector<double> v_cpu = v0;
  std::vector<double> lap_cpu(nlocal);
  auto face_cpu = pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::SeparatedFaceHaloExchanger<double> exch_cpu(decomp, rank, halo_width,
                                                   MPI_COMM_WORLD);
  for (int s = 0; s < n_steps; ++s) {
    (void)s;
    wave2d::step_wave_separated_order2_cpu(u_cpu, v_cpu, lap_cpu, face_cpu, exch_cpu,
                                           nx, ny, nz, decomp, rank, dt,
                                           wave2d::YBoundaryKind::Neumann, Ny, 0.0);
  }

  std::vector<double> u_gpu_host = u0;
  std::vector<double> v_gpu_host = v0;
  auto face_halos_host =
      pfc::halo::allocate_face_halos<double>(decomp, rank, halo_width);
  pfc::SeparatedFaceHaloExchanger<double> exchanger(decomp, rank, halo_width,
                                                    MPI_COMM_WORLD);
  const auto counts = pfc::halo::face_halo_counts(decomp, rank, halo_width);

  double *u_dev = nullptr;
  double *v_dev = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&u_dev), nlocal * sizeof(double)),
             "u");
  cuda_check(cudaMalloc(reinterpret_cast<void **>(&v_dev), nlocal * sizeof(double)),
             "v");
  std::array<double *, 6> face_dev{};
  for (int f = 0; f < 6; ++f) {
    const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
    cuda_check(
        cudaMalloc(reinterpret_cast<void **>(&face_dev[static_cast<std::size_t>(f)]),
                   std::max<std::size_t>(n, 1u) * sizeof(double)),
        "face");
  }
  cuda_check(cudaMemcpy(u_dev, u_gpu_host.data(), nlocal * sizeof(double),
                        cudaMemcpyHostToDevice),
             "H2D u");
  cuda_check(cudaMemcpy(v_dev, v_gpu_host.data(), nlocal * sizeof(double),
                        cudaMemcpyHostToDevice),
             "H2D v");

  for (int step = 0; step < n_steps; ++step) {
    (void)step;
    cuda_check(cudaMemcpy(u_gpu_host.data(), u_dev, nlocal * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H u");
    cuda_check(cudaMemcpy(v_gpu_host.data(), v_dev, nlocal * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "D2H v");
    exchanger.exchange_halos(u_gpu_host.data(), u_gpu_host.size(), face_halos_host);
    wave2d::patch_y_face_halos_neumann_order2(u_gpu_host.data(), nx, ny,
                                              face_halos_host, lower, Ny);
    for (int f = 0; f < 6; ++f) {
      const std::size_t n = counts.counts[static_cast<std::size_t>(f)];
      if (n == 0) {
        continue;
      }
      cuda_check(cudaMemcpy(face_dev[static_cast<std::size_t>(f)],
                            face_halos_host[static_cast<std::size_t>(f)].data(),
                            n * sizeof(double), cudaMemcpyHostToDevice),
                 "H2D face");
    }
    wave2d::wave2d_step_cuda(u_dev, v_dev, face_dev[0], face_dev[1], face_dev[2],
                             face_dev[3], face_dev[4], face_dev[5], nx, ny, nz,
                             halo_width, inv_dx2, inv_dy2, dt, wave2d::kC);
  }

  cuda_check(cudaMemcpy(u_gpu_host.data(), u_dev, nlocal * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "final u");
  cuda_check(cudaMemcpy(v_gpu_host.data(), v_dev, nlocal * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "final v");
  cudaFree(u_dev);
  cudaFree(v_dev);
  for (int f = 0; f < 6; ++f) {
    cudaFree(face_dev[static_cast<std::size_t>(f)]);
  }

  double max_diff = 0.0;
  for (std::size_t i = 0; i < nlocal; ++i) {
    max_diff = std::max(max_diff, std::abs(u_cpu[i] - u_gpu_host[i]));
    max_diff = std::max(max_diff, std::abs(v_cpu[i] - v_gpu_host[i]));
  }
  REQUIRE(max_diff < 1e-9);
}

int main(int argc, char *argv[]) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    std::cerr << "MPI_Init failed\n";
    return 1;
  }
  const int r = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return r;
}
