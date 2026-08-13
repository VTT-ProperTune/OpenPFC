// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/kspace.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/hip/fft_hip.hpp>
#endif

using namespace pfc;
using namespace pfc::fft::kspace;

static inline World make_world(int nx, int ny, int nz) {
  return domain::create_world(GridSize({nx, ny, nz}), PhysicalOrigin({0.0, 0.0, 0.0}),
                       GridSpacing({1.0, 1.0, 1.0}));
}

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
TEST_CASE("CPU vs HIP Laplacian equivalence (multi-rank) [integration][gpu][hip][mpi]",
          "[gpu][hip][mpi]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = make_world(32, 24, 20);
  auto decomp = decomposition::create(world, size);

  auto fft_cpu = fft::create(decomp, rank, MPI_COMM_WORLD);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto fft_gpu = fft::create_hip(decomp, rank, MPI_COMM_WORLD);

  const auto n_in = fft_cpu.size_inbox();
  const auto n_out = fft_cpu.size_outbox();

  const auto inbox = fft::get_inbox(fft_cpu);
  const auto [Lx, Ly, Lz] = world::get_size(world);

  std::vector<double> real_in_cpu(n_in);
  {
    size_t idx = 0;
    for (int k = inbox.low[2]; k <= inbox.high[2]; ++k) {
      for (int j = inbox.low[1]; j <= inbox.high[1]; ++j) {
        for (int i = inbox.low[0]; i <= inbox.high[0]; ++i) {
          double val =
              0.3 +
              0.2 * std::sin(2.0 * std::numbers::pi * i / static_cast<double>(Lx)) +
              0.15 * std::sin(2.0 * std::numbers::pi * j / static_cast<double>(Ly)) +
              0.1 * std::sin(2.0 * std::numbers::pi * k / static_cast<double>(Lz));
          real_in_cpu[idx++] = val;
        }
      }
    }
  }

  std::vector<std::complex<double>> freq_cpu(n_out);
  fft_cpu.forward(real_in_cpu, freq_cpu);

  const auto outbox = fft::get_outbox(fft_cpu);
  const auto [fx, fy, fz] = k_frequency_scaling(world);
  {
    size_t idx = 0;
    for (int k = outbox.low[2]; k <= outbox.high[2]; ++k) {
      for (int j = outbox.low[1]; j <= outbox.high[1]; ++j) {
        for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
          const double ki = k_component(i, Lx, fx);
          const double kj = k_component(j, Ly, fy);
          const double kk = k_component(k, Lz, fz);
          const double kLap = k_laplacian_value(ki, kj, kk);
          freq_cpu[idx] *= kLap;
          ++idx;
        }
      }
    }
  }

  std::vector<double> real_out_cpu(n_in);
  fft_cpu.backward(freq_cpu, real_out_cpu);

  core::DataBuffer<backend::HipTag, double> real_in_gpu(n_in);
  core::DataBuffer<backend::HipTag, double> real_out_gpu(n_in);
  core::DataBuffer<backend::HipTag, std::complex<double>> freq_gpu(n_out);

  real_in_gpu.copy_from_host(real_in_cpu);
  fft_gpu.forward(real_in_gpu, freq_gpu);

  auto freq_gpu_host = freq_gpu.to_host();
  {
    size_t idx = 0;
    for (int k = outbox.low[2]; k <= outbox.high[2]; ++k) {
      for (int j = outbox.low[1]; j <= outbox.high[1]; ++j) {
        for (int i = outbox.low[0]; i <= outbox.high[0]; ++i) {
          const double ki = k_component(i, Lx, fx);
          const double kj = k_component(j, Ly, fy);
          const double kk = k_component(k, Lz, fz);
          const double kLap = k_laplacian_value(ki, kj, kk);
          freq_gpu_host[idx] *= kLap;
          ++idx;
        }
      }
    }
  }
  freq_gpu.copy_from_host(freq_gpu_host);
  fft_gpu.backward(freq_gpu, real_out_gpu);
  auto real_out_gpu_host = real_out_gpu.to_host();

  bool local_fields_match = real_out_gpu_host.size() == real_out_cpu.size();
  for (size_t i = 0; i < real_out_cpu.size(); ++i)
    local_fields_match &= std::abs(real_out_gpu_host[i] - real_out_cpu[i]) <= 1e-10;
  REQUIRE(local_fields_match);

  auto local_l2_sq = [&]() {
    long double s = 0.0L;
    for (double x : real_out_cpu)
      s += static_cast<long double>(x) * x;
    return s;
  }();
  double local_cpu_l2 = static_cast<double>(local_l2_sq);
  double global_cpu_l2 = 0.0;
  MPI_Allreduce(&local_cpu_l2, &global_cpu_l2, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  long double s2 = 0.0L;
  for (double x : real_out_gpu_host)
    s2 += static_cast<long double>(x) * x;
  double local_gpu_l2 = static_cast<double>(s2);
  double global_gpu_l2 = 0.0;
  MPI_Allreduce(&local_gpu_l2, &global_gpu_l2, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  REQUIRE(std::sqrt(global_gpu_l2) ==
          Catch::Approx(std::sqrt(global_cpu_l2)).epsilon(1e-10));
}
#else
TEST_CASE("CPU vs HIP Laplacian (multi-rank) skipped [integration][gpu][hip][mpi]",
          "[gpu][hip][mpi]") {
  SUCCEED("HIP spectral not enabled; skipping multi-rank CPU-vs-HIP Laplacian test");
}
#endif
