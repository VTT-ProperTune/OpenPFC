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
TEST_CASE("CPU vs HIP Laplacian equivalence (double) [integration][gpu][hip]",
          "[gpu][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = make_world(24, 20, 16);
  auto decomp = decomposition::create(world, size);

  auto fft_cpu = fft::create(decomp, rank, MPI_COMM_WORLD);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto fft_gpu = fft::create_hip(decomp, rank, MPI_COMM_WORLD);

  const auto n_in = fft_cpu.size_inbox();
  const auto n_out = fft_cpu.size_outbox();

  std::vector<double> real_in_cpu(n_in);
  for (size_t i = 0; i < real_in_cpu.size(); ++i) {
    real_in_cpu[i] =
        0.3 + 0.4 * std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                             static_cast<double>(real_in_cpu.size()));
  }

  std::vector<std::complex<double>> freq_cpu(n_out);
  fft_cpu.forward(real_in_cpu, freq_cpu);

  const auto [Lx, Ly, Lz] = world::get_size(world);
  const auto outbox = fft::get_outbox(fft_cpu);
  auto low = outbox.low;
  auto high = outbox.high;
  const auto [fx, fy, fz] = k_frequency_scaling(world);

  {
    size_t idx = 0;
    for (int k = low[2]; k <= high[2]; ++k) {
      for (int j = low[1]; j <= high[1]; ++j) {
        for (int i = low[0]; i <= high[0]; ++i) {
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

  core::DataBuffer<backend::HIPTag, double> real_in_gpu(n_in);
  core::DataBuffer<backend::HIPTag, double> real_out_gpu(n_in);
  core::DataBuffer<backend::HIPTag, std::complex<double>> freq_gpu(n_out);

  real_in_gpu.copy_from_host(real_in_cpu);
  fft_gpu.forward(real_in_gpu, freq_gpu);

  auto freq_gpu_host = freq_gpu.to_host();
  {
    size_t idx = 0;
    for (int k = low[2]; k <= high[2]; ++k) {
      for (int j = low[1]; j <= high[1]; ++j) {
        for (int i = low[0]; i <= high[0]; ++i) {
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

  REQUIRE(real_out_gpu_host.size() == real_out_cpu.size());

  auto l2 = [](const std::vector<double> &v) {
    long double s = 0.0L;
    for (double x : v)
      s += static_cast<long double>(x) * x;
    return std::sqrt(static_cast<double>(s));
  };

  const double l2_cpu = l2(real_out_cpu);
  const double l2_gpu = l2(real_out_gpu_host);
  REQUIRE(l2_gpu == Catch::Approx(l2_cpu).epsilon(1e-10));

  bool fields_match = true;
  for (size_t i = 0; i < real_out_cpu.size(); ++i)
    fields_match &= std::abs(real_out_gpu_host[i] - real_out_cpu[i]) <= 1e-10;
  REQUIRE(fields_match);
}
#else
TEST_CASE("CPU vs HIP Laplacian skipped (HIP spectral disabled) [integration][gpu][hip]",
          "[gpu][hip]") {
  SUCCEED("HIP spectral not enabled; skipping CPU-vs-HIP Laplacian test");
}
#endif
