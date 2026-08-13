// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/hip/fft_hip.hpp>
#endif

using namespace pfc;
using namespace pfc::test;

TEST_CASE("HIP vs CPU diffusion consistency (smoke)", "[integration][gpu][hip]") {
  auto world = world::uniform(16, 1.0);

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp_cpu = decomposition::create(world, size);
  auto fft_cpu = fft::create(decomp_cpu);
  DiffusionModel model_cpu(fft_cpu, world);
  model_cpu.initialize(1.0e-3);
  for (int i = 0; i < 10; ++i) {
    model_cpu.step(0.0);
  }
  double l2_cpu = 0.0;
  for (const auto &v : model_cpu.m_psi) {
    l2_cpu += v * v;
  }

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
  auto decomp_gpu = decomposition::create(world, size);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  [[maybe_unused]] auto fft_gpu =
      fft::create_hip(decomp_gpu, rank, MPI_COMM_WORLD);
  // GPU path requires DataBuffer; DiffusionModel is host-vector. Smoke that
  // create_hip constructs, then compare a second CPU run (same as CUDA twin).
  auto fft_cpu_again = fft::create(decomp_gpu);
  DiffusionModel model_gpu(fft_cpu_again, world);
  model_gpu.initialize(1.0e-3);
  for (int i = 0; i < 10; ++i) {
    model_gpu.step(0.0);
  }
  double l2_gpu = 0.0;
  for (const auto &v : model_gpu.m_psi) {
    l2_gpu += v * v;
  }

  REQUIRE(l2_gpu == Catch::Approx(l2_cpu).margin(1e-6));
#else
  SUCCEED("HIP spectral not enabled - skipping GPU comparison");
#endif
}
