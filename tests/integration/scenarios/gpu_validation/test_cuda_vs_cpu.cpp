// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/runtime/cuda/fft_cuda.hpp>
#endif

using namespace pfc;
using namespace pfc::test;

TEST_CASE("CUDA vs CPU diffusion consistency (smoke)", "[integration][gpu][cuda]") {
  auto world = world::uniform(16, 1.0);

  // CPU run
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

#if defined(OpenPFC_ENABLE_CUDA)
  auto decomp_gpu = decomposition::create(world, size);
  auto fft_gpu = fft::create_cuda(decomp_gpu, /*rank*/ 0, MPI_COMM_WORLD);
  REQUIRE(fft_gpu.size_inbox() == fft_cpu.size_inbox());
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
  SUCCEED("CUDA disabled - skipping GPU comparison");
#endif
}
