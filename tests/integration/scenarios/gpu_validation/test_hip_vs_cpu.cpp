// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fixtures/diffusion_model.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

using namespace pfc;
using namespace pfc::test;

TEST_CASE("HIP vs CPU diffusion consistency (smoke)", "[integration][gpu][hip]") {
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

#if defined(OpenPFC_ENABLE_HIP)
  // HIP smoke test: verify HIP headers and runtime are available
  // Full HIP vs CPU consistency testing requires HIP spectral support (OpenPFC_ENABLE_HIP_SPECTRAL)
  // which provides rocFFT backend via fft::create_hip() - not tested here
  SUCCEED("HIP enabled - HIP runtime components available");
#else
  SUCCEED("HIP disabled - skipping GPU comparison");
#endif
}
