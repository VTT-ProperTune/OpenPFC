// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_cpu_vs_hip.cpp
 * @brief Test to compare CPU and HIP implementations of Tungsten model
 *
 * This test verifies that the CPU and HIP implementations produce
 * the same results (within floating-point precision tolerance).
 */

#define CATCH_CONFIG_RUNNER
#if !defined(OpenPFC_ENABLE_HIP)
#error "This test requires HIP support. Enable with -DOpenPFC_ENABLE_HIP=ON"
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/fft/fft.hpp>
#include <openpfc/runtime/hip/fft_hip.hpp>
#include <tungsten/cpu/tungsten_model.hpp>
#include <tungsten/hip/tungsten_model.hpp>
#include <vector>

using namespace pfc;
using namespace Catch::Matchers;

TEST_CASE("Tungsten CPU vs HIP: Same results", "[Tungsten][CPU][HIP]") {
  auto world = world::create(GridSize({32, 32, 32}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft_cpu = fft::create(decomp, rank);

  Tungsten model_cpu(fft_cpu, world);
  TungstenHIP<double> model_hip(fft_cpu, world);

  model_cpu.params.set_n0(-0.4);
  model_cpu.params.set_T(0.5);
  model_hip.params.set_n0(-0.4);
  model_hip.params.set_T(0.5);

  double dt = 0.01;
  model_cpu.initialize(dt);
  model_hip.initialize(dt);

  auto &psi_cpu = model_cpu.get_real_field("psi");
  auto &psi_hip_gpu = model_hip.get_psi();

  for (size_t i = 0; i < psi_cpu.size(); ++i) {
    psi_cpu[i] = -0.4 + 0.1 * std::sin(2.0 * M_PI * i / psi_cpu.size());
  }

  std::vector<double> psi_init(psi_cpu.begin(), psi_cpu.end());
  psi_hip_gpu.copy_from_host(psi_init);

  const int num_steps = 10;
  for (int step = 0; step < num_steps; ++step) {
    model_cpu.step(0.0);
    model_hip.step(0.0);
  }

  std::vector<double> psi_hip_cpu = psi_hip_gpu.to_host();

  REQUIRE(psi_cpu.size() == psi_hip_cpu.size());

  const double tolerance = 1e-10;
  size_t num_differences = 0;
  double max_diff = 0.0;

  for (size_t i = 0; i < psi_cpu.size(); ++i) {
    double diff = std::abs(psi_cpu[i] - psi_hip_cpu[i]);
    if (diff > tolerance) {
      num_differences++;
      max_diff = std::max(max_diff, diff);
    }
  }

  INFO("Number of elements with difference > " << tolerance << ": "
                                               << num_differences);
  INFO("Maximum difference: " << max_diff);

  REQUIRE(num_differences == 0);
  REQUIRE(max_diff < tolerance);
}

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
