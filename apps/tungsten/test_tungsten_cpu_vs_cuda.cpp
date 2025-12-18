// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_tungsten_cpu_vs_cuda.cpp
 * @brief Test to compare CPU and CUDA implementations of Tungsten model
 *
 * This test verifies that the CPU and CUDA implementations produce
 * the same results (within floating-point precision tolerance).
 */

#define CATCH_CONFIG_RUNNER
#if !defined(OpenPFC_ENABLE_CUDA)
#error "This test requires CUDA support. Enable with -DOpenPFC_ENABLE_CUDA=ON"
#endif

#include "tungsten_cuda_model.hpp"
#include "tungsten_model.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/fft_cuda.hpp>
#include <vector>

using namespace pfc;
using namespace Catch::Matchers;

TEST_CASE("Tungsten CPU vs CUDA: Same results", "[Tungsten][CPU][CUDA]") {
  // Create world
  auto world = world::create(GridSize({32, 32, 32}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto decomp = decomposition::create(world, size);
  auto fft_cpu = fft::create(decomp, rank);

  // Create models (CUDA uses double precision for comparison)
  Tungsten model_cpu(fft_cpu, world);
  TungstenCUDA<double> model_cuda(fft_cpu, world);

  // Set same parameters
  model_cpu.params.set_n0(-0.4);
  model_cpu.params.set_T(0.5);
  model_cuda.params.set_n0(-0.4);
  model_cuda.params.set_T(0.5);

  // Initialize both models
  double dt = 0.01;
  model_cpu.initialize(dt);
  model_cuda.initialize(dt);

  // Initialize fields with same initial condition
  auto &psi_cpu = model_cpu.get_real_field("psi");
  auto &psi_cuda_gpu = model_cuda.get_psi();

  // Set initial condition on CPU
  for (size_t i = 0; i < psi_cpu.size(); ++i) {
    psi_cpu[i] = -0.4 + 0.1 * std::sin(2.0 * M_PI * i / psi_cpu.size());
  }

  // Copy to GPU
  std::vector<double> psi_init(psi_cpu.begin(), psi_cpu.end());
  psi_cuda_gpu.copy_from_host(psi_init);

  // Run a few steps
  const int num_steps = 10;
  for (int step = 0; step < num_steps; ++step) {
    model_cpu.step(0.0);
    model_cuda.step(0.0);
  }

  // Compare results
  std::vector<double> psi_cuda_cpu = psi_cuda_gpu.to_host();

  REQUIRE(psi_cpu.size() == psi_cuda_cpu.size());

  // Check that results match (within floating-point tolerance)
  const double tolerance =
      1e-10; // Very tight tolerance since both use double precision
  size_t num_differences = 0;
  double max_diff = 0.0;

  for (size_t i = 0; i < psi_cpu.size(); ++i) {
    double diff = std::abs(psi_cpu[i] - psi_cuda_cpu[i]);
    if (diff > tolerance) {
      num_differences++;
      max_diff = std::max(max_diff, diff);
    }
  }

  INFO("Number of elements with difference > " << tolerance << ": "
                                               << num_differences);
  INFO("Maximum difference: " << max_diff);

  // Allow some small differences due to floating-point operation order
  // but should be very close
  REQUIRE(num_differences == 0);
  REQUIRE(max_diff < tolerance);
}

// MPI-aware main function for Catch2
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
