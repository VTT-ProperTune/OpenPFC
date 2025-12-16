// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "test_helpers.hpp"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <openpfc/core/backend_tags.hpp>
#include <openpfc/core/databuffer.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>
#include <openpfc/fft_cuda.hpp>
#include <vector>

using Catch::Approx;

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#include <mpi.h>
#endif

TEST_CASE("GPU FFT: Forward transform", "[gpu][fft]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  // Initialize MPI if not already initialized
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }

  // Create a simple world and decomposition
  auto world = pfc::world::create(pfc::GridSize({64, 64, 64}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(world, mpi_size);

  // Create GPU FFT
  int rank_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank_id);

  // Create input data (simple pattern: all ones)
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> input(gpu_fft.size_inbox());
  pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> output(
      gpu_fft.size_outbox());

  // Initialize input on CPU, then copy to GPU
  std::vector<double> input_host(gpu_fft.size_inbox(), 1.0);
  input.copy_from_host(input_host);

  // Perform forward FFT
  gpu_fft.forward(input, output);

  // Copy output back to CPU for verification
  std::vector<std::complex<double>> output_host = output.to_host();

  // Verify DC component is non-zero (sum of all ones)
  // The DC component should be approximately the total number of grid points
  if (output_host.size() > 0) {
    REQUIRE(std::abs(output_host[0].real()) > 0.0);
  }

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

TEST_CASE("GPU FFT: Backward transform", "[gpu][fft]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  // Initialize MPI if not already initialized
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }

  // Create a simple world and decomposition
  auto world = pfc::world::create(pfc::GridSize({64, 64, 64}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(world, mpi_size);

  // Create GPU FFT
  int rank_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank_id);

  // Create input in Fourier space (DC component only)
  pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> input(
      gpu_fft.size_outbox());
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> output(gpu_fft.size_inbox());

  // Initialize input on CPU, then copy to GPU
  std::vector<std::complex<double>> input_host(gpu_fft.size_outbox(), 0.0);
  if (input_host.size() > 0) {
    input_host[0] = std::complex<double>(1000.0, 0.0); // DC component
  }
  input.copy_from_host(input_host);

  // Perform backward FFT
  gpu_fft.backward(input, output);

  // Copy output back to CPU for verification
  std::vector<double> output_host = output.to_host();

  // Verify output is approximately constant (DC component in real space)
  if (output_host.size() > 0) {
    double expected = 1000.0 / (64.0 * 64.0 * 64.0); // Normalized by grid size
    REQUIRE(output_host[0] == Approx(expected).margin(1e-6));
  }

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

TEST_CASE("GPU FFT: Round-trip (forward then backward)", "[gpu][fft]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA not available");
  }

  // Initialize MPI if not already initialized
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(nullptr, nullptr);
  }

  // Create a simple world and decomposition
  auto world = pfc::world::create(pfc::GridSize({32, 32, 32}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(world, mpi_size);

  // Create GPU FFT
  int rank_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  auto gpu_fft = pfc::fft::create_cuda(decomp, rank_id);

  // Create input data (simple pattern: alternating values)
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> input(gpu_fft.size_inbox());
  std::vector<double> input_host(gpu_fft.size_inbox());
  for (size_t i = 0; i < input_host.size(); ++i) {
    input_host[i] = static_cast<double>(i % 2); // 0, 1, 0, 1, ...
  }
  input.copy_from_host(input_host);

  pfc::core::DataBuffer<pfc::backend::CudaTag, std::complex<double>> fourier(
      gpu_fft.size_outbox());
  pfc::core::DataBuffer<pfc::backend::CudaTag, double> output(gpu_fft.size_inbox());

  // Forward FFT
  gpu_fft.forward(input, fourier);

  // Backward FFT (should recover original, normalized)
  gpu_fft.backward(fourier, output);

  // Copy output back to CPU for verification
  std::vector<double> output_host = output.to_host();

  // Verify round-trip (within numerical precision)
  REQUIRE(output_host.size() == input_host.size());
  for (size_t i = 0; i < output_host.size(); ++i) {
    REQUIRE(output_host[i] == Approx(input_host[i]).margin(1e-5));
  }

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

int main(int argc, char *argv[]) {
  // Initialize MPI for tests
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    MPI_Init(&argc, &argv);
  }

  int result = Catch::Session().run(argc, argv);

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }

  return result;
}
