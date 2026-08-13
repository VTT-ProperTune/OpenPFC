// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// CPU-only clang-tidy builds omit OpenPFC_ENABLE_HIP_SPECTRAL; this TU must
// still parse.
#if !defined(OpenPFC_ENABLE_HIP_SPECTRAL)

#include <catch2/catch_session.hpp>

int main(int argc, char *argv[]) { return Catch::Session().run(argc, argv); }

#else

#include "test_helpers.hpp"
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/hip/fft_hip.hpp>
#include <vector>

using Catch::Approx;

TEST_CASE("HIP FFT: Forward transform", "[gpu][fft][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  auto world = pfc::domain::create_world(pfc::GridSize({64, 64, 64}),
                                         pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                         pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(world, mpi_size);

  int rank_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto gpu_fft = pfc::fft::create_hip(decomp, rank_id, MPI_COMM_WORLD);

  pfc::core::DataBuffer<pfc::backend::HipTag, double> input(gpu_fft.size_inbox());
  pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> output(
      gpu_fft.size_outbox());

  std::vector<double> input_host(gpu_fft.size_inbox(), 1.0);
  input.copy_from_host(input_host);

  gpu_fft.forward(input, output);

  std::vector<std::complex<double>> output_host = output.to_host();

  if (output_host.size() > 0) {
    REQUIRE(std::abs(output_host[0].real()) > 0.0);
  }

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

TEST_CASE("HIP FFT: Backward transform", "[gpu][fft][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  auto world = pfc::domain::create_world(pfc::GridSize({64, 64, 64}),
                                         pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                         pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(world, mpi_size);

  int rank_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto gpu_fft = pfc::fft::create_hip(decomp, rank_id, MPI_COMM_WORLD);

  pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> input(
      gpu_fft.size_outbox());
  pfc::core::DataBuffer<pfc::backend::HipTag, double> output(gpu_fft.size_inbox());

  std::vector<std::complex<double>> input_host(gpu_fft.size_outbox(), 0.0);
  if (input_host.size() > 0) {
    input_host[0] = std::complex<double>(1000.0, 0.0);
  }
  input.copy_from_host(input_host);

  gpu_fft.backward(input, output);

  std::vector<double> output_host = output.to_host();

  if (output_host.size() > 0) {
    double expected = 1000.0 / (64.0 * 64.0 * 64.0);
    REQUIRE(output_host[0] == Approx(expected).margin(1e-6));
  }

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

TEST_CASE("HIP FFT: Round-trip (forward then backward)", "[gpu][fft][hip]") {
  if (!pfc::gpu::test::is_hip_available()) {
    SKIP("HIP not available");
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(nullptr, nullptr);
  }

  auto world = pfc::domain::create_world(pfc::GridSize({32, 32, 32}),
                                         pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                         pfc::GridSpacing({1.0, 1.0, 1.0}));
  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  auto decomp = pfc::decomposition::create(world, mpi_size);

  int rank_id;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto gpu_fft = pfc::fft::create_hip(decomp, rank_id, MPI_COMM_WORLD);

  pfc::core::DataBuffer<pfc::backend::HipTag, double> input(gpu_fft.size_inbox());
  std::vector<double> input_host(gpu_fft.size_inbox());
  for (size_t i = 0; i < input_host.size(); ++i) {
    input_host[i] = static_cast<double>(i % 2);
  }
  input.copy_from_host(input_host);

  pfc::core::DataBuffer<pfc::backend::HipTag, std::complex<double>> fourier(
      gpu_fft.size_outbox());
  pfc::core::DataBuffer<pfc::backend::HipTag, double> output(gpu_fft.size_inbox());

  gpu_fft.forward(input, fourier);
  gpu_fft.backward(fourier, output);

  std::vector<double> output_host = output.to_host();

  bool roundtrip_matches = output_host.size() == input_host.size();
  for (size_t i = 0; i < output_host.size(); ++i) {
    roundtrip_matches &= std::abs(output_host[i] - input_host[i]) <= 1e-5;
  }
  REQUIRE(roundtrip_matches);

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }

  int result = Catch::Session().run(argc, argv);

  if (mpi_initialized == 0) {
    MPI_Finalize();
  }

  return result;
}

#endif // OpenPFC_ENABLE_HIP_SPECTRAL
