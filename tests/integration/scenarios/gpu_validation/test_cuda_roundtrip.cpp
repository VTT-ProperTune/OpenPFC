// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <vector>

#include <openpfc/core/databuffer.hpp>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <openpfc/core/backend_tags.hpp>
#include <openpfc/fft_cuda.hpp>
#endif

using namespace pfc;

// Utility to create a small test world
static inline World make_world(int nx, int ny, int nz) {
  return world::create(GridSize({nx, ny, nz}), PhysicalOrigin({0.0, 0.0, 0.0}),
                       GridSpacing({1.0, 1.0, 1.0}));
}

#if defined(OpenPFC_ENABLE_CUDA)
TEST_CASE("CUDA FFT roundtrip (double) [integration][gpu]", "[gpu]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = make_world(16, 16, 16);
  auto decomp = decomposition::create(world, size);

  auto fft = fft::create_cuda(decomp, rank);

  // Allocate GPU buffers
  core::DataBuffer<backend::CudaTag, double> real_in(fft.size_inbox());
  core::DataBuffer<backend::CudaTag, double> real_out(fft.size_inbox());
  core::DataBuffer<backend::CudaTag, std::complex<double>> freq(fft.size_outbox());

  // Initialize host data with a smooth function and copy to device
  std::vector<double> host_in(fft.size_inbox());
  for (size_t i = 0; i < host_in.size(); ++i) {
    host_in[i] = 0.25 + 0.5 * std::sin(2.0 * M_PI * static_cast<double>(i) /
                                       static_cast<double>(host_in.size()));
  }
  real_in.copy_from_host(host_in);

  // Roundtrip on GPU
  fft.forward(real_in, freq);
  fft.backward(freq, real_out);

  // Copy back and verify
  auto host_out = real_out.to_host();

  // Expect near-exact roundtrip
  REQUIRE(host_out.size() == host_in.size());
  for (size_t i = 0; i < host_in.size(); ++i) {
    REQUIRE(host_out[i] == Catch::Approx(host_in[i]).margin(1e-10));
  }
}

TEST_CASE("CUDA FFT roundtrip (float) [integration][gpu]", "[gpu]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = make_world(16, 16, 16);
  auto decomp = decomposition::create(world, size);

  auto fft = fft::create_cuda(decomp, rank);

  // Allocate GPU buffers
  core::DataBuffer<backend::CudaTag, float> real_in(fft.size_inbox());
  core::DataBuffer<backend::CudaTag, float> real_out(fft.size_inbox());
  core::DataBuffer<backend::CudaTag, std::complex<float>> freq(fft.size_outbox());

  // Initialize host data with a smooth function and copy to device
  std::vector<float> host_in(fft.size_inbox());
  for (size_t i = 0; i < host_in.size(); ++i) {
    host_in[i] = 0.25f + 0.5f * std::sin(2.0f * static_cast<float>(M_PI) *
                                         static_cast<float>(i) /
                                         static_cast<float>(host_in.size()));
  }
  real_in.copy_from_host(host_in);

  // Roundtrip on GPU
  fft.forward(real_in, freq);
  fft.backward(freq, real_out);

  // Copy back and verify (looser tolerance for float)
  auto host_out = real_out.to_host();

  REQUIRE(host_out.size() == host_in.size());
  for (size_t i = 0; i < host_in.size(); ++i) {
    REQUIRE(host_out[i] == Catch::Approx(host_in[i]).margin(1e-5f));
  }
}
#else
TEST_CASE("CUDA FFT roundtrip skipped (CUDA disabled) [integration][gpu]", "[gpu]") {
  SUCCEED("CUDA not enabled; skipping GPU roundtrip test");
}
#endif
