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
#include <openpfc/kernel/execution/databuffer.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
#include <openpfc/runtime/gpu/backend_tags_gpu.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/hip/fft_hip.hpp>
#endif

using namespace pfc;

static inline World make_world(int nx, int ny, int nz) {
  return domain::create_world(GridSize({nx, ny, nz}), PhysicalOrigin({0.0, 0.0, 0.0}),
                       GridSpacing({1.0, 1.0, 1.0}));
}

#if defined(OpenPFC_ENABLE_HIP_SPECTRAL)
TEST_CASE("HIP FFT roundtrip (double) [integration][gpu][hip]", "[gpu][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = make_world(16, 16, 16);
  auto decomp = decomposition::create(world, size);

  // Cray MPICH types MPI_Comm as int, so the two-arg overload is ambiguous.
  auto fft = fft::create_hip(decomp, rank, MPI_COMM_WORLD);

  core::DataBuffer<backend::HIPTag, double> real_in(fft.size_inbox());
  core::DataBuffer<backend::HIPTag, double> real_out(fft.size_inbox());
  core::DataBuffer<backend::HIPTag, std::complex<double>> freq(fft.size_outbox());

  std::vector<double> host_in(fft.size_inbox());
  for (size_t i = 0; i < host_in.size(); ++i) {
    host_in[i] =
        0.25 + 0.5 * std::sin(2.0 * std::numbers::pi * static_cast<double>(i) /
                              static_cast<double>(host_in.size()));
  }
  real_in.copy_from_host(host_in);

  fft::IDeviceFFT<HIPSpace> &iface = fft;
  iface.forward(real_in, freq);
  iface.backward(freq, real_out);

  auto host_out = real_out.to_host();

  bool roundtrip_matches = host_out.size() == host_in.size();
  for (size_t i = 0; i < host_in.size(); ++i) {
    roundtrip_matches &= std::abs(host_out[i] - host_in[i]) <= 1e-10;
  }
  REQUIRE(roundtrip_matches);
}

TEST_CASE("HIP FFT roundtrip (float) [integration][gpu][hip]", "[gpu][hip]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = make_world(16, 16, 16);
  auto decomp = decomposition::create(world, size);

  auto fft = fft::create_hip(decomp, rank, MPI_COMM_WORLD);

  core::DataBuffer<backend::HIPTag, float> real_in(fft.size_inbox());
  core::DataBuffer<backend::HIPTag, float> real_out(fft.size_inbox());
  core::DataBuffer<backend::HIPTag, std::complex<float>> freq(fft.size_outbox());

  std::vector<float> host_in(fft.size_inbox());
  for (size_t i = 0; i < host_in.size(); ++i) {
    host_in[i] = 0.25f + 0.5f * std::sin(2.0f * std::numbers::pi_v<float> *
                                         static_cast<float>(i) /
                                         static_cast<float>(host_in.size()));
  }
  real_in.copy_from_host(host_in);

  fft.forward(real_in, freq);
  fft.backward(freq, real_out);

  auto host_out = real_out.to_host();

  bool roundtrip_matches = host_out.size() == host_in.size();
  for (size_t i = 0; i < host_in.size(); ++i) {
    roundtrip_matches &= std::abs(host_out[i] - host_in[i]) <= 1e-5f;
  }
  REQUIRE(roundtrip_matches);
}
#else
TEST_CASE("HIP FFT roundtrip skipped (HIP spectral disabled) [integration][gpu][hip]",
          "[gpu][hip]") {
  SUCCEED("HIP spectral not enabled; skipping GPU roundtrip test");
}
#endif
