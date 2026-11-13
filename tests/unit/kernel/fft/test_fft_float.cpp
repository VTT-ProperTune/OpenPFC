// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_fft_float.cpp
 * @brief Single-precision host transforms (ADR 0006): the HeFFTe FFTW path and
 *        `LocalFFT` round-trip float data and agree with their own double
 *        transform of the same samples to single-precision accuracy.
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/fft/local_fft.hpp>

namespace {

constexpr int N = 8;

std::vector<double> random_samples(std::size_t n, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(n);
  for (double &x : v) {
    x = dist(gen);
  }
  return v;
}

/// Run forward+backward in both precisions on the same samples and return
/// (max |float round-trip - input|, max |float spectrum - double spectrum|).
template <class FFT>
std::pair<double, double> compare_precisions(FFT &fft, unsigned seed) {
  const auto in_d = random_samples(fft.size_inbox(), seed);
  std::vector<float> in_f(in_d.begin(), in_d.end());

  std::vector<std::complex<double>> hat_d(fft.size_outbox());
  std::vector<std::complex<float>> hat_f(fft.size_outbox());
  fft.forward(in_d, hat_d);
  fft.forward(in_f, hat_f);

  double max_spec = 0.0;
  for (std::size_t i = 0; i < hat_d.size(); ++i) {
    max_spec = std::max(
        max_spec, std::abs(static_cast<std::complex<double>>(hat_f[i]) - hat_d[i]));
  }

  std::vector<float> back_f(fft.size_inbox());
  fft.backward(hat_f, back_f);
  double max_rt = 0.0;
  for (std::size_t i = 0; i < back_f.size(); ++i) {
    max_rt = std::max(max_rt, std::abs(static_cast<double>(back_f[i]) - in_f[i]));
  }
  return {max_rt, max_spec};
}

} // namespace

TEST_CASE("HeFFTe host FFT transforms float buffers", "[fft][float]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto domain = pfc::domain::create(pfc::GridSize({N, N, N}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(domain, size);
  auto fft = pfc::fft::create(decomp);
  pfc::fft::IHostFFT &iface = fft;

  const std::size_t before = iface.get_allocated_memory_bytes();
  const auto [max_rt, max_spec] = compare_precisions(iface, 7u);
  // The spectrum of N^3 = 512 unit-scale samples is O(sqrt(512)) ~ 23, so
  // single-precision agreement is a few times 1e-6 absolute.
  REQUIRE(max_rt < 1e-5);
  REQUIRE(max_spec < 1e-4);
  // The float workspace is allocated on first use and reported.
  REQUIRE(iface.get_allocated_memory_bytes() > before);

  SECTION("float buffers obey the inbox/outbox size contract") {
    std::vector<float> wrong(iface.size_inbox() + 1);
    std::vector<std::complex<float>> out(iface.size_outbox());
    REQUIRE_THROWS_AS(iface.forward(wrong, out), std::invalid_argument);
    std::vector<float> back(iface.size_inbox());
    std::vector<std::complex<float>> wrong_hat(iface.size_outbox() + 1);
    REQUIRE_THROWS_AS(iface.backward(wrong_hat, back), std::invalid_argument);
  }
}

TEST_CASE("LocalFFT transforms float buffers", "[fft][float][local_fft]") {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    SKIP("LocalFFT is single-rank");
  }
  pfc::fft::LocalFFT fft({N, N, N});
  pfc::fft::IHostFFT &iface = fft;

  const std::size_t before = iface.get_allocated_memory_bytes();
  const auto [max_rt, max_spec] = compare_precisions(iface, 11u);
  REQUIRE(max_rt < 1e-5);
  REQUIRE(max_spec < 1e-4);
  REQUIRE(iface.get_allocated_memory_bytes() > before);
}
