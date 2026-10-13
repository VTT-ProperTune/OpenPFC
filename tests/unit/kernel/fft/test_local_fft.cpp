// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/kernel/fft/local_fft.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <random>

#if defined(OpenPFC_ENABLE_HEFFTE)
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/stacks/spectral_cpu_stack.hpp>
#endif

TEST_CASE("Local FFT agrees with an independent multidimensional DFT",
          "[local_fft]") {
  for (const auto &n :
       {std::array<int, 3>{5, 3, 2}, {1, 4, 3}, {7, 1, 1}, {1, 1, 1}}) {
    INFO("grid " << n[0] << " x " << n[1] << " x " << n[2]);
    pfc::fft::LocalFFT fft(n);
    std::vector<double> real(fft.size_inbox());
    for (std::size_t i = 0; i < real.size(); ++i)
      real[i] =
          std::sin(0.37 * static_cast<double>(i)) + 0.1 * static_cast<double>(i);
    const auto original = real;
    std::vector<std::complex<double>> spectrum(fft.size_outbox());
    fft.forward(real, spectrum);
    CHECK(real == original);
    for (int kz = 0; kz < n[2]; ++kz) {
      for (int ky = 0; ky < n[1]; ++ky) {
        for (int kx = 0; kx <= n[0] / 2; ++kx) {
          std::complex<double> expected{};
          for (int z = 0; z < n[2]; ++z)
            for (int y = 0; y < n[1]; ++y)
              for (int x = 0; x < n[0]; ++x) {
                const double phase = -2 * std::numbers::pi *
                                     (double(kx * x) / n[0] + double(ky * y) / n[1] +
                                      double(kz * z) / n[2]);
                expected += real[(z * n[1] + y) * n[0] + x] * std::polar(1.0, phase);
              }
          const auto actual = spectrum[(kz * n[1] + ky) * (n[0] / 2 + 1) + kx];
          CHECK(std::abs(actual - expected) < 1e-12);
        }
      }
    }
    const auto saved = spectrum;
    std::vector<double> result(real.size());
    fft.backward(spectrum, result);
    CHECK(spectrum == saved);
    for (std::size_t i = 0; i < result.size(); ++i) {
      REQUIRE(std::isfinite(result[i]));
      CHECK(std::abs(result[i] - real[i]) < 1e-12);
    }
    CHECK(fft.get_fft_time() > 0);
    fft.reset_fft_time();
    CHECK(fft.get_fft_time() == 0);
  }
}

TEST_CASE("Local FFT validates storage and buffer sizes", "[local_fft]") {
  REQUIRE_THROWS_AS((pfc::fft::LocalFFT({0, 2, 3})), std::invalid_argument);
  REQUIRE_THROWS_AS((pfc::fft::LocalFFT({2, -1, 3})), std::invalid_argument);
  const int huge = std::numeric_limits<int>::max();
  REQUIRE_THROWS_AS((pfc::fft::LocalFFT({huge, huge, huge})), std::invalid_argument);
  pfc::fft::LocalFFT fft({6, 5, 4});
  std::vector<double> real(fft.size_inbox()), wrong_real(real.size() - 1);
  std::vector<std::complex<double>> complex(fft.size_outbox()),
      wrong_complex(complex.size() + 1);
  REQUIRE_THROWS_AS(fft.forward(wrong_real, complex), std::invalid_argument);
  REQUIRE_THROWS_AS(fft.forward(real, wrong_complex), std::invalid_argument);
  REQUIRE_THROWS_AS(fft.backward(wrong_complex, real), std::invalid_argument);
  REQUIRE_THROWS_AS(fft.backward(complex, wrong_real), std::invalid_argument);
  CHECK(fft.get_allocated_memory_bytes() ==
        real.size() * sizeof(double) +
            complex.size() * sizeof(std::complex<double>));
}

#if defined(OpenPFC_ENABLE_HEFFTE)
TEST_CASE("Local FFT preserves the HeFFTe contract on varied grids", "[local_fft]") {
  // Reduced x extent 1 is tested against the independent DFT above: the
  // HeFFTe 2.4.1 FFTW reference crashes for that extent on this stack (#380).
  for (const auto &n : {std::array<int, 3>{16, 16, 16},
                        {43, 19, 6},
                        {15, 9, 7},
                        {13, 17, 11},
                        {32, 1, 1}}) {
    INFO("grid " << n[0] << " x " << n[1] << " x " << n[2]);
    auto domain = pfc::domain::create(n);
    auto decomp = pfc::decomposition::create(domain, 1);
    auto reference = pfc::fft::create(decomp, 0, MPI_COMM_SELF);
    pfc::fft::LocalFFT local(n);
    CHECK(local.get_inbox_bounds() == reference.get_inbox_bounds());
    CHECK(local.get_outbox_bounds() == reference.get_outbox_bounds());
    std::mt19937_64 rng(23456);
    std::normal_distribution<double> sample;
    // Different caller allocations for each reuse; plans never depend on them.
    for (int repeat = 0; repeat < 3; ++repeat) {
      std::vector<double> values(local.size_inbox());
      for (auto &v : values) v = sample(rng);
      const auto original = values;
      std::vector<std::complex<double>> actual(local.size_outbox()),
          expected(actual.size());
      local.forward(values, actual);
      reference.forward(values, expected);
      CHECK(values == original);
      const auto saved = actual;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(std::isfinite(actual[i].real()));
        REQUIRE(std::isfinite(actual[i].imag()));
        CHECK(std::abs(actual[i] - expected[i]) <
              1e-12 * (1 + std::abs(expected[i])));
      }
      std::vector<double> restored(values.size()), control(values.size());
      local.backward(actual, restored);
      reference.backward(expected, control);
      CHECK(actual == saved);
      for (std::size_t i = 0; i < restored.size(); ++i) {
        REQUIRE(std::isfinite(restored[i]));
        CHECK(std::abs(restored[i] - values[i]) < 1e-12);
        CHECK(std::abs(restored[i] - control[i]) < 1e-12);
      }
    }
  }
}

TEST_CASE("Host local stacks reject inconsistent MPI ownership",
          "[local_fft][MPI]") {
  using Stack = pfc::sim::stacks::SpectralCPUStack;
  const auto local = pfc::fft::Implementation::Local;
  auto domain = pfc::domain::create({8, 8, 8});
  REQUIRE_THROWS_AS((Stack(domain, 0, 1, MPI_COMM_NULL, local)),
                    std::invalid_argument);
  Stack independent(domain, 0, 1, MPI_COMM_SELF, local);
  CHECK(independent.fft_choice() == local);
  CHECK(dynamic_cast<pfc::fft::LocalFFT *>(&independent.fft()) != nullptr);
  CHECK(independent.u().box() == pfc::domain::index_box(domain));
  REQUIRE_THROWS_AS((Stack(domain, 0, 2, MPI_COMM_SELF, local)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS((Stack(domain, 1, 1, MPI_COMM_SELF, local)),
                    std::invalid_argument);
  int count = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &count);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (count > 1) {
    REQUIRE_THROWS_AS((Stack(domain, 0, 1, MPI_COMM_WORLD, local)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS((Stack(domain, rank, count, MPI_COMM_WORLD, local)),
                      std::invalid_argument);
  }
}
#endif
