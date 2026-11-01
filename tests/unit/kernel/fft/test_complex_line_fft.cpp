// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/fft/complex_line_fft.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Complex line DFT is an unnormalized complex transform",
          "[fft][line_dft]") {
  constexpr int n = 8;
  pfc::fft::ComplexLineFFT fft(n);
  REQUIRE(fft.length() == n);

  std::vector<std::complex<double>> impulse(n);
  impulse[0] = {1.0, 0.0};
  std::vector<std::complex<double>> spectrum(n);
  const auto impulse_original = impulse;
  fft.forward(impulse, spectrum);
  REQUIRE(impulse == impulse_original);
  for (const auto &bin : spectrum) {
    REQUIRE_THAT(bin.real(), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(bin.imag(), WithinAbs(0.0, 1e-12));
  }

  std::vector<std::complex<double>> input(n);
  for (int i = 0; i < n; ++i) {
    input[static_cast<std::size_t>(i)] = {std::sin(0.3 * i), std::cos(0.7 * i)};
  }
  const auto original = input;
  fft.forward(input, spectrum);
  std::vector<std::complex<double>> back(n);
  fft.backward(spectrum, back);
  REQUIRE(input == original);
  REQUIRE(spectrum.size() == static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const auto &sample = input[static_cast<std::size_t>(i)];
    const auto &returned = back[static_cast<std::size_t>(i)];
    REQUIRE_THAT(returned.real() / n, WithinAbs(sample.real(), 1e-12));
    REQUIRE_THAT(returned.imag() / n, WithinAbs(sample.imag(), 1e-12));
  }
}

TEST_CASE("Complex line DFT rejects a short plan and a wrong span",
          "[fft][line_dft]") {
  REQUIRE_THROWS_AS(pfc::fft::ComplexLineFFT(1), std::invalid_argument);
  pfc::fft::ComplexLineFFT fft(4);
  std::vector<std::complex<double>> in(4);
  std::vector<std::complex<double>> short_out(3);
  REQUIRE_THROWS_AS(fft.forward(in, short_out), std::invalid_argument);
  std::vector<std::complex<double>> short_in(3);
  REQUIRE_THROWS_AS(fft.backward(short_in, in), std::invalid_argument);
}
