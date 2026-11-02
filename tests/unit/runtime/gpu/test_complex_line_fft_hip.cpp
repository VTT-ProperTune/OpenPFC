// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "../../../fixtures/session.hpp"

#if !defined(OpenPFC_ENABLE_HIP)

int main(int argc, char *argv[]) { return pfc::test::run(argc, argv); }

#else

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <hip/hip_runtime_api.h>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/fft/complex_line_fft.hpp>
#include <openpfc/runtime/gpu/complex_line_fft_hip.hpp>

using Catch::Matchers::WithinAbs;

namespace {
bool device_present() {
  int count = 0;
  return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

std::vector<std::complex<double>> even_extension(const std::vector<double> &values) {
  const int n = static_cast<int>(values.size()) - 1;
  const int m = 2 * n;
  std::vector<std::complex<double>> extended(static_cast<std::size_t>(m));
  extended[0] = values[0];
  for (int j = 1; j < n; ++j) {
    extended[static_cast<std::size_t>(j)] = values[static_cast<std::size_t>(j)];
    extended[static_cast<std::size_t>(m - j)] = values[static_cast<std::size_t>(j)];
  }
  extended[static_cast<std::size_t>(n)] = values[static_cast<std::size_t>(n)];
  return extended;
}
} // namespace

TEST_CASE("HIP line DFT rejects a short plan", "[fft][line_dft][hip]") {
  REQUIRE_THROWS_AS(pfc::fft::HipComplexLineFFT(1), std::invalid_argument);
}

TEST_CASE("HIP line DFT matches the host complex line DFT", "[fft][line_dft][hip]") {
  if (!device_present()) {
    SKIP("HIP device not available");
  }
  constexpr int n = 8;
  pfc::fft::HipComplexLineFFT device(n);
  REQUIRE(device.length() == n);
  std::vector<std::complex<double>> in(n);
  std::vector<std::complex<double>> short_out(n - 1);
  REQUIRE_THROWS_AS(device.forward(in, short_out), std::invalid_argument);

  in[0] = {1.0, 0.0};
  std::vector<std::complex<double>> spectrum(n);
  const auto impulse = in;
  device.forward(in, spectrum);
  REQUIRE(in == impulse);
  for (const auto &bin : spectrum) {
    REQUIRE_THAT(bin.real(), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(bin.imag(), WithinAbs(0.0, 1e-12));
  }

  for (int i = 0; i < n; ++i) {
    in[static_cast<std::size_t>(i)] = {std::sin(0.3 * i), std::cos(0.7 * i)};
  }
  const auto original = in;
  pfc::fft::ComplexLineFFT host(n);
  std::vector<std::complex<double>> host_spectrum(n);
  host.forward(in, host_spectrum);
  device.forward(in, spectrum);
  REQUIRE(in == original);
  for (int i = 0; i < n; ++i) {
    REQUIRE_THAT(
        spectrum[static_cast<std::size_t>(i)].real(),
        WithinAbs(host_spectrum[static_cast<std::size_t>(i)].real(), 1e-12));
    REQUIRE_THAT(
        spectrum[static_cast<std::size_t>(i)].imag(),
        WithinAbs(host_spectrum[static_cast<std::size_t>(i)].imag(), 1e-12));
  }
  std::vector<std::complex<double>> back(n);
  device.backward(spectrum, back);
  for (int i = 0; i < n; ++i) {
    const auto &sample = original[static_cast<std::size_t>(i)];
    const auto &returned = back[static_cast<std::size_t>(i)];
    REQUIRE_THAT(returned.real() / n, WithinAbs(sample.real(), 1e-12));
    REQUIRE_THAT(returned.imag() / n, WithinAbs(sample.imag(), 1e-12));
  }
}

TEST_CASE("HIP line DFT matches the host even extension", "[fft][line_dft][hip]") {
  if (!device_present()) {
    SKIP("HIP device not available");
  }
  const std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0};
  const auto extended = even_extension(values);
  pfc::fft::ComplexLineFFT host(static_cast<int>(extended.size()));
  pfc::fft::HipComplexLineFFT device(static_cast<int>(extended.size()));
  std::vector<std::complex<double>> host_bins(extended.size());
  std::vector<std::complex<double>> device_bins(extended.size());
  host.forward(extended, host_bins);
  device.forward(extended, device_bins);
  for (std::size_t i = 0; i < extended.size(); ++i) {
    REQUIRE_THAT(device_bins[i].real(), WithinAbs(host_bins[i].real(), 1e-12));
    REQUIRE_THAT(device_bins[i].imag(), WithinAbs(host_bins[i].imag(), 1e-12));
  }
}

int main(int argc, char *argv[]) { return pfc::test::run(argc, argv); }

#endif
