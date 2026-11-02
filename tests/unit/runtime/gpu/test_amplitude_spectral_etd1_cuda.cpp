// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * CUDA complex-amplitude ETD1 against the host stepper and against
 * exp(s dt) on an even symbol. HIP is not run.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <span>
#include <sstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <fixtures/session.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/amplitude_spectral_etd1.hpp>
#include <openpfc/runtime/cuda/fft_cuda.hpp>
#include <openpfc/runtime/gpu/amplitude_spectral_etd1_cuda.hpp>

#include "test_helpers.hpp"

using pfc::sim::AmplitudeSpectralETD1;
using pfc::sim::AmplitudeSpectralETD1Cuda;

namespace {

constexpr int kN = 8;

pfc::Domain square() {
  return pfc::domain::create(pfc::GridSize({kN, kN, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({1.0, 1.0, 1.0}));
}

void zero_n(const std::vector<std::span<const double>> &,
            const std::vector<std::span<const double>> &,
            const std::vector<std::span<double>> &,
            const std::vector<std::span<double>> &) {}

void fill_wave(std::span<double> re, std::span<double> im, int shift) {
  for (int y = 0; y < kN; ++y) {
    for (int x = 0; x < kN; ++x) {
      const std::size_t p = static_cast<std::size_t>(x + kN * y);
      re[p] = 0.15 + 0.04 * std::cos(2.0 * pfc::pi * x / kN + 0.2 * shift);
      im[p] = 0.01 * std::sin(2.0 * pfc::pi * y / kN) * (1.0 + 0.1 * shift);
    }
  }
}

double max_abs(std::span<const double> a, std::span<const double> b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

void constant_symbol(auto &sys, std::size_t j, double s_plus, double s_minus) {
  std::vector<double> plus(sys.outbox_size(), s_plus);
  std::vector<double> minus(sys.outbox_size(), s_minus);
  sys.set_symbol(j, plus, minus);
}

struct ApfcRates {
  double t_coef;
  void operator()(const std::vector<std::span<const double>> &re,
                  const std::vector<std::span<const double>> &im,
                  const std::vector<std::span<double>> &nre,
                  const std::vector<std::span<double>> &nim) const {
    const std::size_t n = re[0].size();
    for (std::size_t p = 0; p < n; ++p) {
      std::complex<double> e[3];
      double a2 = 0.0;
      for (int l = 0; l < 3; ++l) {
        e[l] = {re[static_cast<std::size_t>(l)][p],
                im[static_cast<std::size_t>(l)][p]};
        a2 += 2.0 * std::norm(e[l]);
      }
      for (int j = 0; j < 3; ++j) {
        const int l1 = (j + 1) % 3;
        const int l2 = (j + 2) % 3;
        const auto v = 3.0 * (a2 - std::norm(e[j])) * e[j] -
                       2.0 * t_coef * std::conj(e[l1]) * std::conj(e[l2]);
        nre[static_cast<std::size_t>(j)][p] = -v.real();
        nim[static_cast<std::size_t>(j)][p] = -v.imag();
      }
    }
  }
};

void print_line(const char *name, double value) {
  std::ostringstream out;
  out.setf(std::ios::scientific);
  out.precision(17);
  out << "AMPLITUDE_ETD1_CUDA " << name << " " << value << "\n";
  std::cout << out.str();
}

} // namespace

TEST_CASE("CUDA amplitude ETD1 matches exp(s dt) and the host APFC step",
          "[gpu][amplitude_etd1]") {
  if (!pfc::gpu::test::is_cuda_available()) {
    SKIP("CUDA device not available");
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const auto domain = square();
  auto decomp = pfc::decomposition::create(domain, 1);
  auto host_fft = pfc::fft::create(decomp);
  auto device_fft = pfc::fft::create_cuda(decomp, rank, MPI_COMM_WORLD);

  constexpr double s0 = -1.25;
  constexpr double dt = 0.02;
  AmplitudeSpectralETD1 host(host_fft, 1);
  AmplitudeSpectralETD1Cuda device(device_fft, 1);
  fill_wave(host.real(0), host.imag(0), 0);
  std::copy(host.real(0).begin(), host.real(0).end(), device.real(0).begin());
  std::copy(host.imag(0).begin(), host.imag(0).end(), device.imag(0).begin());
  const std::vector<double> re0(host.real(0).begin(), host.real(0).end());
  const std::vector<double> im0(host.imag(0).begin(), host.imag(0).end());
  constant_symbol(host, 0, s0, s0);
  constant_symbol(device, 0, s0, s0);
  host.set_dt(dt);
  device.set_dt(dt);
  REQUIRE(host.step(0.0, zero_n) == dt);
  REQUIRE(device.step(0.0, zero_n) == dt);
  const double factor = std::exp(s0 * dt);
  std::vector<double> want_re = re0;
  std::vector<double> want_im = im0;
  for (double &v : want_re) v *= factor;
  for (double &v : want_im) v *= factor;
  const double host_err =
      std::max(max_abs(want_re, host.real(0)), max_abs(want_im, host.imag(0)));
  const double device_err =
      std::max(max_abs(want_re, device.real(0)), max_abs(want_im, device.imag(0)));
  const double pair_err = std::max(max_abs(host.real(0), device.real(0)),
                                   max_abs(host.imag(0), device.imag(0)));
  print_line("even_host", host_err);
  print_line("even_device", device_err);
  print_line("even_pair", pair_err);
  REQUIRE(host_err < 1e-12);
  REQUIRE(device_err < 1e-10);
  REQUIRE(pair_err < 1e-10);

  constexpr double psibar = -0.15;
  constexpr double t_coef = -3.0 * psibar;
  constexpr double dB = -0.1 + 3.0 * psibar * psibar;
  constexpr double dt_apfc = 0.01;
  ApfcRates rates{t_coef};
  AmplitudeSpectralETD1 host3(host_fft, 3);
  AmplitudeSpectralETD1Cuda device3(device_fft, 3);
  const std::size_t m = host3.outbox_size();
  REQUIRE(device3.outbox_size() == m);
  for (std::size_t j = 0; j < 3; ++j) {
    fill_wave(host3.real(j), host3.imag(j), static_cast<int>(j));
    std::copy(host3.real(j).begin(), host3.real(j).end(), device3.real(j).begin());
    std::copy(host3.imag(j).begin(), host3.imag(j).end(), device3.imag(j).begin());
    std::vector<double> plus(m);
    std::vector<double> minus(m);
    for (std::size_t q = 0; q < m; ++q) {
      const double q2 = 0.15 * static_cast<double>(q);
      const double shift = 0.2 * static_cast<double>(j + 1);
      plus[q] = -(dB + (q2 + shift) * (q2 + shift));
      minus[q] = -(dB + (q2 - shift) * (q2 - shift));
    }
    host3.set_symbol(j, plus, minus);
    device3.set_symbol(j, plus, minus);
  }
  host3.set_dt(dt_apfc);
  device3.set_dt(dt_apfc);
  REQUIRE(host3.step(0.0, rates) == dt_apfc);
  REQUIRE(device3.step(0.0, rates) == dt_apfc);
  double apfc = 0.0;
  for (std::size_t j = 0; j < 3; ++j) {
    apfc = std::max(apfc, max_abs(host3.real(j), device3.real(j)));
    apfc = std::max(apfc, max_abs(host3.imag(j), device3.imag(j)));
  }
  print_line("apfc_pair", apfc);
  REQUIRE(apfc < 1e-10);
}

int main(int argc, char *argv[]) {
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(&argc, &argv);
  }
  const int result = pfc::test::run(argc, argv);
  int mpi_finalized = 0;
  MPI_Finalized(&mpi_finalized);
  if (mpi_finalized == 0) {
    MPI_Finalize();
  }
  return result;
}
