// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Complex-amplitude ETD1 on the host FFT. Linear decay is the closed
 * form exp(s dt). The manufactured three-amplitude step is the APFC
 * reference update, checked with a real-space mix that is not the
 * stepper's c1/c2 loop.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/amplitude_spectral_etd1.hpp>

using pfc::sim::AmplitudeSpectralETD1;

namespace {

constexpr int kN = 8;

pfc::Domain square() {
  const double dx = 1.0;
  return pfc::domain::create(pfc::GridSize({kN, kN, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, 1.0}));
}

void zero_nonlinear(const std::vector<std::span<const double>> &,
                    const std::vector<std::span<const double>> &,
                    const std::vector<std::span<double>> &,
                    const std::vector<std::span<double>> &) {}

double max_abs(const std::vector<double> &a, std::span<const double> b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

void fill_wave(std::span<double> re, std::span<double> im, int shift) {
  const double dx = 1.0;
  for (int y = 0; y < kN; ++y) {
    for (int x = 0; x < kN; ++x) {
      const std::size_t p = static_cast<std::size_t>(x + kN * y);
      const double X = x * dx;
      const double Y = y * dx;
      re[p] = 0.15 + 0.04 * std::cos(2.0 * pfc::pi * X / kN + 0.2 * shift) +
              0.02 * std::sin(4.0 * pfc::pi * Y / kN);
      im[p] = 0.01 * std::sin(2.0 * pfc::pi * X / kN) * (1.0 + 0.1 * shift);
    }
  }
}

void constant_symbol(AmplitudeSpectralETD1 &sys, std::size_t j, double s_plus,
                     double s_minus) {
  std::vector<double> plus(sys.outbox_size(), s_plus);
  std::vector<double> minus(sys.outbox_size(), s_minus);
  sys.set_symbol(j, plus, minus);
}

/// APFC weight, written here and not taken from `amplitude_etd1_weight`.
double reference_phi(double symbol, double dt) {
  const double z = symbol * dt;
  const double e = std::exp(z);
  if (std::abs(z) < 1e-8) {
    return dt * (1.0 + 0.5 * z);
  }
  return (e - 1.0) / symbol;
}

/// Real mix of the two half-spectrum coefficients. Not the stepper's c1/c2 loop.
void mix_mode(const std::complex<double> &ha, const std::complex<double> &hb,
              const std::complex<double> &na, const std::complex<double> &nb,
              double e1, double p1, double e2, double p2, std::complex<double> &ore,
              std::complex<double> &oim) {
  const double a = 0.5 * (e1 + e2);
  const double b = 0.5 * (e1 - e2);
  const double c = 0.5 * (p1 + p2);
  const double d = 0.5 * (p1 - p2);
  const std::complex<double> i(0.0, 1.0);
  ore = a * ha + i * b * hb + c * na + i * d * nb;
  oim = a * hb - i * b * ha + c * nb - i * d * na;
}

void c1c2_mode(const std::complex<double> &ha, const std::complex<double> &hb,
               const std::complex<double> &na, const std::complex<double> &nb,
               double e1, double p1, double e2, double p2, std::complex<double> &ore,
               std::complex<double> &oim) {
  const std::complex<double> i(0.0, 1.0);
  const auto c1 = ha + i * hb;
  const auto c2 = ha - i * hb;
  const auto n1 = na + i * nb;
  const auto n2 = na - i * nb;
  const auto d1 = e1 * c1 + p1 * n1;
  const auto d2 = e2 * c2 + p2 * n2;
  ore = 0.5 * (d1 + d2);
  oim = (d1 - d2) / (2.0 * i);
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
  out << "AMPLITUDE_ETD1 " << name << " " << value << "\n";
  std::cout << out.str();
}

} // namespace

TEST_CASE("amplitude ETD1 decays each mode by its own symbol",
          "[unit][simulation][amplitude_etd1]") {
  auto fft = pfc::fft::create(pfc::decomposition::create(square(), 1));
  AmplitudeSpectralETD1 sys(fft, 2);
  fill_wave(sys.real(0), sys.imag(0), 0);
  fill_wave(sys.real(1), sys.imag(1), 1);
  const std::vector<double> re0(sys.real(0).begin(), sys.real(0).end());
  const std::vector<double> im0(sys.imag(0).begin(), sys.imag(0).end());
  const std::vector<double> re1(sys.real(1).begin(), sys.real(1).end());
  const std::vector<double> im1(sys.imag(1).begin(), sys.imag(1).end());

  constexpr double s0 = -1.25;
  constexpr double s1 = -0.5;
  constexpr double dt = 0.02;
  constant_symbol(sys, 0, s0, s0);
  constant_symbol(sys, 1, s1, s1);
  sys.set_dt(dt);
  const double t1 = sys.step(0.0, zero_nonlinear);
  REQUIRE(t1 == dt);

  const double f0 = std::exp(s0 * dt);
  const double f1 = std::exp(s1 * dt);
  std::vector<double> want_re0 = re0;
  std::vector<double> want_im0 = im0;
  std::vector<double> want_re1 = re1;
  std::vector<double> want_im1 = im1;
  for (double &v : want_re0) v *= f0;
  for (double &v : want_im0) v *= f0;
  for (double &v : want_re1) v *= f1;
  for (double &v : want_im1) v *= f1;
  const double err0 =
      std::max(max_abs(want_re0, sys.real(0)), max_abs(want_im0, sys.imag(0)));
  const double err1 =
      std::max(max_abs(want_re1, sys.real(1)), max_abs(want_im1, sys.imag(1)));
  print_line("even_decay_0", err0);
  print_line("even_decay_1", err1);
  REQUIRE(err0 < 1e-12);
  REQUIRE(err1 < 1e-12);
  // The second amplitude did not take the first symbol.
  REQUIRE(max_abs(want_re0, sys.real(1)) > 1e-4);

  // s = 0 uses the Taylor weight: a constant rate adds dt * N.
  AmplitudeSpectralETD1 force(fft, 1);
  std::fill(force.real(0).begin(), force.real(0).end(), 0.4);
  std::fill(force.imag(0).begin(), force.imag(0).end(), -0.1);
  constant_symbol(force, 0, 0.0, 0.0);
  constexpr double dt_force = 0.05;
  constexpr double rate_re = 0.3;
  constexpr double rate_im = -0.2;
  force.set_dt(dt_force);
  const double t_force =
      force.step(0.1, [=](const std::vector<std::span<const double>> &,
                          const std::vector<std::span<const double>> &,
                          const std::vector<std::span<double>> &nre,
                          const std::vector<std::span<double>> &nim) {
        std::fill(nre[0].begin(), nre[0].end(), rate_re);
        std::fill(nim[0].begin(), nim[0].end(), rate_im);
      });
  REQUIRE(t_force == 0.1 + dt_force);
  double force_err = 0.0;
  for (double v : force.real(0)) {
    force_err = std::max(force_err, std::abs(v - (0.4 + dt_force * rate_re)));
  }
  for (double v : force.imag(0)) {
    force_err = std::max(force_err, std::abs(v - (-0.1 + dt_force * rate_im)));
  }
  print_line("taylor_force", force_err);
  REQUIRE(force_err < 1e-12);
}

TEST_CASE("amplitude ETD1 matches the APFC reference on a manufactured step",
          "[unit][simulation][amplitude_etd1]") {
  auto fft = pfc::fft::create(pfc::decomposition::create(square(), 1));
  constexpr double psibar = -0.15;
  constexpr double t_coef = -3.0 * psibar;
  constexpr double dt = 0.01;
  constexpr double dB = -0.1 + 3.0 * psibar * psibar;
  ApfcRates rates{t_coef};

  AmplitudeSpectralETD1 sys(fft, 3);
  for (std::size_t j = 0; j < 3; ++j) {
    fill_wave(sys.real(j), sys.imag(j), static_cast<int>(j));
  }
  std::vector<std::vector<double>> sym_q(3);
  std::vector<std::vector<double>> sym_m(3);
  const std::size_t m = sys.outbox_size();
  for (std::size_t j = 0; j < 3; ++j) {
    sym_q[j].resize(m);
    sym_m[j].resize(m);
    for (std::size_t q = 0; q < m; ++q) {
      const double q2 = 0.15 * static_cast<double>(q);
      sym_q[j][q] = -(dB + (q2 + 0.2 * static_cast<double>(j + 1)) *
                               (q2 + 0.2 * static_cast<double>(j + 1)));
      sym_m[j][q] = -(dB + (q2 - 0.2 * static_cast<double>(j + 1)) *
                               (q2 - 0.2 * static_cast<double>(j + 1)));
    }
    sys.set_symbol(j, sym_q[j], sym_m[j]);
  }
  sys.set_dt(dt);

  std::vector<std::vector<double>> re0(3);
  std::vector<std::vector<double>> im0(3);
  std::vector<std::vector<double>> nre(3, std::vector<double>(sys.inbox_size()));
  std::vector<std::vector<double>> nim(3, std::vector<double>(sys.inbox_size()));
  std::vector<std::span<const double>> re_view;
  std::vector<std::span<const double>> im_view;
  std::vector<std::span<double>> nre_view;
  std::vector<std::span<double>> nim_view;
  for (std::size_t j = 0; j < 3; ++j) {
    re0[j].assign(sys.real(j).begin(), sys.real(j).end());
    im0[j].assign(sys.imag(j).begin(), sys.imag(j).end());
    re_view.emplace_back(re0[j]);
    im_view.emplace_back(im0[j]);
    nre_view.emplace_back(nre[j]);
    nim_view.emplace_back(nim[j]);
  }
  rates(re_view, im_view, nre_view, nim_view);

  double oracle_gap = 0.0;
  double step_gap = 0.0;
  std::vector<std::vector<double>> expect_re(3);
  std::vector<std::vector<double>> expect_im(3);
  for (std::size_t j = 0; j < 3; ++j) {
    std::vector<std::complex<double>> ha(m), hb(m), na(m), nb(m);
    fft.forward(re0[j], ha);
    fft.forward(im0[j], hb);
    fft.forward(nre[j], na);
    fft.forward(nim[j], nb);
    std::vector<std::complex<double>> mix_re(m), mix_im(m), ref_re(m), ref_im(m);
    for (std::size_t q = 0; q < m; ++q) {
      const double e1 = std::exp(sym_q[j][q] * dt);
      const double e2 = std::exp(sym_m[j][q] * dt);
      const double p1 = reference_phi(sym_q[j][q], dt);
      const double p2 = reference_phi(sym_m[j][q], dt);
      mix_mode(ha[q], hb[q], na[q], nb[q], e1, p1, e2, p2, mix_re[q], mix_im[q]);
      c1c2_mode(ha[q], hb[q], na[q], nb[q], e1, p1, e2, p2, ref_re[q], ref_im[q]);
      oracle_gap = std::max(oracle_gap, std::abs(mix_re[q] - ref_re[q]));
      oracle_gap = std::max(oracle_gap, std::abs(mix_im[q] - ref_im[q]));
    }
    expect_re[j].resize(sys.inbox_size());
    expect_im[j].resize(sys.inbox_size());
    fft.backward(mix_re, expect_re[j]);
    fft.backward(mix_im, expect_im[j]);
  }
  REQUIRE(oracle_gap < 1e-12);

  const double t1 = sys.step(0.0, rates);
  REQUIRE(t1 == dt);
  for (std::size_t j = 0; j < 3; ++j) {
    step_gap = std::max(step_gap, max_abs(expect_re[j], sys.real(j)));
    step_gap = std::max(step_gap, max_abs(expect_im[j], sys.imag(j)));
  }
  print_line("oracle_gap", oracle_gap);
  print_line("apfc_step", step_gap);
  REQUIRE(step_gap < 1e-12);
  // The nonlinear step is not a pure rescaling of the initial wave.
  REQUIRE(max_abs(re0[0], sys.real(0)) > 1e-6);
}

TEST_CASE("amplitude ETD1 rejects a bad step without writing the fields",
          "[unit][simulation][amplitude_etd1]") {
  auto fft = pfc::fft::create(pfc::decomposition::create(square(), 1));
  REQUIRE_THROWS_AS(AmplitudeSpectralETD1(fft, 0), std::invalid_argument);

  AmplitudeSpectralETD1 sys(fft, 1);
  std::fill(sys.real(0).begin(), sys.real(0).end(), 0.25);
  std::fill(sys.imag(0).begin(), sys.imag(0).end(), -0.05);
  const std::vector<double> re0(sys.real(0).begin(), sys.real(0).end());
  const std::vector<double> im0(sys.imag(0).begin(), sys.imag(0).end());
  constant_symbol(sys, 0, -0.3, -0.4);
  REQUIRE_THROWS_AS(sys.set_dt(0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(sys.set_dt(-1.0), std::invalid_argument);
  sys.set_dt(0.01);
  std::vector<double> short_symbol(sys.outbox_size() - 1, -0.2);
  REQUIRE_THROWS_AS(sys.set_symbol(0, short_symbol, short_symbol),
                    std::invalid_argument);
  REQUIRE(max_abs(re0, sys.real(0)) == 0.0);

  REQUIRE_THROWS_AS(sys.step(0.0,
                             [](const std::vector<std::span<const double>> &,
                                const std::vector<std::span<const double>> &,
                                const std::vector<std::span<double>> &nre,
                                const std::vector<std::span<double>> &) {
                               nre[0][0] = std::numeric_limits<double>::infinity();
                             }),
                    std::domain_error);
  REQUIRE(max_abs(re0, sys.real(0)) == 0.0);
  REQUIRE(max_abs(im0, sys.imag(0)) == 0.0);
  REQUIRE_THROWS_AS(sys.real(2), std::out_of_range);
}

TEST_CASE("failed amplitude ETD1 timestep change preserves all weights",
          "[unit][simulation][amplitude_etd1]") {
  auto fft = pfc::fft::create(pfc::decomposition::create(square(), 1));
  AmplitudeSpectralETD1 sys(fft, 2);
  for (std::size_t j = 0; j < 2; ++j)
    fill_wave(sys.real(j), sys.imag(j), static_cast<int>(j));
  const std::vector<double> original0(sys.real(0).begin(), sys.real(0).end());
  const std::vector<double> original1(sys.real(1).begin(), sys.real(1).end());
  constexpr double old_dt = 1.0e-4;
  constant_symbol(sys, 0, -1.0, -1.0);
  constant_symbol(sys, 1, 1000.0, 1000.0);
  sys.set_dt(old_dt);
  REQUIRE_THROWS_AS(sys.set_dt(1.0), std::domain_error);
  REQUIRE(sys.dt() == old_dt);
  REQUIRE(sys.step(0.0, zero_nonlinear) == old_dt);
  std::vector<double> expected0 = original0;
  std::vector<double> expected1 = original1;
  for (double &v : expected0) v *= std::exp(-old_dt);
  for (double &v : expected1) v *= std::exp(1000.0 * old_dt);
  REQUIRE(max_abs(expected0, sys.real(0)) < 1e-12);
  REQUIRE(max_abs(expected1, sys.real(1)) < 1e-12);
}

TEST_CASE("amplitude ETD1 phi weight is accurate at the Taylor threshold",
          "[unit][simulation][amplitude_etd1]") {
  // The independent integral/Taylor oracle is accurate to O(z^4) here.
  // Subtracting exp(z)-1 loses significant digits immediately above 1e-8.
  const double dt = 0.5;
  for (double z : {-1.01e-8, -0.99e-8, 0.0, 0.99e-8, 1.01e-8}) {
    const double symbol = z / dt;
    const auto weight = pfc::sim::amplitude_etd1_weight(symbol, dt);
    const double expected = dt * (1.0 + 0.5 * z + z * z / 6.0 + z * z * z / 24.0);
    REQUIRE(std::abs(weight.phi1 - expected) < 2.0e-15);
    REQUIRE(std::isfinite(weight.exp_s));
    REQUIRE(std::isfinite(weight.phi1));
  }
}
