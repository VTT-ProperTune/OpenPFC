// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Coupled two-field ETD2: exact linear evolution with off-diagonal and
 * nearly degenerate symbols, an exact full-RHS split of a finite-difference
 * operator, second-order convergence on a nonlinear manufactured problem,
 * adaptive tolerance response, rejection and retry, end-time clipping and
 * the coefficient cache.
 */

#include <array>
#include <cmath>
#include <complex>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/constants.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>
#include <openpfc/kernel/simulation/adaptive_controller.hpp>
#include <openpfc/kernel/simulation/coupled_spectral_etd2.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using pfc::integrator::Matrix2x2;
using pfc::sim::AdaptiveControlConfig;
using pfc::sim::AdaptiveControlMode;
using pfc::sim::AdaptiveTimeController;
using pfc::sim::CoupledSpectralETD2;
using pfc::sim::CoupledStageOutput;

namespace {

constexpr double kLength = 2.0 * pfc::pi;

pfc::Domain square(int n) {
  const double dx = kLength / static_cast<double>(n);
  return pfc::domain::create(pfc::GridSize({n, n, 1}),
                             pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                             pfc::GridSpacing({dx, dx, 1.0}));
}

/// exp(tL) by long-double Taylor series with scaling and squaring.
/// Independent of `matrix_phi`: no eigenvalues and no φ functions.
std::array<long double, 4> exact_exp(const Matrix2x2 &l, double t) {
  long double a[4] = {
      l.m00 * static_cast<long double>(t), l.m01 * static_cast<long double>(t),
      l.m10 * static_cast<long double>(t), l.m11 * static_cast<long double>(t)};
  long double nrm =
      std::max(std::fabs(a[0]) + std::fabs(a[2]), std::fabs(a[1]) + std::fabs(a[3]));
  int s = 0;
  while (nrm > 0.125L) {
    nrm *= 0.5L;
    ++s;
  }
  for (auto &v : a) v = std::ldexp(v, -s);
  long double sum[4] = {1, 0, 0, 1};
  long double term[4] = {1, 0, 0, 1};
  for (int n = 1; n <= 30; ++n) {
    const long double t0 = (term[0] * a[0] + term[1] * a[2]) / n;
    const long double t1 = (term[0] * a[1] + term[1] * a[3]) / n;
    const long double t2 = (term[2] * a[0] + term[3] * a[2]) / n;
    const long double t3 = (term[2] * a[1] + term[3] * a[3]) / n;
    term[0] = t0, term[1] = t1, term[2] = t2, term[3] = t3;
    for (int i = 0; i < 4; ++i) sum[i] += term[i];
  }
  for (int i = 0; i < s; ++i) {
    const long double s0 = sum[0] * sum[0] + sum[1] * sum[2];
    const long double s1 = sum[0] * sum[1] + sum[1] * sum[3];
    const long double s2 = sum[2] * sum[0] + sum[3] * sum[2];
    const long double s3 = sum[2] * sum[1] + sum[3] * sum[3];
    sum[0] = s0, sum[1] = s1, sum[2] = s2, sum[3] = s3;
  }
  return {sum[0], sum[1], sum[2], sum[3]};
}

/// Five-point Laplacian symbol on spacing h.
double fd_laplacian_symbol(double kx, double ky, double h) {
  const double sx = std::sin(0.5 * kx * h);
  const double sy = std::sin(0.5 * ky * h);
  return -4.0 / (h * h) * (sx * sx + sy * sy);
}

/// Periodic five-point Laplacian in real space.
void fd_laplacian(std::span<const double> u, std::span<double> out, int n,
                  double h) {
  auto at = [&](int i, int j) {
    return u[static_cast<std::size_t>(((i + n) % n) + ((j + n) % n) * n)];
  };
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i + j * n)] =
          (at(i + 1, j) + at(i - 1, j) + at(i, j + 1) + at(i, j - 1) -
           4.0 * at(i, j)) /
          (h * h);
    }
  }
}

template <class F> void fill(std::span<double> u, int n, F &&f) {
  const double h = kLength / static_cast<double>(n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      u[static_cast<std::size_t>(i + j * n)] = f(i * h, j * h);
    }
  }
}

double max_abs_diff(std::span<const double> a, std::span<const double> b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

struct ZeroStage {
  void operator()(double, std::span<const double>, std::span<const double>,
                  std::span<double> f0, std::span<double> f1) const {
    std::fill(f0.begin(), f0.end(), 0.0);
    std::fill(f1.begin(), f1.end(), 0.0);
  }
};

/// Expected fields after time t: per mode exp(tL) Û(0), then inverse FFT.
std::array<std::vector<double>, 2> exact_linear(pfc::fft::IHostFFT &fft,
                                                std::span<const Matrix2x2> symbol,
                                                const std::vector<double> &u0,
                                                const std::vector<double> &u1,
                                                double t) {
  std::vector<std::complex<double>> h0(fft.size_outbox()), h1(fft.size_outbox());
  fft.forward(u0, h0);
  fft.forward(u1, h1);
  for (std::size_t i = 0; i < h0.size(); ++i) {
    const auto e = exact_exp(symbol[i], t);
    const std::complex<long double> a(h0[i].real(), h0[i].imag());
    const std::complex<long double> b(h1[i].real(), h1[i].imag());
    const auto r0 = e[0] * a + e[1] * b;
    const auto r1 = e[2] * a + e[3] * b;
    h0[i] = {static_cast<double>(r0.real()), static_cast<double>(r0.imag())};
    h1[i] = {static_cast<double>(r1.real()), static_cast<double>(r1.imag())};
  }
  std::array<std::vector<double>, 2> out{std::vector<double>(fft.size_inbox()),
                                         std::vector<double>(fft.size_inbox())};
  fft.backward(h0, out[0]);
  fft.backward(h1, out[1]);
  return out;
}

void several_modes(std::span<double> u0, std::span<double> u1, int n) {
  fill(u0, n, [](double x, double y) {
    return 0.3 + std::cos(x) + 0.5 * std::sin(2.0 * x + 3.0 * y) +
           0.2 * std::cos(5.0 * y);
  });
  fill(u1, n, [](double x, double y) {
    return -0.1 + 0.7 * std::sin(y) + 0.4 * std::cos(3.0 * x - y) +
           0.1 * std::sin(6.0 * x);
  });
}

// ---- nonlinear manufactured problem -------------------------------------
//   u_t = L u + N(u) + g(t, x),  L = [[ν0 Δ − 1, 1/2], [−1/2, ν1 Δ]]
//   N(u) = (−u0³ + u0 u1, −u1² / 2)
// with g chosen so that the exact solution is
//   u0 = 1/2 + 3/10 sin(x + t) cos y,  u1 = 1/5 cos x cos(y − 2t).

constexpr double kNu0 = 1.0;
constexpr double kNu1 = 0.5;

Matrix2x2 manufactured_symbol(double kx, double ky, double) {
  const double lap = -(kx * kx + ky * ky);
  return {kNu0 * lap - 1.0, 0.5, -0.5, kNu1 * lap};
}

struct Exact {
  double u0, u1, u0_t, u1_t, lap0, lap1;
};

Exact exact_at(double x, double y, double t) {
  const double s = std::sin(x + t), c = std::cos(x + t);
  const double cy = std::cos(y);
  const double a = std::cos(x), b = std::cos(y - 2.0 * t),
               bs = std::sin(y - 2.0 * t);
  return {0.5 + 0.3 * s * cy, 0.2 * a * b,         0.3 * c * cy,
          0.4 * a * bs,       -2.0 * 0.3 * s * cy, -2.0 * 0.2 * a * b};
}

/// Returns N(u) + g(t), the part of the RHS not in L.
struct ManufacturedStage {
  int n = 16;
  long calls = 0;
  void operator()(double t, std::span<const double> u0, std::span<const double> u1,
                  std::span<double> f0, std::span<double> f1) {
    ++calls;
    const double h = kLength / static_cast<double>(n);
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        const auto q = static_cast<std::size_t>(i + j * n);
        const Exact e = exact_at(i * h, j * h, t);
        const double lu0 = kNu0 * e.lap0 - e.u0 + 0.5 * e.u1;
        const double lu1 = -0.5 * e.u0 + kNu1 * e.lap1;
        const double g0 = e.u0_t - lu0 - (-e.u0 * e.u0 * e.u0 + e.u0 * e.u1);
        const double g1 = e.u1_t - lu1 - (-0.5 * e.u1 * e.u1);
        f0[q] = -u0[q] * u0[q] * u0[q] + u0[q] * u1[q] + g0;
        f1[q] = -0.5 * u1[q] * u1[q] + g1;
      }
    }
  }
};

double manufactured_error(std::span<const double> u0, std::span<const double> u1,
                          int n, double t) {
  const double h = kLength / static_cast<double>(n);
  double m = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const auto q = static_cast<std::size_t>(i + j * n);
      const Exact e = exact_at(i * h, j * h, t);
      m = std::max({m, std::abs(u0[q] - e.u0), std::abs(u1[q] - e.u1)});
    }
  }
  return m;
}

template <class Stage>
void load_manufactured(CoupledSpectralETD2<Stage> &sys, int n) {
  const double h = kLength / static_cast<double>(n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const Exact e = exact_at(i * h, j * h, 0.0);
      sys.field(0)[static_cast<std::size_t>(i + j * n)] = e.u0;
      sys.field(1)[static_cast<std::size_t>(i + j * n)] = e.u1;
    }
  }
  sys.load_state();
}

AdaptiveControlConfig adaptive_config(double tol) {
  AdaptiveControlConfig cfg;
  cfg.mode = AdaptiveControlMode::adaptive;
  cfg.atol = tol;
  cfg.rtol = tol;
  cfg.min_dt = 1e-8;
  cfg.max_dt = 1.0;
  cfg.max_sequential_rejections = 20;
  return cfg;
}

} // namespace

TEST_CASE("coupled ETD2 is exact for coupled linear evolution of several modes",
          "[coupled_etd2]") {
  constexpr int n = 16;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));

  struct Case {
    const char *name;
    Matrix2x2 (*symbol)(double, double, double);
  };
  const Case cases[] = {
      {"off-diagonal coupling",
       [](double kx, double ky, double) {
         const double lap = -(kx * kx + ky * ky);
         return Matrix2x2{lap - 1.0, 0.8, -1.5, 0.5 * lap};
       }},
      {"complex pair",
       [](double kx, double ky, double) {
         const double lap = -(kx * kx + ky * ky);
         return Matrix2x2{0.3 * lap, 4.0, -4.0, 0.3 * lap};
       }},
      {"nearly degenerate, defective",
       [](double kx, double ky, double) {
         const double lap = -(kx * kx + ky * ky);
         return Matrix2x2{lap, 2.0, 0.0, lap * (1.0 + 1e-9)};
       }},
      {"zero symbol", [](double, double, double) { return Matrix2x2{}; }},
  };
  for (const auto &c : cases) {
    INFO(c.name);
    auto symbol = pfc::sim::coupled_linear_symbol(fft, fft.get_outbox_bounds(),
                                                  domain, c.symbol);
    CoupledSpectralETD2 sys(fft, symbol, ZeroStage{}, CoupledStageOutput::nonlinear);
    several_modes(sys.field(0), sys.field(1), n);
    const std::vector<double> u0(sys.field(0).begin(), sys.field(0).end());
    const std::vector<double> u1(sys.field(1).begin(), sys.field(1).end());
    sys.load_state();
    double t = 0.0;
    for (int s = 0; s < 7; ++s) t = sys.step(t, 0.03);
    const auto want = exact_linear(fft, sys.linear_symbol(), u0, u1, t);
    CHECK(max_abs_diff(sys.field(0), want[0]) < 1e-12);
    CHECK(max_abs_diff(sys.field(1), want[1]) < 1e-12);
    // The predictor and the corrector agree when N vanishes.
    CHECK(max_abs_diff(sys.embedded_error(),
                       std::vector<double>(2 * sys.size(), 0.0)) < 1e-13);
  }
}

TEST_CASE("coupled ETD2 full-RHS split is exact for a finite-difference operator",
          "[coupled_etd2]") {
  // F = L_h U with L_h the five-point Laplacian plus a latent-heat-like
  // coupling. The stage returns F in real space; with the matching discrete
  // symbol the system's N̂ = F̂ − L Û vanishes and the step is exact.
  constexpr int n = 24;
  const double h = kLength / n;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  auto symbol = pfc::sim::coupled_linear_symbol(
      fft, fft.get_outbox_bounds(), domain, [h](double kx, double ky, double) {
        const double lap = fd_laplacian_symbol(kx, ky, h);
        return Matrix2x2{lap - 2.0, 0.0, 0.5 * (lap - 2.0), 2.0 * lap};
      });
  std::vector<double> lap0(n * n), lap1(n * n);
  auto stage = [&](double, std::span<const double> u0, std::span<const double> u1,
                   std::span<double> f0, std::span<double> f1) {
    fd_laplacian(u0, lap0, n, h);
    fd_laplacian(u1, lap1, n, h);
    for (std::size_t q = 0; q < f0.size(); ++q) {
      f0[q] = lap0[q] - 2.0 * u0[q];
      f1[q] = 0.5 * f0[q] + 2.0 * lap1[q];
    }
  };
  CoupledSpectralETD2 sys(fft, symbol, stage, CoupledStageOutput::full_rhs);
  several_modes(sys.field(0), sys.field(1), n);
  const std::vector<double> u0(sys.field(0).begin(), sys.field(0).end());
  const std::vector<double> u1(sys.field(1).begin(), sys.field(1).end());
  sys.load_state();
  double t = 0.0;
  // dt far above the explicit limit h²/8 of the temperature diffusion.
  for (int s = 0; s < 5; ++s) t = sys.step(t, 0.2);
  const auto want = exact_linear(fft, sys.linear_symbol(), u0, u1, t);
  CHECK(max_abs_diff(sys.field(0), want[0]) < 1e-11);
  CHECK(max_abs_diff(sys.field(1), want[1]) < 1e-11);
}

TEST_CASE(
    "coupled ETD2 converges at second order on a nonlinear manufactured problem",
    "[coupled_etd2]") {
  constexpr int n = 16;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  const auto symbol = pfc::sim::coupled_linear_symbol(fft, fft.get_outbox_bounds(),
                                                      domain, manufactured_symbol);
  const double t_end = 1.0;
  std::vector<double> errors;
  for (const int steps : {10, 20, 40, 80}) {
    CoupledSpectralETD2 sys(fft, symbol, ManufacturedStage{n},
                            CoupledStageOutput::nonlinear);
    load_manufactured(sys, n);
    const double dt = t_end / steps;
    double t = 0.0;
    for (int s = 0; s < steps; ++s) t = sys.step(t, dt);
    REQUIRE(std::abs(t - t_end) < 1e-12);
    errors.push_back(manufactured_error(sys.field(0), sys.field(1), n, t_end));
    // Two stage evaluations and eight transforms per step.
    CHECK(sys.statistics().stage_evaluations == 2 * steps);
    CHECK(sys.statistics().forward_transforms == 2 + 4 * steps);
    CHECK(sys.statistics().backward_transforms == 4 * steps);
    CHECK(sys.statistics().coefficient_builds == 1);
  }
  for (std::size_t i = 1; i < errors.size(); ++i) {
    const double order = std::log2(errors[i - 1] / errors[i]);
    INFO("dt halving " << i << ": errors " << errors[i - 1] << " -> " << errors[i]
                       << ", observed order " << order);
    CHECK(order > 1.85);
    CHECK(order < 2.3);
  }
}

TEST_CASE("adaptive coupled ETD2 error follows the tolerance", "[coupled_etd2]") {
  constexpr int n = 16;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  const auto symbol = pfc::sim::coupled_linear_symbol(fft, fft.get_outbox_bounds(),
                                                      domain, manufactured_symbol);
  const double t_end = 1.0;
  std::vector<double> errors;
  std::vector<int> accepted;
  for (const double tol : {1e-3, 1e-4, 1e-5, 1e-6}) {
    CoupledSpectralETD2 sys(fft, symbol, ManufacturedStage{n},
                            CoupledStageOutput::nonlinear);
    load_manufactured(sys, n);
    pfc::Time time({0.0, t_end, 0.01}, t_end);
    AdaptiveTimeController controller(adaptive_config(tol), 2);
    const auto summary = pfc::sim::run_coupled_etd2_adaptive(sys, time, controller);
    REQUIRE(std::abs(time.get_current() - t_end) < 1e-14);
    REQUIRE(summary.accepted == controller.accepted_count());
    REQUIRE(summary.rejected == controller.rejected_count());
    REQUIRE(sys.statistics().attempts == summary.accepted + summary.rejected);
    REQUIRE(summary.dt_min <= summary.dt_mean());
    REQUIRE(summary.dt_mean() <= summary.dt_max);
    errors.push_back(manufactured_error(sys.field(0), sys.field(1), n, t_end));
    accepted.push_back(summary.accepted);
    INFO("tol " << tol << " error " << errors.back() << " accepted "
                << summary.accepted << " rejected " << summary.rejected);
    CHECK(errors.back() < 50.0 * tol);
  }
  for (std::size_t i = 1; i < errors.size(); ++i) {
    CHECK(errors[i] < 0.6 * errors[i - 1]);
    CHECK(accepted[i] > accepted[i - 1]);
  }
}

TEST_CASE(
    "coupled ETD2 rejection keeps the accepted state and the retry reuses N(U_n)",
    "[coupled_etd2]") {
  constexpr int n = 16;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  const auto symbol = pfc::sim::coupled_linear_symbol(fft, fft.get_outbox_bounds(),
                                                      domain, manufactured_symbol);
  CoupledSpectralETD2 sys(fft, symbol, ManufacturedStage{n},
                          CoupledStageOutput::nonlinear);
  load_manufactured(sys, n);
  const std::vector<double> saved0(sys.field(0).begin(), sys.field(0).end());
  const std::vector<double> saved1(sys.field(1).begin(), sys.field(1).end());
  const std::vector<std::complex<double>> saved_hat(sys.field_hat(0).begin(),
                                                    sys.field_hat(0).end());

  pfc::Time time({0.0, 1.0, 0.8}, 1.0);
  AdaptiveTimeController controller(adaptive_config(1e-6), 2);

  time.begin_attempt(time.get_dt());
  sys.attempt(time.get_accepted_time(), time.get_attempted_dt());
  auto d = controller.decide_from_embedded_error(
      time.get_attempted_dt(), sys.embedded_error(), sys.attempt_start(),
      sys.attempt_candidate(), MPI_COMM_SELF);
  REQUIRE_FALSE(d.accepted);
  sys.reject();
  controller.apply(time, d);
  CHECK(time.get_current() == 0.0);
  CHECK(time.get_dt() < 0.8);
  CHECK(max_abs_diff(sys.field(0), saved0) == 0.0);
  CHECK(max_abs_diff(sys.field(1), saved1) == 0.0);
  for (std::size_t i = 0; i < saved_hat.size(); ++i) {
    REQUIRE(sys.field_hat(0)[i] == saved_hat[i]);
  }
  CHECK(sys.stage().calls == 2);

  // Retry with the controller's dt: one new stage evaluation, then accept.
  bool accepted = false;
  for (int tries = 0; tries < 10 && !accepted; ++tries) {
    time.begin_attempt(time.get_dt());
    const long before = sys.stage().calls;
    sys.attempt(time.get_accepted_time(), time.get_attempted_dt());
    CHECK(sys.stage().calls == before + 1);
    d = controller.decide_from_embedded_error(
        time.get_attempted_dt(), sys.embedded_error(), sys.attempt_start(),
        sys.attempt_candidate(), MPI_COMM_SELF);
    accepted = d.accepted;
    if (accepted) {
      sys.commit();
    } else {
      sys.reject();
    }
    controller.apply(time, d);
  }
  REQUIRE(accepted);
  CHECK(time.get_current() > 0.0);
  CHECK(controller.rejected_count() >= 1);
  CHECK(max_abs_diff(sys.field(0), saved0) > 0.0);

  // A retry with the same dt as an accepted attempt from another state is
  // not served from the cache.
  const long before = sys.stage().calls;
  sys.attempt(time.get_current(), 0.01);
  CHECK(sys.stage().calls == before + 2);
  sys.reject();
}

TEST_CASE(
    "coupled ETD2 clips the last step to the end time and rebuilds coefficients",
    "[coupled_etd2]") {
  constexpr int n = 16;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  auto symbol = pfc::sim::coupled_linear_symbol(fft, fft.get_outbox_bounds(), domain,
                                                manufactured_symbol);
  CoupledSpectralETD2 sys(fft, symbol, ManufacturedStage{n},
                          CoupledStageOutput::nonlinear);
  load_manufactured(sys, n);

  // Fixed control: every attempt accepted; the last one is clipped.
  AdaptiveControlConfig fixed;
  AdaptiveTimeController controller(fixed, 2);
  pfc::Time time({0.0, 0.37, 0.1}, 0.37);
  const auto summary = pfc::sim::run_coupled_etd2_adaptive(sys, time, controller);
  CHECK(summary.accepted == 4);
  CHECK(summary.rejected == 0);
  CHECK(std::abs(time.get_current() - 0.37) < 1e-15);
  CHECK(std::abs(summary.dt_min - 0.07) < 1e-12);
  CHECK(summary.dt_max == 0.1);
  CHECK(std::abs(sys.coefficient_dt() - 0.07) < 1e-12);
  CHECK(sys.statistics().coefficient_builds == 2);

  // The stored coefficients are those of the clipped step.
  const auto c = sys.coefficients();
  const auto l = sys.linear_symbol();
  for (std::size_t i = 0; i < c.size(); i += 17) {
    const auto want = pfc::integrator::etd2_coefficients(l[i], sys.coefficient_dt());
    REQUIRE(c[i].exp_hL.m00 == want.exp_hL.m00);
    REQUIRE(c[i].h_phi2.m10 == want.h_phi2.m10);
  }

  // A new symbol invalidates the coefficients; the next attempt rebuilds.
  for (auto &m : symbol) m.m01 = 0.0;
  sys.set_linear_symbol(symbol);
  CHECK(sys.coefficient_dt() == 0.0);
  sys.attempt(time.get_current(), 0.07);
  sys.reject();
  CHECK(sys.statistics().coefficient_builds == 3);
  CHECK(sys.coefficients()[1].exp_hL.m01 == 0.0);

  // The same step on a fresh system gives the same candidate. The accepted
  // transform of `sys` came from a commit, not from a forward FFT, so the
  // two agree to roundoff.
  CoupledSpectralETD2 fresh(fft, symbol, ManufacturedStage{n},
                            CoupledStageOutput::nonlinear);
  std::copy(sys.field(0).begin(), sys.field(0).end(), fresh.field(0).begin());
  std::copy(sys.field(1).begin(), sys.field(1).end(), fresh.field(1).begin());
  fresh.load_state();
  sys.attempt(0.5, 0.05);
  fresh.attempt(0.5, 0.05);
  CHECK(max_abs_diff(sys.candidate(0), fresh.candidate(0)) < 1e-14);
  CHECK(max_abs_diff(sys.candidate(1), fresh.candidate(1)) < 1e-14);
}

TEST_CASE("coupled ETD2 enforces the attempt protocol", "[coupled_etd2]") {
  constexpr int n = 8;
  const auto domain = square(n);
  auto fft = pfc::fft::create(pfc::decomposition::create(domain, 1));
  const std::vector<Matrix2x2> symbol(fft.size_outbox());
  CoupledSpectralETD2 sys(fft, symbol, ZeroStage{}, CoupledStageOutput::nonlinear);
  CHECK_THROWS_AS(sys.commit(), std::logic_error);
  CHECK_THROWS_AS(sys.reject(), std::logic_error);
  CHECK_THROWS_AS(sys.attempt(0.0, 0.0), std::invalid_argument);
  sys.attempt(0.0, 0.1);
  CHECK_THROWS_AS(sys.attempt(0.0, 0.1), std::logic_error);
  CHECK_THROWS_AS(sys.load_state(), std::logic_error);
  sys.commit();
  CHECK_THROWS_AS(sys.set_linear_symbol(std::vector<Matrix2x2>(3)),
                  std::invalid_argument);
  CHECK_THROWS_AS(sys.field(2), std::out_of_range);
}
