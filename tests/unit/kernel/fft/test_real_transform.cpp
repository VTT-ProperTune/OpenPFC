// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <complex>
#include <numbers>
#include <openpfc/kernel/fft/real_transform.hpp>

using namespace pfc::fft;
using Catch::Matchers::WithinAbs;
namespace {
BoundedAxis axis(RealTransformType t, int n) {
  return {t,
          t == RealTransformType::DCT1   ? UniformGrid::Endpoints
          : t == RealTransformType::DST1 ? UniformGrid::Interior
                                         : UniformGrid::CellCentered,
          n, 2.7, -0.3};
}
constexpr RealTransformType types[] = {
    RealTransformType::DCT1, RealTransformType::DCT2, RealTransformType::DST1,
    RealTransformType::DST2};
} // namespace

TEST_CASE("uniform real transforms match independently summed mirrored DFT",
          "[real_transform]") {
  for (auto t : types)
    for (int n : {1, 2, 3, 4, 5, 6, 9}) {
      if (t == RealTransformType::DCT1 && n == 1) continue;
      auto a = axis(t, n);
      RealTransform plan(a);
      std::vector<double> f(n), c(n), back(n), mirror(mirrored_size(a));
      for (int j = 0; j < n; ++j) f[j] = std::sin(0.31 * (j + 1)) + 0.2 * j;
      auto saved = f;
      for (int j = 0; j < n; ++j) {
        int p = a.grid == UniformGrid::Interior ? j + 1 : j;
        int q = a.grid == UniformGrid::CellCentered
                    ? int(mirror.size()) - 1 - j
                    : (int(mirror.size()) - p) % int(mirror.size());
        mirror[p] = f[j];
        if (p != q) mirror[q] = (is_cosine(t) ? 1 : -1) * f[j];
      }
      plan.forward(f, c);
      auto saved_c = c;
      plan.backward(c, back);
      REQUIRE(f == saved);
      REQUIRE(c == saved_c);
      for (int j = 0; j < n; ++j) {
        REQUIRE_THAT(back[j], WithinAbs(f[j], 2e-12));
        REQUIRE_THAT(plan.evaluate(c, coordinate(a, j)), WithinAbs(f[j], 2e-12));
      }
      for (int mode = 0; mode < n; ++mode) {
        int m = mode + (is_cosine(t) ? 0 : 1);
        std::complex<double> F = 0;
        for (std::size_t j = 0; j < mirror.size(); ++j)
          F += mirror[j] * std::exp(std::complex<double>(
                               0, -2 * std::numbers::pi * j * m / mirror.size()));
        if (a.grid == UniformGrid::CellCentered)
          F *= std::exp(
              std::complex<double>(0, -std::numbers::pi * m / mirror.size()));
        REQUIRE_THAT(c[mode], WithinAbs(is_cosine(t) ? F.real() : -F.imag(), 2e-11));
      }
      // Aliased callers are safe because execution uses owned scratch.
      plan.forward(f, f);
      plan.backward(f, f);
      for (int j = 0; j < n; ++j) REQUIRE_THAT(f[j], WithinAbs(saved[j], 2e-12));
    }
}

TEST_CASE("uniform sine cosine Poisson derivatives and boundary traces",
          "[real_transform]") {
  for (auto t : types)
    for (int n : {3, 4, 7, 8}) {
      auto a = axis(t, n);
      RealTransform plan(a);
      // Include highest mode to exercise endpoint weights and DST-II Nyquist.
      for (int mode : {1, n - 1}) {
        double k = wavenumber(a, mode);
        std::vector<double> u(n), f(n), back(n), lap(n), c(n);
        for (int j = 0; j < n; ++j) {
          double phase = k * (coordinate(a, j) - a.origin);
          u[j] = is_cosine(t) ? std::cos(phase) : std::sin(phase);
        }
        for (auto s :
             {LaplacianSymbol::Spectral, LaplacianSymbol::CenteredDifference}) {
          double lambda = eigenvalue(a, mode, s);
          for (int j = 0; j < n; ++j) f[j] = lambda * u[j];
          plan.poisson(f, back, s);
          plan.laplacian(u, lap, s);
          for (int j = 0; j < n; ++j) {
            REQUIRE_THAT(back[j], WithinAbs(u[j], 2e-12));
            REQUIRE_THAT(lap[j], WithinAbs(-f[j], 2e-10));
            // Independent FD stencil on the mirrored samples.
            if (s == LaplacianSymbol::CenteredDifference) {
              double x = coordinate(a, j) - a.origin, h = spacing(a);
              auto sample = [&](double p) {
                return is_cosine(t) ? std::cos(k * p) : std::sin(k * p);
              };
              double fd = (sample(x - h) - 2 * sample(x) + sample(x + h)) / (h * h);
              REQUIRE_THAT(lap[j], WithinAbs(fd, 2e-11));
            }
          }
        }
        plan.forward(u, c);
        for (double x :
             {a.origin, a.origin + 0.237 * a.length, a.origin + a.length}) {
          double phase = k * (x - a.origin);
          double v = is_cosine(t) ? std::cos(phase) : std::sin(phase);
          double d = k * (is_cosine(t) ? -std::sin(phase) : std::cos(phase));
          REQUIRE_THAT(plan.evaluate(c, x), WithinAbs(v, 2e-12));
          REQUIRE_THAT(plan.evaluate(c, x, 1), WithinAbs(d, 2e-11));
          REQUIRE_THAT(plan.evaluate(c, x, 2), WithinAbs(-k * k * v, 2e-10));
        }
        int boundary_order = is_cosine(t) ? 1 : 0;
        REQUIRE_THAT(plan.evaluate(c, a.origin, boundary_order),
                     WithinAbs(0, 2e-11));
        REQUIRE_THAT(plan.evaluate(c, a.origin + a.length, boundary_order),
                     WithinAbs(0, 2e-11));
      }
    }
}

TEST_CASE("Neumann compatibility and explicit grid validation", "[real_transform]") {
  for (auto t : {RealTransformType::DCT1, RealTransformType::DCT2}) {
    auto a = axis(t, 5);
    RealTransform plan(a);
    std::vector<double> f(5, 1), u(5), c(5);
    REQUIRE_THROWS_AS(plan.poisson(f, u), std::invalid_argument);
    std::fill(f.begin(), f.end(), 0);
    plan.poisson(f, u);
    REQUIRE(u == f);
    // Type I has trapezoidal endpoint weights, not arithmetic-mean weights.
    f = {1, -0.5, 0, -0.5, 1};
    if (t == RealTransformType::DCT1) {
      plan.poisson(f, u);
      plan.forward(u, c);
      REQUIRE_THAT(c[0], WithinAbs(0, 1e-13));
    } else
      REQUIRE_THROWS_AS(plan.poisson(f, u), std::invalid_argument);
  }
  REQUIRE_THROWS_AS(RealTransform(axis(RealTransformType::DCT1, 1)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(RealTransform(BoundedAxis{RealTransformType::DST2,
                                              UniformGrid::Endpoints, 5, 1}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(RealTransform(BoundedAxis{RealTransformType::DST1,
                                              UniformGrid::Interior, 5, -1}),
                    std::invalid_argument);
  RealTransform plan(axis(RealTransformType::DST1, 3));
  std::vector<double> short_buffer(2), out(3);
  REQUIRE_THROWS_AS(plan.forward(short_buffer, out), std::invalid_argument);
}
