// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <openpfc/solvers/scalar_root.hpp>
using namespace pfc::solvers;

namespace {
double logistic(double y) {
  return y >= 0 ? 1 / (1 + std::exp(-y)) : std::exp(y) / (1 + std::exp(y));
}
} // namespace

TEST_CASE("scalar_root finds polynomial and transcendental roots",
          "[solvers][scalar_root]") {
  auto sq = [](double x) { return ScalarValue{x * x - 2, 2 * x}; };
  const auto r = scalar_root(sq, 0.0, 2.0, 1.0);
  REQUIRE(r.converged());
  REQUIRE(r.x == Catch::Approx(std::sqrt(2.0)).epsilon(1e-15));
  REQUIRE(r.iterations < 12);

  // Decreasing function: only the sign change is required.
  auto cosx = [](double x) {
    return ScalarValue{std::cos(x) - x, -std::sin(x) - 1};
  };
  const auto c = scalar_root(cosx, 0.0, 1.0, 0.0);
  REQUIRE(c.converged());
  REQUIRE(std::abs(std::cos(c.x) - c.x) <
          4 * std::numeric_limits<double>::epsilon());
}

TEST_CASE("scalar_root inverts a saturated logistic mixture",
          "[solvers][scalar_root]") {
  // h s(y - a) + (1 - h) s(y - b) = c: the two-phase inversion of an ideal
  // solution, with c within 1e-9 of 0 and 1 where f' is tiny.
  const double a = 1.0, b = -3.0;
  for (double h : {0.0, 0.3, 0.9, 1.0})
    for (double c : {1e-9, 0.2, 0.7, 1 - 1e-9}) {
      auto fun = [&](double y) {
        const double s1 = logistic(y - a), s2 = logistic(y - b);
        return ScalarValue{h * s1 + (1 - h) * s2 - c,
                           h * s1 * (1 - s1) + (1 - h) * s2 * (1 - s2)};
      };
      const double lc = std::log(c) - std::log1p(-c);
      const auto r =
          scalar_root(fun, std::min(a, b) + lc, std::max(a, b) + lc, 0.0,
                      {.f_tol = 4 * std::numeric_limits<double>::epsilon() * c});
      REQUIRE(r.converged());
      REQUIRE(std::abs(r.f) <=
              1e-13 * c + 4 * std::numeric_limits<double>::epsilon());
      REQUIRE(r.lo <= r.x);
      REQUIRE(r.x <= r.hi);
      REQUIRE(r.iterations <= 60);
    }
}

TEST_CASE("scalar_root reports a missing bracket, non-finite values and the "
          "iteration bound",
          "[solvers][scalar_root]") {
  auto pos = [](double x) { return ScalarValue{x * x + 1, 2 * x}; };
  REQUIRE(scalar_root(pos, -1.0, 1.0, 0.0).status == ScalarRootStatus::NoBracket);
  REQUIRE(scalar_root(pos, 1.0, -1.0, 0.0).status == ScalarRootStatus::NoBracket);

  auto bad = [](double x) {
    return ScalarValue{
        x > 0.25 && x < 0.75 ? std::numeric_limits<double>::quiet_NaN() : x - 0.6,
        1.0};
  };
  REQUIRE(scalar_root(bad, 0.0, 1.0, 0.5).status == ScalarRootStatus::NonFinite);

  // f' = 0 everywhere: pure bisection; the bound counts both bracket ends.
  int calls = 0;
  auto flat = [&](double x) {
    ++calls;
    return ScalarValue{x - 1.0 / 3.0, 0.0};
  };
  const auto r = scalar_root(flat, 0.0, 1.0, 0.5,
                             {.f_tol = 0, .x_tol = 0, .max_iterations = 10});
  REQUIRE(r.status == ScalarRootStatus::IterationLimit);
  REQUIRE(r.iterations == 10);
  REQUIRE(calls == 10);
  REQUIRE(r.hi - r.lo <= 1.0 / 128);

  auto root_at_end = [](double x) { return ScalarValue{x - 2.0, 1.0}; };
  const auto e = scalar_root(root_at_end, 0.0, 2.0, 1.0);
  REQUIRE(e.converged());
  REQUIRE(e.x == 2.0);
}
