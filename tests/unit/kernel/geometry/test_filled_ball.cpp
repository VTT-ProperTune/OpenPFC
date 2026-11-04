// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/geometry/filled_ball.hpp>

#include <cmath>
#include <complex>
#include <numbers>

using Catch::Matchers::WithinAbs;
using pfc::geometry::add_ball_harmonic;
using pfc::geometry::affine_map_from_columns;
using pfc::geometry::ball_boundary_flux;
using pfc::geometry::ball_contains;
using pfc::geometry::ball_embedding;
using pfc::geometry::ball_euler;
using pfc::geometry::ball_gradient;
using pfc::geometry::ball_has_boundary;
using pfc::geometry::ball_laplacian;
using pfc::geometry::ball_physical_laplacian;
using pfc::geometry::ball_physical_point;
using pfc::geometry::ball_physical_volume;
using pfc::geometry::BallBoundary;
using pfc::geometry::BallHarmonic;
using pfc::geometry::BallSeries;
using pfc::geometry::evaluate_ball;
using pfc::geometry::filled_ball;
using pfc::geometry::identity_affine_map;
using pfc::geometry::integrate_ball;
using pfc::geometry::Vec3;

namespace {

BallSeries series_with(pfc::geometry::FilledBall domain, BallHarmonic mode) {
  BallSeries series;
  series.domain = domain;
  add_ball_harmonic(series, mode);
  return series;
}

void require_complex(std::complex<double> got, std::complex<double> want,
                     double tol) {
  REQUIRE_THAT(got.real(), WithinAbs(want.real(), tol));
  REQUIRE_THAT(got.imag(), WithinAbs(want.imag(), tol));
}

} // namespace

TEST_CASE("a filled ball is a volume and the origin is not a boundary",
          "[geometry][ball]") {
  const auto ball = filled_ball(1.25);
  const double volume = 4.0 * std::numbers::pi * std::pow(1.25, 3) / 3.0;
  REQUIRE(ball.volume() == volume);
  REQUIRE(ball_has_boundary(ball, BallBoundary::RadialWall));
  REQUIRE(ball_contains(ball, 0.0, 0.0, 0.0));
  REQUIRE(ball_contains(ball, 1.25, std::numbers::pi, 4.0));
  REQUIRE_FALSE(ball_contains(ball, 1.2500001, 0.0, 0.0));
  REQUIRE_FALSE(ball_contains(ball, 0.2, std::numbers::pi + 0.01, 0.0));
  REQUIRE_FALSE(ball_contains(ball, -0.1, 0.0, 0.0));
  REQUIRE_THROWS_AS(filled_ball(0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(evaluate_ball(series_with(ball, {0, 0, 0, 1.0}), -0.2, 0.0, 0.0),
                    std::invalid_argument);
  auto rejected = series_with(ball, {0, 0, 0, 1.0});
  REQUIRE_THROWS_AS(add_ball_harmonic(rejected, {4, 0, 0, 1.0}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(add_ball_harmonic(rejected, {1, 2, 0, 1.0}),
                    std::invalid_argument);
}

TEST_CASE("a constant ball field and a radial polynomial", "[geometry][ball]") {
  const auto ball = filled_ball(0.7);
  const auto constant = series_with(ball, {0, 0, 0, 1.0});
  require_complex(evaluate_ball(constant, 0.0, 1.0, 0.4), 1.0, 0.0);
  require_complex(ball_laplacian(constant, 0.3, 0.4, 0.2), 0.0, 0.0);
  require_complex(integrate_ball(constant), ball.volume(), 1.0e-12);
  require_complex(ball_boundary_flux(constant), 0.0, 0.0);

  const auto radial = series_with(ball, {0, 0, 1, 1.0});
  const double r = 0.4;
  require_complex(evaluate_ball(radial, r, 0.5, 1.0), r * r, 1.0e-12);
  require_complex(evaluate_ball(radial, 0.0, 0.0, 0.0), 0.0, 0.0);
  require_complex(ball_laplacian(radial, r, 1.1, 0.3), 6.0, 1.0e-12);
  require_complex(ball_laplacian(radial, 0.0, 0.2, 0.2), 6.0, 1.0e-12);
  require_complex(ball_euler(radial, r, 0.7, 0.1), 2.0 * r * r, 1.0e-12);
}

TEST_CASE("solid harmonics are regular at the origin", "[geometry][ball]") {
  const auto ball = filled_ball(1.0);
  const auto dipole = series_with(ball, {1, 1, 0, 1.0});
  require_complex(evaluate_ball(dipole, 0.0, 0.4, 0.2), 0.0, 0.0);
  const auto axis = ball_gradient(dipole, 0.0, 0.0, 0.0);
  require_complex(axis[0], 1.0, 0.0);
  require_complex(axis[1], std::complex<double>(0.0, 1.0), 0.0);
  require_complex(axis[2], 0.0, 0.0);
  require_complex(ball_laplacian(dipole, 0.0, 1.0, 0.0), 0.0, 0.0);

  const auto quadrupole = series_with(ball, {2, 2, 0, 1.0});
  require_complex(evaluate_ball(quadrupole, 0.0, 0.2, 0.3), 0.0, 0.0);
  const auto flat = ball_gradient(quadrupole, 0.0, 0.2, 0.3);
  require_complex(flat[0], 0.0, 0.0);
  require_complex(flat[1], 0.0, 0.0);
  require_complex(flat[2], 0.0, 0.0);

  const double r = 0.3;
  const double theta = 0.8;
  const double phi = 0.4;
  require_complex(evaluate_ball(dipole, r, theta, phi),
                  evaluate_ball(dipole, r, theta, phi + 2.0 * std::numbers::pi),
                  1.0e-12);
}

TEST_CASE("a solid harmonic has spherical eigenvalue through the Euler operator",
          "[geometry][ball]") {
  const auto ball = filled_ball(1.1);
  const auto mode = series_with(ball, {2, 2, 0, 1.0});
  const double r = 0.45;
  const double theta = 0.7;
  const double phi = 0.35;
  const Vec3 point = ball_embedding(r, theta, phi);
  const auto zeta = std::complex<double>(point.x, point.y);
  require_complex(evaluate_ball(mode, r, theta, phi), zeta * zeta, 1.0e-12);
  require_complex(ball_laplacian(mode, r, theta, phi), 0.0, 1.0e-12);
  require_complex(ball_laplacian(mode, 0.0, 0.0, 0.0), 0.0, 0.0);
  require_complex(ball_euler(mode, r, theta, phi),
                  2.0 * evaluate_ball(mode, r, theta, phi), 1.0e-12);
  require_complex(integrate_ball(mode), 0.0, 0.0);
  require_complex(ball_boundary_flux(mode), 0.0, 0.0);

  const auto cubic = series_with(ball, {3, 1, 0, 1.0});
  const auto cubic_value =
      zeta * (4.0 * point.z * point.z - point.x * point.x - point.y * point.y);
  require_complex(evaluate_ball(cubic, r, theta, phi), cubic_value, 1.0e-12);
  require_complex(ball_laplacian(cubic, r, theta, phi), 0.0, 1.0e-12);
  require_complex(ball_euler(cubic, r, theta, phi),
                  3.0 * evaluate_ball(cubic, r, theta, phi), 1.0e-12);
}

TEST_CASE("the ball Laplacian integral equals the wall flux", "[geometry][ball]") {
  const auto ball = filled_ball(0.8);
  const auto radial = series_with(ball, {0, 0, 1, 1.0});
  require_complex(integrate_ball(radial),
                  4.0 * std::numbers::pi * std::pow(0.8, 5) / 5.0, 1.0e-12);
  require_complex(ball_boundary_flux(radial), 6.0 * ball.volume(), 1.0e-12);
  require_complex(integrate_ball(series_with(ball, {0, 0, 0, 6.0})),
                  ball_boundary_flux(radial), 1.0e-12);
}

TEST_CASE("an affine map sends the ball to an ellipsoid", "[geometry][ball]") {
  const auto ball = filled_ball(1.0);
  const auto stretch =
      affine_map_from_columns({2, 0, 0}, {0, 3, 0}, {0, 0, 4}, {0, 0, 0});
  const auto point = ball_physical_point(0.5, 0.5 * std::numbers::pi, 0.0, stretch);
  REQUIRE(point.x == 1.0);
  REQUIRE(point.y == 0.0);
  REQUIRE_THAT(point.z, WithinAbs(0.0, 1.0e-15));
  REQUIRE(ball_physical_volume(ball, stretch) == 24.0 * ball.volume());

  const auto radial = series_with(ball, {0, 0, 1, 1.0});
  REQUIRE(ball_physical_laplacian(radial, identity_affine_map(), 0.4, 0.2, 0.1) ==
          ball_laplacian(radial, 0.4, 0.2, 0.1));
  const double stretched = 2.0 / 4.0 + 2.0 / 9.0 + 2.0 / 16.0;
  require_complex(ball_physical_laplacian(radial, stretch, 0.4, 0.2, 0.1), stretched,
                  1.0e-12);
  require_complex(ball_physical_laplacian(radial, stretch, 0.0, 0.0, 0.0), stretched,
                  1.0e-12);

  const auto shear =
      affine_map_from_columns({1, 0, 0}, {0.5, 1, 0}, {0, 0, 1}, {1, -2, 0.25});
  require_complex(ball_physical_laplacian(radial, shear, 0.3, 1.0, 0.4), 6.5,
                  1.0e-12);
}
