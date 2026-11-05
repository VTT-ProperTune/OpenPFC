// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_pipe.cpp
 * @brief Laminar Hagen–Poiseuille balance on the filled cylinder.
 *
 * The pointwise profile error at independent interior radii, wall value,
 * flow rate, signed kinematic wall flux and momentum residual come from
 * `hagen_poiseuille`. That function calls
 * the cylinder Laplacian, gradient, and integral. Degree 0 cannot
 * carry the parabola. Degrees 1–3 recover it, including when one
 * angular mode and one axial mode are admitted. The periodic cube is
 * not masked.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>

#include <openpfc/kernel/geometry/filled_cylinder.hpp>

#include <flow/pipe.hpp>

using Catch::Matchers::WithinAbs;

namespace {

struct Oracle {
  double center;
  double flow_rate;
  double wall_shear;
  double leading;
};

Oracle oracle_for(const flow::PipeRequest &request) {
  const double pi = std::acos(-1.0);
  const double radius2 = request.radius * request.radius;
  Oracle oracle;
  oracle.leading = request.force / (4.0 * request.viscosity);
  oracle.center = oracle.leading * radius2;
  oracle.flow_rate =
      pi * request.force * radius2 * radius2 / (8.0 * request.viscosity);
  oracle.wall_shear = -0.5 * request.force * request.radius;
  return oracle;
}

void expect_resolved(const flow::PipeRequest &request) {
  const Oracle oracle = oracle_for(request);
  const auto report = flow::hagen_poiseuille(request);
  const auto domain =
      pfc::geometry::filled_cylinder(request.radius, 0.0, request.length,
                                     pfc::geometry::CylinderAxialPolicy::Periodic);
  REQUIRE(pfc::geometry::cylinder_has_boundary(
      domain, pfc::geometry::CylinderBoundary::RadialWall));
  REQUIRE_FALSE(pfc::geometry::cylinder_has_boundary(
      domain, pfc::geometry::CylinderBoundary::LowerCap));
  REQUIRE_FALSE(pfc::geometry::cylinder_has_boundary(
      domain, pfc::geometry::CylinderBoundary::UpperCap));
  REQUIRE(report.radial_wall);
  REQUIRE_FALSE(report.has_caps);
  REQUIRE(report.wall_error < 1e-12);
  REQUIRE(report.profile_error < 1e-9);
  REQUIRE(report.axis_gradient < 1e-9);
  REQUIRE_THAT(report.axis_value, WithinAbs(oracle.center, 1e-9));
  REQUIRE_THAT(report.flow_rate, WithinAbs(oracle.flow_rate, 1e-9));
  REQUIRE(std::abs(report.flow_imag) < 1e-9);
  REQUIRE_THAT(report.wall_shear, WithinAbs(oracle.wall_shear, 1e-9));
  REQUIRE(report.shear_spread < 1e-9);
  REQUIRE(report.momentum < 1e-8);
  REQUIRE(report.divergence < 1e-9);
  REQUIRE(report.metric_gap < 1e-9);
  REQUIRE_THAT(report.leading_coefficient, WithinAbs(oracle.leading, 1e-9));
  REQUIRE(std::abs(report.higher_coefficient) < 1e-8);
  REQUIRE(std::abs(report.angular_coefficient) < 1e-8);
  REQUIRE(std::abs(report.axial_coefficient) < 1e-8);
}

} // namespace

TEST_CASE("Hagen-Poiseuille recovers the filled-cylinder parabola", "[pipe]") {
  const flow::PipeRequest unresolved{1.0, 2.0, 0.2, 0.8, 0, false, false};
  const auto missed = flow::hagen_poiseuille(unresolved);
  REQUIRE_THAT(missed.flow_rate, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(missed.axis_value, WithinAbs(0.0, 1e-15));
  REQUIRE(missed.momentum > 0.5 * unresolved.force);
  REQUIRE(missed.profile_error > 0.1);
  REQUIRE(missed.radial_wall);
  REQUIRE_FALSE(missed.has_caps);

  for (const int degree : {1, 2, 3}) {
    flow::PipeRequest request{1.0, 2.0, 0.2, 0.8, degree, true, true};
    expect_resolved(request);
  }

  flow::PipeRequest narrow{0.5, 1.25, 0.05, 1.2, 2, true, true};
  expect_resolved(narrow);
  flow::PipeRequest wide{1.4, 0.8, 0.3, 0.4, 1, false, false};
  expect_resolved(wide);
}

TEST_CASE("Hagen-Poiseuille normal equations preserve small viscosity scaling",
          "[pipe]") {
  // The unscaled normal matrix's diagonal is below 1e-14 here, but the
  // physical profile u(r) = 1 - r² is perfectly regular.
  const flow::PipeRequest tiny{1.0, 2.0, 1.0e-9, 4.0e-9, 1, false, false};
  expect_resolved(tiny);
}

TEST_CASE("Hagen-Poiseuille rejects a nonpositive pipe", "[pipe]") {
  flow::PipeRequest request{1.0, 2.0, 0.2, 0.8, 1, false, false};
  request.radius = 0.0;
  REQUIRE_THROWS_AS(flow::hagen_poiseuille(request), std::invalid_argument);
  request.radius = 1.0;
  request.viscosity = 0.0;
  REQUIRE_THROWS_AS(flow::hagen_poiseuille(request), std::invalid_argument);
  request.viscosity = 0.2;
  request.radial_degree = 5;
  REQUIRE_THROWS_AS(flow::hagen_poiseuille(request), std::invalid_argument);
}
