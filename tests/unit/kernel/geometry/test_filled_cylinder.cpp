// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/geometry/affine_map.hpp>
#include <openpfc/kernel/geometry/filled_cylinder.hpp>

#include <cmath>
#include <complex>
#include <numbers>

using Catch::Matchers::WithinAbs;
using pfc::geometry::add_cylinder_harmonic;
using pfc::geometry::affine_map_from_columns;
using pfc::geometry::AffineMap;
using pfc::geometry::cylinder_boundary_flux;
using pfc::geometry::cylinder_contains;
using pfc::geometry::cylinder_embedding;
using pfc::geometry::cylinder_gradient;
using pfc::geometry::cylinder_has_boundary;
using pfc::geometry::cylinder_laplacian;
using pfc::geometry::cylinder_physical_laplacian;
using pfc::geometry::cylinder_physical_point;
using pfc::geometry::cylinder_physical_volume;
using pfc::geometry::CylinderAxialPolicy;
using pfc::geometry::CylinderBoundary;
using pfc::geometry::CylinderHarmonic;
using pfc::geometry::CylinderSeries;
using pfc::geometry::diffuse_axisymmetric_cylinder;
using pfc::geometry::evaluate_cylinder;
using pfc::geometry::filled_cylinder;
using pfc::geometry::identity_affine_map;
using pfc::geometry::integrate_cylinder;
using pfc::geometry::Vec3;

namespace {

CylinderSeries series_with(pfc::geometry::FilledCylinder domain,
                           CylinderHarmonic mode) {
  CylinderSeries series;
  series.domain = domain;
  add_cylinder_harmonic(series, mode);
  return series;
}

void require_complex(std::complex<double> got, std::complex<double> want,
                     double tol) {
  REQUIRE_THAT(got.real(), WithinAbs(want.real(), tol));
  REQUIRE_THAT(got.imag(), WithinAbs(want.imag(), tol));
}

} // namespace

TEST_CASE("a filled cylinder distinguishes the axis from the wall and the caps",
          "[geometry][cylinder]") {
  const auto periodic =
      filled_cylinder(1.5, -0.25, 1.75, CylinderAxialPolicy::Periodic);
  REQUIRE(periodic.volume() == std::numbers::pi * 1.5 * 1.5 * 2.0);
  REQUIRE(cylinder_has_boundary(periodic, CylinderBoundary::RadialWall));
  REQUIRE_FALSE(cylinder_has_boundary(periodic, CylinderBoundary::LowerCap));
  REQUIRE_FALSE(cylinder_has_boundary(periodic, CylinderBoundary::UpperCap));
  REQUIRE(cylinder_contains(periodic, 0.0, 0.0, 0.0));
  REQUIRE(cylinder_contains(periodic, 1.5, 12.0, 1.75));
  REQUIRE_FALSE(cylinder_contains(periodic, 1.5000001, 0.0, 0.0));

  const auto finite = filled_cylinder(1.5, -0.25, 1.75, CylinderAxialPolicy::Finite);
  REQUIRE(cylinder_has_boundary(finite, CylinderBoundary::LowerCap));
  REQUIRE(cylinder_has_boundary(finite, CylinderBoundary::UpperCap));
  REQUIRE_THROWS_AS(filled_cylinder(0.0, 0.0, 1.0, CylinderAxialPolicy::Periodic),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(filled_cylinder(1.0, 2.0, 2.0, CylinderAxialPolicy::Finite),
                    std::invalid_argument);
}

TEST_CASE("constant and manufactured cylinder fields integrate with weight r",
          "[geometry][cylinder]") {
  const auto domain = filled_cylinder(1.25, 0.0, 2.0, CylinderAxialPolicy::Finite);
  const double volume = std::numbers::pi * 1.25 * 1.25 * 2.0;
  auto constant = series_with(domain, {0, 0, 0, 1.0});
  REQUIRE(evaluate_cylinder(constant, 0.0, 0.3, 0.4) == 1.0);
  REQUIRE(cylinder_laplacian(constant, 0.0, 0.3, 0.4) == 0.0);
  REQUIRE(integrate_cylinder(constant) == volume);
  REQUIRE(cylinder_boundary_flux(constant) == 0.0);

  add_cylinder_harmonic(constant, {0, 1, 0, 3.0});
  add_cylinder_harmonic(constant, {0, 0, 2, 2.0});
  const double r = 0.4;
  const double z = 0.7;
  const double field = 1.0 + 3.0 * r * r + 2.0 * z * z;
  require_complex(evaluate_cylinder(constant, r, -1.2, z), field, 1.0e-12);
  require_complex(evaluate_cylinder(constant, 0.0, 4.0, z), 1.0 + 2.0 * z * z,
                  1.0e-12);
  require_complex(cylinder_laplacian(constant, r, 0.2, z), 16.0, 1.0e-12);
  require_complex(cylinder_laplacian(constant, 0.0, 0.0, 0.0), 16.0, 1.0e-12);
  const auto gradient = cylinder_gradient(constant, r, 0.0, z);
  require_complex(gradient[0], 6.0 * r, 1.0e-12);
  require_complex(gradient[1], 0.0, 1.0e-12);
  require_complex(gradient[2], 4.0 * z, 1.0e-12);
}

TEST_CASE("the axis is regular for every Fourier angular mode",
          "[geometry][cylinder]") {
  const auto domain = filled_cylinder(2.0, 0.0, 1.0, CylinderAxialPolicy::Periodic);
  for (int angular = 0; angular <= 4; ++angular) {
    const auto mode = series_with(domain, {angular, 0, 0, 1.0});
    const auto value = evaluate_cylinder(mode, 0.0, 1.7, 0.3);
    const auto laplacian = cylinder_laplacian(mode, 0.0, 1.7, 0.3);
    const auto gradient = cylinder_gradient(mode, 0.0, 1.7, 0.3);
    REQUIRE(std::isfinite(value.real()));
    REQUIRE(std::isfinite(laplacian.real()));
    REQUIRE(std::isfinite(gradient[0].real()));
    if (angular > 0) REQUIRE(value == 0.0);
    if (angular != 1) {
      REQUIRE(gradient[0] == 0.0);
      REQUIRE(gradient[1] == 0.0);
    }
    REQUIRE(laplacian == 0.0);
  }
  const auto dipole = series_with(domain, {1, 0, 0, 1.0});
  const auto axis_gradient = cylinder_gradient(dipole, 0.0, 0.0, 0.2);
  REQUIRE(axis_gradient[0] == 1.0);
  REQUIRE(axis_gradient[1] == std::complex<double>(0.0, 1.0));
  const double theta = 0.7;
  const auto base = evaluate_cylinder(dipole, 0.3, theta, 0.2);
  require_complex(
      evaluate_cylinder(dipole, 0.3, theta + 2.0 * std::numbers::pi, 0.2), base,
      1.0e-12);
  require_complex(evaluate_cylinder(dipole, 0.3, theta + std::numbers::pi, 0.2),
                  -base, 1.0e-12);
}

TEST_CASE("periodic cylinder axial Fourier modes admit a real cosine",
          "[geometry][cylinder]") {
  const auto domain = filled_cylinder(1.0, 0.0, 2.0, CylinderAxialPolicy::Periodic);
  CylinderSeries series;
  series.domain = domain;
  add_cylinder_harmonic(series, {0, 0, 1, 0.5});
  add_cylinder_harmonic(series, {0, 0, -1, 0.5});
  const double z = 0.37;
  const double wave = std::numbers::pi;
  const auto value = evaluate_cylinder(series, 0.4, 0.1, z);
  const auto lap = cylinder_laplacian(series, 0.4, 0.1, z);
  require_complex(value, std::cos(wave * z), 1.0e-12);
  require_complex(lap, -wave * wave * std::cos(wave * z), 1.0e-11);
  require_complex(integrate_cylinder(series), 0.0, 1.0e-12);
}

TEST_CASE("a Fourier angular mode is a Laplace eigenmode along periodic z",
          "[geometry][cylinder]") {
  const auto domain = filled_cylinder(1.0, -0.5, 1.5, CylinderAxialPolicy::Periodic);
  const int angular = 2;
  const int axial = 1;
  const auto mode = series_with(domain, {angular, 0, axial, 1.0});
  const double kappa = 2.0 * std::numbers::pi / domain.length();
  const double r = 0.35;
  const double theta = 0.4;
  const double z = 0.2;
  const auto phase = std::polar(1.0, angular * theta + kappa * (z - domain.z_lower));
  const auto expected = std::pow(r, angular) * phase;
  require_complex(evaluate_cylinder(mode, r, theta, z), expected, 1.0e-12);
  require_complex(cylinder_laplacian(mode, r, theta, z), -kappa * kappa * expected,
                  1.0e-11);
  require_complex(cylinder_laplacian(mode, 0.0, 0.0, z), 0.0, 1.0e-12);
  require_complex(integrate_cylinder(mode), 0.0, 1.0e-12);
  require_complex(cylinder_boundary_flux(mode), 0.0, 1.0e-12);
}

TEST_CASE("axisymmetric diffusion follows the cylindrical Laplacian",
          "[geometry][cylinder]") {
  const auto domain = filled_cylinder(1.0, 0.0, 3.0, CylinderAxialPolicy::Periodic);
  CylinderSeries profile;
  profile.domain = domain;
  add_cylinder_harmonic(profile, {0, 0, 0, 3.0});
  add_cylinder_harmonic(profile, {0, 1, 0, 4.0});
  const double r = 0.5;
  require_complex(cylinder_laplacian(profile, r, 1.0, 1.0), 16.0, 1.0e-12);
  require_complex(cylinder_laplacian(profile, 0.0, 0.0, 2.0), 16.0, 1.0e-12);

  constexpr double diffusivity = 0.5;
  constexpr double dt = 0.25;
  const auto advanced = diffuse_axisymmetric_cylinder(profile, diffusivity, dt);
  const double expected = 3.0 + 16.0 * diffusivity * dt + 4.0 * r * r;
  require_complex(evaluate_cylinder(advanced, r, 2.0, 0.5), expected, 1.0e-12);
  const auto residual = evaluate_cylinder(advanced, r, 0.3, 1.2) -
                        evaluate_cylinder(profile, r, 0.3, 1.2);
  require_complex(residual,
                  dt * diffusivity * cylinder_laplacian(profile, r, 0.3, 1.2),
                  1.0e-12);

  CylinderSeries curved;
  curved.domain = domain;
  add_cylinder_harmonic(curved, {0, 2, 0, 1.0});
  require_complex(cylinder_laplacian(curved, r, 0.0, 0.0), 16.0 * r * r, 1.0e-12);
  REQUIRE_THROWS_AS(diffuse_axisymmetric_cylinder(
                        series_with(domain, {1, 0, 0, 1.0}), diffusivity, dt),
                    std::invalid_argument);
}

TEST_CASE("the Laplacian integral equals the physical-boundary flux",
          "[geometry][cylinder]") {
  const auto finite = filled_cylinder(0.8, 0.0, 1.7, CylinderAxialPolicy::Finite);
  const double volume = std::numbers::pi * 0.8 * 0.8 * 1.7;
  const auto radial = series_with(finite, {0, 1, 0, 1.0});
  require_complex(integrate_cylinder(radial),
                  std::numbers::pi * std::pow(0.8, 4) * 1.7 / 2.0, 1.0e-12);
  require_complex(cylinder_boundary_flux(radial), 4.0 * volume, 1.0e-12);
  const auto lap = series_with(finite, {0, 0, 0, 4.0});
  require_complex(integrate_cylinder(lap), cylinder_boundary_flux(radial), 1.0e-12);

  const auto axial = series_with(finite, {0, 0, 2, 1.0});
  require_complex(cylinder_laplacian(axial, 0.2, 0.4, 0.5), 2.0, 1.0e-12);
  require_complex(cylinder_boundary_flux(axial), 2.0 * 1.7 * std::numbers::pi * 0.64,
                  1.0e-12);
  require_complex(integrate_cylinder(series_with(finite, {0, 0, 0, 2.0})),
                  cylinder_boundary_flux(axial), 1.0e-12);
}

TEST_CASE("an affine map composes with the cylinder embedding",
          "[geometry][cylinder]") {
  const auto domain = filled_cylinder(1.0, 0.0, 1.0, CylinderAxialPolicy::Finite);
  const AffineMap stretch =
      affine_map_from_columns({2, 0, 0}, {0, 3, 0}, {0, 0, 4}, {0, 0, 0});
  const Vec3 physical = cylinder_physical_point(0.5, 0.0, 0.25, stretch);
  REQUIRE(physical.x == 1.0);
  REQUIRE(physical.y == 0.0);
  REQUIRE(physical.z == 1.0);
  REQUIRE(cylinder_physical_volume(domain, stretch) == 24.0 * domain.volume());

  const auto radial = series_with(domain, {0, 1, 0, 1.0});
  REQUIRE(
      cylinder_physical_laplacian(radial, identity_affine_map(), 0.4, 0.2, 0.1) ==
      cylinder_laplacian(radial, 0.4, 0.2, 0.1));
  const double expected = 2.0 / (2.0 * 2.0) + 2.0 / (3.0 * 3.0);
  require_complex(cylinder_physical_laplacian(radial, stretch, 0.0, 0.0, 0.3),
                  expected, 1.0e-12);
  require_complex(cylinder_physical_laplacian(radial, stretch, 0.7, 1.1, 0.3),
                  expected, 1.0e-12);
}
