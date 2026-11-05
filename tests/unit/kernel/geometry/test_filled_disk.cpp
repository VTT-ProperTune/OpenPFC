// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/geometry/filled_disk.hpp>

#include <cmath>
#include <complex>
#include <numbers>

using Catch::Matchers::WithinAbs;
using pfc::geometry::add_disk_harmonic;
using pfc::geometry::affine_map_from_columns;
using pfc::geometry::disk_boundary_flux;
using pfc::geometry::disk_contains;
using pfc::geometry::disk_gradient;
using pfc::geometry::disk_has_boundary;
using pfc::geometry::disk_laplacian;
using pfc::geometry::disk_physical_area;
using pfc::geometry::disk_physical_laplacian;
using pfc::geometry::disk_physical_point;
using pfc::geometry::DiskBoundary;
using pfc::geometry::DiskHarmonic;
using pfc::geometry::DiskSeries;
using pfc::geometry::evaluate_disk;
using pfc::geometry::filled_disk;
using pfc::geometry::identity_affine_map;
using pfc::geometry::integrate_disk;

namespace {

DiskSeries series_with(pfc::geometry::FilledDisk domain, DiskHarmonic mode) {
  DiskSeries series;
  series.domain = domain;
  add_disk_harmonic(series, mode);
  return series;
}

void require_complex(std::complex<double> got, std::complex<double> want,
                     double tol) {
  REQUIRE_THAT(got.real(), WithinAbs(want.real(), tol));
  REQUIRE_THAT(got.imag(), WithinAbs(want.imag(), tol));
}

} // namespace

TEST_CASE("a filled disk distinguishes the origin from the wall",
          "[geometry][disk]") {
  const auto disk = filled_disk(1.25);
  REQUIRE(disk.area() == std::numbers::pi * 1.25 * 1.25);
  REQUIRE(disk_has_boundary(disk, DiskBoundary::RadialWall));
  REQUIRE(disk_contains(disk, 0.0, 0.0));
  REQUIRE(disk_contains(disk, 1.25, 12.0));
  REQUIRE_FALSE(disk_contains(disk, 1.2500001, 0.0));
  REQUIRE_FALSE(disk_contains(disk, -0.1, 0.0));
  REQUIRE_THROWS_AS(filled_disk(0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(filled_disk(-1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(evaluate_disk(series_with(disk, {0, 0, 1.0}), -0.2, 0.0),
                    std::invalid_argument);
}

TEST_CASE("a constant disk field and a smooth radial polynomial",
          "[geometry][disk]") {
  const auto disk = filled_disk(0.7);
  const auto constant = series_with(disk, {0, 0, 1.0});
  require_complex(evaluate_disk(constant, 0.0, 1.0), 1.0, 0.0);
  require_complex(disk_laplacian(constant, 0.3, 0.4), 0.0, 0.0);
  require_complex(integrate_disk(constant), disk.area(), 1.0e-12);
  require_complex(disk_boundary_flux(constant), 0.0, 0.0);

  DiskSeries made;
  made.domain = disk;
  add_disk_harmonic(made, {0, 0, 1.0});
  add_disk_harmonic(made, {0, 1, 3.0});
  const double r = 0.4;
  require_complex(evaluate_disk(made, r, 0.2), 1.0 + 3.0 * r * r, 1.0e-12);
  require_complex(evaluate_disk(made, 0.0, 0.0), 1.0, 0.0);
  require_complex(disk_laplacian(made, r, 1.1), 12.0, 1.0e-12);
  require_complex(disk_laplacian(made, 0.0, 0.0), 12.0, 1.0e-12);
  const auto gradient = disk_gradient(made, r, 0.0);
  require_complex(gradient[0], 6.0 * r, 1.0e-12);
  require_complex(gradient[1], 0.0, 1.0e-12);
}

TEST_CASE("disk origin regularity holds for angular modes", "[geometry][disk]") {
  const auto disk = filled_disk(1.0);
  for (int angular = 0; angular <= 4; ++angular) {
    const auto mode = series_with(disk, {angular, 0, 1.0});
    const auto value = evaluate_disk(mode, 0.0, 0.3);
    REQUIRE(std::isfinite(value.real()));
    REQUIRE(std::isfinite(value.imag()));
    if (angular > 0) require_complex(value, 0.0, 0.0);
    const auto gradient = disk_gradient(mode, 0.0, 0.2);
    if (angular != 1) {
      require_complex(gradient[0], 0.0, 0.0);
      require_complex(gradient[1], 0.0, 0.0);
    }
    require_complex(disk_laplacian(mode, 0.0, 0.5), 0.0, 1.0e-12);
  }
  const auto dipole = series_with(disk, {1, 0, 1.0});
  const auto axis = disk_gradient(dipole, 0.0, 0.0);
  require_complex(axis[0], 1.0, 0.0);
  require_complex(axis[1], std::complex<double>(0.0, 1.0), 0.0);
  require_complex(evaluate_disk(dipole, 0.3, 0.4),
                  evaluate_disk(dipole, 0.3, 0.4 + 2.0 * std::numbers::pi), 1.0e-12);
  auto rejected = series_with(disk, {0, 0, 1.0});
  REQUIRE_THROWS_AS(add_disk_harmonic(rejected, {0, 13, 1.0}),
                    std::invalid_argument);
}

TEST_CASE("an angular Fourier mode is a disk Laplace solution", "[geometry][disk]") {
  const auto disk = filled_disk(1.1);
  constexpr int angular = 2;
  const auto mode = series_with(disk, {angular, 0, std::complex<double>(0.0, 1.0)});
  const double r = 0.6;
  const double theta = 0.7;
  const auto expected =
      std::pow(r, angular) * std::exp(std::complex<double>(0.0, angular * theta));
  require_complex(evaluate_disk(mode, r, theta),
                  std::complex<double>(0.0, 1.0) * expected, 1.0e-12);
  require_complex(disk_laplacian(mode, r, theta), 0.0, 1.0e-12);
  require_complex(disk_laplacian(mode, 0.0, 0.0), 0.0, 0.0);
  require_complex(integrate_disk(mode), 0.0, 0.0);
}

TEST_CASE("the disk Laplacian integral equals the wall flux", "[geometry][disk]") {
  const auto disk = filled_disk(0.8);
  const auto radial = series_with(disk, {0, 1, 1.0});
  require_complex(integrate_disk(radial), std::numbers::pi * std::pow(0.8, 4) / 2.0,
                  1.0e-12);
  require_complex(disk_boundary_flux(radial), 4.0 * disk.area(), 1.0e-12);
  require_complex(integrate_disk(series_with(disk, {0, 0, 4.0})),
                  disk_boundary_flux(radial), 1.0e-12);
}

TEST_CASE("an affine map sends the disk to a flat ellipse", "[geometry][disk]") {
  const auto disk = filled_disk(1.0);
  const auto stretch =
      affine_map_from_columns({2, 0, 0}, {0, 3, 0}, {0, 0, 1}, {0, 0, 0});
  const auto point = disk_physical_point(0.5, 0.0, stretch);
  REQUIRE(point.x == 1.0);
  REQUIRE(point.y == 0.0);
  REQUIRE(point.z == 0.0);
  REQUIRE(disk_physical_area(disk, stretch) == 6.0 * disk.area());

  const auto radial = series_with(disk, {0, 1, 1.0});
  REQUIRE(disk_physical_laplacian(radial, identity_affine_map(), 0.4, 0.2) ==
          disk_laplacian(radial, 0.4, 0.2));
  const double stretched = 2.0 / 4.0 + 2.0 / 9.0;
  require_complex(disk_physical_laplacian(radial, stretch, 0.4, 0.2), stretched,
                  1.0e-12);
  require_complex(disk_physical_laplacian(radial, stretch, 0.0, 0.0), stretched,
                  1.0e-12);

  const auto shear =
      affine_map_from_columns({1, 0, 0}, {0.5, 1, 0}, {0, 0, 1}, {0, 0, 0});
  require_complex(disk_physical_laplacian(radial, shear, 0.3, 1.0), 4.5, 1.0e-12);

  // xy = ((x+iy)^2 - (x-iy)^2)/(4i) has only a mixed Hessian.
  DiskSeries mixed;
  mixed.domain = disk;
  add_disk_harmonic(mixed, {2, 0, {0.0, -0.25}});
  add_disk_harmonic(mixed, {-2, 0, {0.0, 0.25}});
  require_complex(disk_laplacian(mixed, 0.3, 1.0), 0.0, 1.0e-12);
  // The induced inverse metric of the in-plane shear has g_xy=-0.5.
  const auto mapped = disk_physical_laplacian(mixed, shear, 0.3, 1.0);
  require_complex(mapped, -1.0, 1.0e-12);

  const auto tilt =
      affine_map_from_columns({1, 0, 1}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0});
  const auto tilted = disk_physical_point(0.5, 0.0, tilt);
  REQUIRE(tilted.x == 0.5);
  REQUIRE(tilted.y == 0.0);
  REQUIRE(tilted.z == 0.5);
  require_complex(disk_physical_laplacian(radial, tilt, 0.3, 0.4), 3.0, 1.0e-12);
}
