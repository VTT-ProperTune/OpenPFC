// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_reference_scalar.cpp
 * @brief One operator reads a cylinder, a disk, and a ball without a shape enum.
 *
 * `sample_reference` is the only geometry call in the comparisons below.
 * The expected Laplacian, integral, and measure are the analytical values
 * for `1` and for `R² - r²`. An identity map and a uniform scale both go
 * through that function. The Cartesian symbol stays on `laplacian_symbol`.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/fft/affine_laplacian.hpp>
#include <openpfc/kernel/geometry/reference_scalar.hpp>

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

using Catch::Matchers::WithinAbs;
using pfc::geometry::affine_map_from_matrix;

namespace {

template <class Series, class Station>
void expect_reading(const Series &series, const pfc::geometry::AffineMap &map,
                    Station station, double laplacian, double integral,
                    double measure) {
  const pfc::geometry::ReferenceReading reading =
      pfc::geometry::sample_reference(series, map, station);
  REQUIRE_THAT(reading.laplacian.real(), WithinAbs(laplacian, 1e-12));
  REQUIRE_THAT(reading.laplacian.imag(), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(reading.reference_integral.real(), WithinAbs(integral, 1e-12));
  REQUIRE_THAT(reading.reference_integral.imag(), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(reading.physical_measure, WithinAbs(measure, 1e-12));
  const double reference_size =
      pfc::geometry::reference_domain_measure(series.domain);
  const double physical_integral = integral * measure / reference_size;
  REQUIRE_THAT(reading.physical_integral.real(),
               WithinAbs(physical_integral, 1e-12));
  REQUIRE_THAT(reading.physical_integral.imag(), WithinAbs(0.0, 1e-12));
}

void require_constant_physical_mean(const pfc::geometry::ReferenceReading &reading) {
  const auto physical_mean = reading.physical_integral / reading.physical_measure;
  REQUIRE_THAT(physical_mean.real(), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(physical_mean.imag(), WithinAbs(0.0, 1e-12));
}

} // namespace

TEST_CASE("A reference scalar reads the map without a domain enum",
          "[reference_domain]") {
  constexpr double radius = 1.25;
  constexpr double length = 0.8;
  constexpr double scale = 2.0;
  const double pi = std::numbers::pi;
  const double radius2 = radius * radius;
  const double cylinder_volume = pi * radius2 * length;
  const double disk_area = pi * radius2;
  const double ball_volume = (4.0 / 3.0) * pi * radius * radius2;
  const auto identity = pfc::geometry::identity_affine_map();
  const auto scaled = pfc::geometry::affine_map_from_columns(
      pfc::geometry::Vec3{scale, 0.0, 0.0}, pfc::geometry::Vec3{0.0, scale, 0.0},
      pfc::geometry::Vec3{0.0, 0.0, scale}, pfc::geometry::Vec3{0.0, 0.0, 0.0});
  REQUIRE(identity.uses_cartesian_laplacian());
  REQUIRE_FALSE(scaled.uses_cartesian_laplacian());

  pfc::geometry::CylinderSeries cylinder_one;
  cylinder_one.domain = pfc::geometry::filled_cylinder(
      radius, 0.0, length, pfc::geometry::CylinderAxialPolicy::Periodic);
  pfc::geometry::add_cylinder_harmonic(cylinder_one, {0, 0, 0, 1.0});
  pfc::geometry::CylinderSeries cylinder_parabola = cylinder_one;
  cylinder_parabola.modes.clear();
  pfc::geometry::add_cylinder_harmonic(cylinder_parabola, {0, 0, 0, radius2});
  pfc::geometry::add_cylinder_harmonic(cylinder_parabola, {0, 1, 0, -1.0});
  const pfc::geometry::CylinderStation cylinder_station{0.3 * radius, 0.7,
                                                        0.4 * length};
  const double cylinder_parabola_integral = 0.5 * pi * length * radius2 * radius2;
  expect_reading(cylinder_one, identity, cylinder_station, 0.0, cylinder_volume,
                 cylinder_volume);
  expect_reading(cylinder_one, scaled, cylinder_station, 0.0, cylinder_volume,
                 cylinder_volume * scale * scale * scale);
  const auto cylinder_physical =
      pfc::geometry::sample_reference(cylinder_one, scaled, cylinder_station);
  require_constant_physical_mean(cylinder_physical);
  expect_reading(cylinder_parabola, identity, cylinder_station, -4.0,
                 cylinder_parabola_integral, cylinder_volume);
  expect_reading(cylinder_parabola, scaled, cylinder_station, -4.0 / (scale * scale),
                 cylinder_parabola_integral,
                 cylinder_volume * scale * scale * scale);

  pfc::geometry::DiskSeries disk_one;
  disk_one.domain = pfc::geometry::filled_disk(radius);
  pfc::geometry::add_disk_harmonic(disk_one, {0, 0, 1.0});
  pfc::geometry::DiskSeries disk_parabola = disk_one;
  disk_parabola.modes.clear();
  pfc::geometry::add_disk_harmonic(disk_parabola, {0, 0, radius2});
  pfc::geometry::add_disk_harmonic(disk_parabola, {0, 1, -1.0});
  const pfc::geometry::DiskStation disk_station{0.45 * radius, 1.2};
  const double disk_parabola_integral = 0.5 * pi * radius2 * radius2;
  expect_reading(disk_one, identity, disk_station, 0.0, disk_area, disk_area);
  expect_reading(disk_one, scaled, disk_station, 0.0, disk_area,
                 disk_area * scale * scale);
  const auto disk_physical =
      pfc::geometry::sample_reference(disk_one, scaled, disk_station);
  require_constant_physical_mean(disk_physical);
  expect_reading(disk_parabola, identity, disk_station, -4.0, disk_parabola_integral,
                 disk_area);
  expect_reading(disk_parabola, scaled, disk_station, -4.0 / (scale * scale),
                 disk_parabola_integral, disk_area * scale * scale);

  pfc::geometry::BallSeries ball_one;
  ball_one.domain = pfc::geometry::filled_ball(radius);
  pfc::geometry::add_ball_harmonic(ball_one, {0, 0, 0, 1.0});
  pfc::geometry::BallSeries ball_parabola = ball_one;
  ball_parabola.modes.clear();
  pfc::geometry::add_ball_harmonic(ball_parabola, {0, 0, 0, radius2});
  pfc::geometry::add_ball_harmonic(ball_parabola, {0, 0, 1, -1.0});
  const pfc::geometry::BallStation ball_station{0.55 * radius, 0.8, 1.4};
  const double ball_parabola_integral =
      (8.0 / 15.0) * pi * radius2 * radius2 * radius;
  expect_reading(ball_one, identity, ball_station, 0.0, ball_volume, ball_volume);
  expect_reading(ball_one, scaled, ball_station, 0.0, ball_volume,
                 ball_volume * scale * scale * scale);
  const auto ball_physical =
      pfc::geometry::sample_reference(ball_one, scaled, ball_station);
  require_constant_physical_mean(ball_physical);
  expect_reading(ball_parabola, identity, ball_station, -6.0, ball_parabola_integral,
                 ball_volume);
  expect_reading(ball_parabola, scaled, ball_station, -6.0 / (scale * scale),
                 ball_parabola_integral, ball_volume * scale * scale * scale);

  REQUIRE_THAT(pfc::fft::laplacian_symbol(identity, 3.0, 0.0, 0.0),
               WithinAbs(-9.0, 1e-15));
  REQUIRE_THAT(pfc::fft::laplacian_symbol(scaled, 3.0, 0.0, 0.0),
               WithinAbs(-2.25, 1e-12));

  // Invalid maps cannot be manufactured through the validated public API.
  const double singular[9] = {0, 0, 0, 0, 1, 0, 0, 0, 1};
  const pfc::geometry::Vec3 shift{0, 0, 0};
  REQUIRE_THROWS_AS(affine_map_from_matrix(singular, shift), std::invalid_argument);
}
