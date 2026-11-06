// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// Pure affine-geometry checks: intentionally no FFTW or HeFFTe includes.
// These compile even when the FFT-backed spectral parity TU is excluded.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/geometry/affine_map.hpp>

#include <limits>
#include <stdexcept>

using Catch::Matchers::WithinAbs;

TEST_CASE("affine coordinate geometry works without an FFT backend",
          "[geometry][affine_map_core]") {
  using namespace pfc::geometry;
  const auto map =
      affine_map_from_columns({2, 0, 0}, {0.5, 1, 0}, {0, 0, 3}, {1, -2, 4});
  const Vec3 xi{1, 2, -1};
  const Vec3 physical = map.to_physical(xi);
  REQUIRE(physical.x == 4.0);
  REQUIRE(physical.y == 0.0);
  REQUIRE(physical.z == 1.0);
  const Vec3 back = map.to_reference(physical);
  REQUIRE_THAT(back.x, WithinAbs(xi.x, 1e-14));
  REQUIRE_THAT(back.y, WithinAbs(xi.y, 1e-14));
  REQUIRE_THAT(back.z, WithinAbs(xi.z, 1e-14));
  REQUIRE(map.volume_scale() == 6.0);
  REQUIRE(physical_volume(map, 2.0) == 12.0);
}

TEST_CASE("pure affine factory rejects invalid mappings without FFTW",
          "[geometry][affine_map_core]") {
  using namespace pfc::geometry;
  const double singular[9] = {1, 0, 0, 2, 0, 0, 0, 0, 1};
  REQUIRE_THROWS_AS(affine_map_from_matrix(singular, {}), std::invalid_argument);
  const double identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const Vec3 invalid_shift{0, std::numeric_limits<double>::infinity(), 0};
  REQUIRE_THROWS_AS(affine_map_from_matrix(identity, invalid_shift),
                    std::invalid_argument);
}
