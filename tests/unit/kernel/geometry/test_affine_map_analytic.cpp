// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/geometry/affine_map.hpp>

#include <stdexcept>
#include <type_traits>

TEST_CASE("an affine map is an immutable geometry value independent of FFTW",
          "[geometry][affine_map][analytic]") {
  using pfc::geometry::AffineMap;
  using pfc::geometry::Vec3;
  static_assert(std::is_trivially_copyable_v<AffineMap>);
  static_assert(!std::is_copy_assignable_v<AffineMap>);
  const double matrix[9] = {2, 0, 0, 0.5, 1, 0, 0, 0, 3};
  const AffineMap map = pfc::geometry::affine_map_from_matrix(matrix, {1, -2, 4});
  REQUIRE(map.determinant == 6.0);
  REQUIRE_FALSE(map.uses_cartesian_laplacian());
  const Vec3 x = map.to_physical({1, 2, -1});
  const Vec3 xi = map.to_reference(x);
  REQUIRE(x.x == 4.0);
  REQUIRE(x.y == 0.0);
  REQUIRE(x.z == 1.0);
  REQUIRE(xi.x == 1.0);
  REQUIRE(xi.y == 2.0);
  REQUIRE(xi.z == -1.0);
  REQUIRE(map.volume_scale() == 6.0);
  const double singular[9] = {1, 0, 0, 2, 0, 0, 0, 0, 1};
  REQUIRE_THROWS_AS(pfc::geometry::affine_map_from_matrix(singular, {}),
                    std::invalid_argument);
}
