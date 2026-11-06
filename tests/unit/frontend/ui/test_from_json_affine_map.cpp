// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/frontend/ui/from_json_affine_map.hpp>
#include <openpfc/frontend/utils/toml_to_json.hpp>
#include <openpfc/kernel/fft/affine_laplacian.hpp>

using Catch::Matchers::ContainsSubstring;
using pfc::geometry::Vec3;
using pfc::ui::affine_map_to_json;
using pfc::ui::from_json;

TEST_CASE("JSON and TOML round-trip a constant affine map", "[json][affine_map]") {
  const auto map = from_json<pfc::geometry::AffineMap>(nlohmann::json::parse(R"({
    "columns": [[2, 0, 0], [0.5, 1, 0], [0, 0, 3]],
    "translation": [1, -2, 4]
  })"));
  REQUIRE(map.determinant == 6.0);
  const Vec3 physical = map.to_physical({1, 2, -1});
  REQUIRE(physical.x == 4.0);
  REQUIRE(physical.y == 0.0);
  REQUIRE(physical.z == 1.0);
  REQUIRE_FALSE(map.uses_cartesian_laplacian());

  const auto again = from_json<pfc::geometry::AffineMap>(affine_map_to_json(map));
  REQUIRE(again.determinant == map.determinant);
  const Vec3 repeated = again.to_physical({1, 2, -1});
  REQUIRE(repeated.x == physical.x);
  REQUIRE(repeated.y == physical.y);
  REQUIRE(repeated.z == physical.z);
  REQUIRE(again.translation[0] == 1.0);
  REQUIRE(again.translation[1] == -2.0);
  REQUIRE(again.translation[2] == 4.0);

  const auto config = toml::parse(R"(
columns = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
translation = [0.0, 3.0, 0.0]
)");
  const auto shifted =
      from_json<pfc::geometry::AffineMap>(pfc::utils::toml_to_json(config));
  REQUIRE(shifted.uses_cartesian_laplacian());
  const Vec3 shifted_point = shifted.to_physical({1, 1, 1});
  REQUIRE(shifted_point.x == 1.0);
  REQUIRE(shifted_point.y == 4.0);
  REQUIRE(shifted_point.z == 1.0);
  REQUIRE(pfc::fft::laplacian_symbol(shifted, 0.5, -0.25, 2.0) ==
          pfc::fft::kspace::k_laplacian_value(0.5, -0.25, 2.0));
}

TEST_CASE("affine-map JSON rejects a singular or malformed matrix",
          "[json][affine_map]") {
  const char *rejected[] = {
      "[]",
      "{}",
      R"({"columns": [[1, 0, 0], [0, 1, 0]]})",
      R"({"columns": [[1, 0], [0, 1, 0], [0, 0, 1]]})",
      R"({"columns": [[1, 0, 0], [2, 0, 0], [0, 0, 1]]})",
      R"({"columns": [[1, 0, null], [0, 1, 0], [0, 0, 1]]})",
      R"({"columns": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": [1, 2]})",
  };
  for (const char *text : rejected) {
    REQUIRE_THROWS_WITH(
        from_json<pfc::geometry::AffineMap>(nlohmann::json::parse(text)),
        ContainsSubstring("affine_map"));
  }
}
