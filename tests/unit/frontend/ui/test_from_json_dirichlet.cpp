// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/frontend/ui/from_json_dirichlet.hpp>
#include <openpfc/frontend/utils/toml_to_json.hpp>

using Catch::Matchers::ContainsSubstring;
using pfc::communication::CartesianFace;
using pfc::communication::DirichletBoundary;
using pfc::communication::DirichletCentering;
using pfc::ui::from_json;

TEST_CASE("JSON and TOML configure named Dirichlet faces", "[dirichlet_bc][json]") {
  const auto boundary =
      from_json<DirichletBoundary<double>>(nlohmann::json::parse(R"({
    "faces": [
      {"face": "xm", "value": 0.0},
      {"face": "xp", "value": 1.5},
      {"face": "zm", "value": -2.0}
    ]
  })"));
  REQUIRE(boundary.centering == DirichletCentering::Node);
  REQUIRE(boundary.enabled(CartesianFace::Xm));
  REQUIRE(boundary.enabled(CartesianFace::Xp));
  REQUIRE(boundary.enabled(CartesianFace::Zm));
  REQUIRE_FALSE(boundary.enabled(CartesianFace::Ym));
  REQUIRE(boundary.evaluate(CartesianFace::Xp, {}, 0.0) == 1.5);

  const auto config = toml::parse(R"(
centering = "cell"
[[faces]]
face = "ym"
value = -0.25
)");
  const auto from_toml =
      from_json<DirichletBoundary<double>>(pfc::utils::toml_to_json(config));
  REQUIRE(from_toml.centering == DirichletCentering::CellFace);
  REQUIRE(from_toml.evaluate(CartesianFace::Ym, {}, 0.0) == -0.25);
}

TEST_CASE("Dirichlet JSON rejects a bad face list", "[dirichlet_bc][json]") {
  for (const char *text :
       {"[]", R"({"faces": []})", R"({"faces": [{"face": "left", "value": 0}]})",
        R"({"faces": [{"face": "xm", "value": 0}, {"face": "xm", "value": 1}]})",
        R"({"centering": "face", "faces": [{"face": "xm", "value": 0}]})",
        R"({"faces": [{"face": "xm", "value": null}]})"}) {
    INFO(text);
    REQUIRE_THROWS_WITH(
        from_json<DirichletBoundary<double>>(nlohmann::json::parse(text)),
        ContainsSubstring("dirichlet"));
  }
}
