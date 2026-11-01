// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/frontend/ui/from_json_robin.hpp>
#include <openpfc/frontend/utils/toml_to_json.hpp>

using Catch::Matchers::ContainsSubstring;
using pfc::communication::CartesianFace;
using pfc::communication::RobinBoundary;
using pfc::ui::from_json;

TEST_CASE("JSON and TOML configure named Robin faces", "[robin_bc][json]") {
  const auto boundary = from_json<RobinBoundary<double>>(nlohmann::json::parse(R"({
    "faces": [
      {"face": "xm", "alpha": 1.0, "beta": 0.5, "gamma": -0.25},
      {"face": "zp", "alpha": 0.0, "beta": 2.0, "gamma": 0.125}
    ]
  })"));
  REQUIRE(boundary.enabled(CartesianFace::Xm));
  REQUIRE(boundary.enabled(CartesianFace::Zp));
  REQUIRE_FALSE(boundary.enabled(CartesianFace::Ym));
  REQUIRE(boundary.alpha(CartesianFace::Xm) == 1.0);
  REQUIRE(boundary.beta(CartesianFace::Xm) == 0.5);
  REQUIRE(boundary.gamma(CartesianFace::Zp, 0.0) == 0.125);

  const auto config = toml::parse(R"(
[[faces]]
face = "ym"
alpha = 0.0
beta = 1.0
gamma = -0.25
)");
  const auto from_toml =
      from_json<RobinBoundary<double>>(pfc::utils::toml_to_json(config));
  REQUIRE(from_toml.enabled(CartesianFace::Ym));
  REQUIRE(from_toml.beta(CartesianFace::Ym) == 1.0);
  REQUIRE(from_toml.gamma(CartesianFace::Ym, 0.0) == -0.25);
}

TEST_CASE("Robin JSON rejects a singular or unknown face", "[robin_bc][json]") {
  const char *rejected[] = {
      "[]",
      "{}",
      R"({"faces": []})",
      R"({"centering": "cell", "faces": [{"face": "xm", "alpha": 1.0, "beta": 0.0, "gamma": 0.0}]})",
      R"({"faces": [{"face": "left", "alpha": 1.0, "beta": 0.0, "gamma": 0.0}]})",
      R"({"faces": [{"face": "xm", "alpha": 1.0, "beta": 0.0, "gamma": 0.0}, {"face": "xm", "alpha": 2.0, "beta": 0.0, "gamma": 1.0}]})",
      R"({"faces": [{"face": "xm", "alpha": 0.0, "beta": 0.0, "gamma": 1.0}]})",
      R"({"faces": [{"face": "xm", "alpha": 1.0, "beta": 0.5, "gamma": null}]})",
  };
  for (const char *text : rejected) {
    REQUIRE_THROWS_WITH(
        from_json<RobinBoundary<double>>(nlohmann::json::parse(text)),
        ContainsSubstring("robin"));
  }
}
