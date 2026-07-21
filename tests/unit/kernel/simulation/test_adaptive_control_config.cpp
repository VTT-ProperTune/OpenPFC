// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/kernel/simulation/adaptive_control_config.hpp>

using namespace Catch::Matchers;
using namespace pfc::sim;

namespace {

[[nodiscard]] AdaptiveControlConfig valid_adaptive_scalar() {
  return AdaptiveControlConfig{
      .mode = AdaptiveControlMode::adaptive,
      .atol = 1e-6,
      .rtol = 1e-4,
      .safety_factor = 0.9,
      .growth_max = 2.0,
      .shrink_max = 0.5,
      .min_dt = 1e-12,
      .max_dt = 1.0,
      .max_sequential_rejections = 5,
  };
}

[[nodiscard]] bool has_parameter(const AdaptiveConfigValidationResult &result,
                                 std::string_view needle) {
  for (const auto &issue : result.issues) {
    if (issue.parameter.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] const AdaptiveConfigIssue *
find_parameter(const AdaptiveConfigValidationResult &result, std::string_view needle) {
  for (const auto &issue : result.issues) {
    if (issue.parameter.find(needle) != std::string::npos) {
      return &issue;
    }
  }
  return nullptr;
}

} // namespace

TEST_CASE("AdaptiveControlConfig fixed mode defaults validate",
          "[adaptive_control_config][unit]") {
  SECTION("default-constructed") {
    const AdaptiveControlConfig cfg{};
    REQUIRE(cfg.mode == AdaptiveControlMode::fixed);
    REQUIRE(validate(cfg).ok());
    REQUIRE_NOTHROW(make_adaptive_control_config(cfg));
  }

  SECTION("explicit fixed with zero tolerances") {
    const AdaptiveControlConfig cfg{.mode = AdaptiveControlMode::fixed,
                                    .atol = 0.0,
                                    .rtol = 0.0};
    REQUIRE(validate(cfg).ok());
    REQUIRE_NOTHROW(make_adaptive_control_config(cfg));
  }
}

TEST_CASE("AdaptiveControlConfig adaptive scalar success",
          "[adaptive_control_config][unit]") {
  const AdaptiveControlConfig cfg = valid_adaptive_scalar();
  const auto result = validate(cfg);
  REQUIRE(result.ok());
  REQUIRE_NOTHROW(make_adaptive_control_config(cfg));

  const AdaptiveConfigIdentity id = make_identity(cfg);
  REQUIRE(id.semantic_version() == "1.0.0");
  REQUIRE(id.version_major == k_adaptive_control_config_version_major);
  REQUIRE_FALSE(id.parameter_signature.empty());
}

TEST_CASE("AdaptiveControlConfig adaptive per-field success",
          "[adaptive_control_config][unit]") {
  const AdaptiveControlConfig cfg{
      .mode = AdaptiveControlMode::adaptive,
      .atol = 0.0,
      .rtol = 0.0,
      .atol_per_field = {1e-6, 0.0},
      .rtol_per_field = {0.0, 1e-4},
      .field_count = 2,
      .safety_factor = 0.9,
      .growth_max = 2.0,
      .shrink_max = 0.5,
      .min_dt = 1e-12,
      .max_dt = 1.0,
      .max_sequential_rejections = 10,
      .error_weights = {1.0, 0.0},
  };
  REQUIRE(validate(cfg).ok());
}

TEST_CASE("AdaptiveControlConfig rejects both-zero tolerances",
          "[adaptive_control_config][unit]") {
  SECTION("scalar both-zero") {
    AdaptiveControlConfig cfg = valid_adaptive_scalar();
    cfg.atol = 0.0;
    cfg.rtol = 0.0;
    const auto result = validate(cfg);
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_parameter(result, "atol"));
    REQUIRE(has_parameter(result, "rtol"));
  }

  SECTION("per-field index both-zero") {
    AdaptiveControlConfig cfg = valid_adaptive_scalar();
    cfg.field_count = 2;
    cfg.atol_per_field = {1e-6, 0.0};
    cfg.rtol_per_field = {1e-4, 0.0};
    const auto result = validate(cfg);
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_parameter(result, "atol_per_field[1]"));
  }

  SECTION("factory throws with parameter name") {
    AdaptiveControlConfig cfg = valid_adaptive_scalar();
    cfg.atol = 0.0;
    cfg.rtol = 0.0;
    REQUIRE_THROWS_AS(make_adaptive_control_config(cfg), std::invalid_argument);
    REQUIRE_THROWS_WITH(make_adaptive_control_config(cfg), ContainsSubstring("atol"));
  }
}

TEST_CASE("AdaptiveControlConfig rejects growth_max below one",
          "[adaptive_control_config][unit]") {
  AdaptiveControlConfig cfg = valid_adaptive_scalar();
  cfg.growth_max = 0.5;
  const auto result = validate(cfg);
  REQUIRE_FALSE(result.ok());
  const auto *issue = find_parameter(result, "growth_max");
  REQUIRE(issue != nullptr);
  REQUIRE_THAT(issue->allowed_range, ContainsSubstring(">= 1"));
}

TEST_CASE("AdaptiveControlConfig rejects shrink_max outside open unit interval",
          "[adaptive_control_config][unit]") {
  for (const double shrink : {0.0, 1.0, -0.1}) {
    SECTION("shrink_max=" + std::to_string(shrink)) {
      AdaptiveControlConfig cfg = valid_adaptive_scalar();
      cfg.shrink_max = shrink;
      const auto result = validate(cfg);
      REQUIRE_FALSE(result.ok());
      const auto *issue = find_parameter(result, "shrink_max");
      REQUIRE(issue != nullptr);
      REQUIRE_THAT(issue->allowed_range, ContainsSubstring("(0, 1)"));
    }
  }
}

TEST_CASE("AdaptiveControlConfig rejects min_dt greater than max_dt",
          "[adaptive_control_config][unit]") {
  SECTION("min_dt > max_dt") {
    AdaptiveControlConfig cfg = valid_adaptive_scalar();
    cfg.min_dt = 1.0;
    cfg.max_dt = 0.1;
    const auto result = validate(cfg);
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_parameter(result, "min_dt"));
  }

  SECTION("non-positive min_dt") {
    AdaptiveControlConfig cfg = valid_adaptive_scalar();
    cfg.min_dt = 0.0;
    const auto result = validate(cfg);
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_parameter(result, "min_dt"));
  }

  SECTION("non-positive max_dt") {
    AdaptiveControlConfig cfg = valid_adaptive_scalar();
    cfg.max_dt = -1.0;
    const auto result = validate(cfg);
    REQUIRE_FALSE(result.ok());
    REQUIRE(has_parameter(result, "max_dt"));
  }
}

TEST_CASE("AdaptiveControlConfig rejects non-positive rejection limit",
          "[adaptive_control_config][unit]") {
  for (const int limit : {0, -1}) {
    SECTION("max_sequential_rejections=" + std::to_string(limit)) {
      AdaptiveControlConfig cfg = valid_adaptive_scalar();
      cfg.max_sequential_rejections = limit;
      const auto result = validate(cfg);
      REQUIRE_FALSE(result.ok());
      REQUIRE(has_parameter(result, "max_sequential_rejections"));
    }
  }
}

TEST_CASE("AdaptiveControlConfig rejects all-zero error weights",
          "[adaptive_control_config][unit]") {
  AdaptiveControlConfig cfg = valid_adaptive_scalar();
  cfg.field_count = 2;
  cfg.error_weights = {0.0, 0.0};
  // Keep scalar tolerances so length checks on atol vectors are not required.
  const auto result = validate(cfg);
  REQUIRE_FALSE(result.ok());
  REQUIRE(has_parameter(result, "error_weights"));
}

TEST_CASE("AdaptiveControlConfig identity stable for identical parameters",
          "[adaptive_control_config][unit]") {
  const AdaptiveControlConfig a = valid_adaptive_scalar();
  AdaptiveControlConfig b = valid_adaptive_scalar();
  REQUIRE(make_identity(a).parameter_signature == make_identity(b).parameter_signature);

  b.atol = 2e-6;
  REQUIRE(make_identity(a).parameter_signature != make_identity(b).parameter_signature);
}
