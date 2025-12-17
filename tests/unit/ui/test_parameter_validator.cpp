// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/ui/parameter_validator.hpp"
#include <catch2/catch_all.hpp>

using namespace pfc::ui;

TEST_CASE("ParameterValidator with required parameter",
          "[parameter_validator][unit]") {
  ParameterValidator validator;
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("temperature")
                             .required(true)
                             .range(0.0, 10000.0)
                             .build());

  SECTION("Valid configuration with required parameter") {
    nlohmann::json config = {{"temperature", 3300.0}};
    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Missing required parameter") {
    nlohmann::json config = {};
    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE_FALSE(result.errors.empty());
    auto errors_str = result.format_errors();
    REQUIRE(errors_str.find("temperature") != std::string::npos);
  }

  SECTION("Required parameter out of range") {
    nlohmann::json config = {{"temperature", -100.0}};
    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    auto errors_str = result.format_errors();
    REQUIRE(errors_str.find("below minimum") != std::string::npos);
  }
}

TEST_CASE("ParameterValidator with optional parameter",
          "[parameter_validator][unit]") {
  ParameterValidator validator;
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("mobility")
                             .required(false)
                             .range(0.0, 100.0)
                             .build());

  SECTION("Valid configuration without optional parameter") {
    nlohmann::json config = {};
    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Valid configuration with optional parameter") {
    nlohmann::json config = {{"mobility", 50.0}};
    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Optional parameter out of range") {
    nlohmann::json config = {{"mobility", 150.0}};
    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
  }
}

TEST_CASE("ParameterValidator with default value", "[parameter_validator][unit]") {
  ParameterValidator validator;
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("tolerance")
                             .default_val(1e-6)
                             .range(1e-12, 1e-3)
                             .build());

  SECTION("Missing parameter uses default") {
    nlohmann::json config = {};
    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    // Default value handling is internal to validator
    REQUIRE(result.validated_params.find("tolerance") !=
            result.validated_params.end());
  }

  SECTION("Explicit value overrides default") {
    nlohmann::json config = {{"tolerance", 1e-8}};
    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.find("tolerance") !=
            result.validated_params.end());
  }
}

TEST_CASE("Special floating point values", "[parameter_validator][unit]") {
  ParameterValidator validator;
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("temperature")
                             .range(0.0, 10000.0)
                             .build());

  SECTION("NaN is rejected") {
    nlohmann::json config = {
        {"temperature", std::numeric_limits<double>::quiet_NaN()}};
    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
  }

  SECTION("Infinity is rejected") {
    nlohmann::json config = {
        {"temperature", std::numeric_limits<double>::infinity()}};
    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
  }
}

TEST_CASE("Nested parameter validation", "[parameter_validator][unit]") {
  ParameterValidator validator;
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("model.temperature")
                             .required(true)
                             .range(0.0, 10000.0)
                             .build());

  SECTION("Valid nested parameter") {
    nlohmann::json config = {{"model", {{"temperature", 3300.0}}}};
    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Missing nested parameter") {
    nlohmann::json config = {{"model", {}}};
    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    auto errors_str = result.format_errors();
    REQUIRE(errors_str.find("model.temperature") != std::string::npos);
  }

  SECTION("Deeply nested parameters") {
    ParameterValidator deep_validator;
    deep_validator.add_metadata(ParameterMetadata<double>::builder()
                                    .name("a.b.c.d.value")
                                    .required(true)
                                    .build());

    nlohmann::json config = {{"a", {{"b", {{"c", {{"d", {{"value", 42.0}}}}}}}}}};
    auto result = deep_validator.validate(config);
    REQUIRE(result.is_valid());
  }
}
