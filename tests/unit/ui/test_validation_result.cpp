// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/ui/parameter_validator.hpp"
#include <catch2/catch_all.hpp>

using namespace pfc::ui;

TEST_CASE("ValidationResult error formatting", "[validation_result][unit]") {
  ValidationResult result;

  SECTION("Single error") {
    result.errors.push_back("temperature: Value below minimum (0)");
    result.valid = false;
    REQUIRE_FALSE(result.is_valid());

    std::string errors = result.format_errors();
    REQUIRE(errors.find("temperature") != std::string::npos);
    REQUIRE(errors.find("below minimum") != std::string::npos);
  }

  SECTION("Multiple errors") {
    result.errors.push_back("temperature: Value below minimum");
    result.errors.push_back("mobility: Value exceeds maximum");
    result.valid = false;
    REQUIRE_FALSE(result.is_valid());

    std::string errors = result.format_errors();
    REQUIRE(errors.find("temperature") != std::string::npos);
    REQUIRE(errors.find("mobility") != std::string::npos);
  }

  SECTION("No errors") {
    REQUIRE(result.is_valid());
    std::string errors = result.format_errors();
    REQUIRE(errors.find("No errors") != std::string::npos);
  }
}

TEST_CASE("ValidationResult summary", "[validation_result][unit]") {
  ValidationResult result;

  SECTION("Valid result summary") {
    std::string summary = result.format_summary();
    REQUIRE(summary.find("Validation Summary") != std::string::npos);
  }

  SECTION("With validated parameters") {
    result.validated_params["temperature"] = "3300.0";
    result.validated_params["mobility"] = "50.0";

    std::string summary = result.format_summary();
    REQUIRE(summary.find("2 parameter") != std::string::npos);
    REQUIRE(summary.find("temperature") != std::string::npos);
    REQUIRE(summary.find("mobility") != std::string::npos);
  }
}
