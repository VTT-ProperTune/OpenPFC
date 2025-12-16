// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/ui/parameter_metadata.hpp"
#include "openpfc/ui/parameter_validator.hpp"
#include <catch2/catch_all.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace pfc::ui;

TEST_CASE("ParameterMetadata double validation", "[parameter_validation][unit]") {
  SECTION("Valid value within range") {
    auto meta = ParameterMetadata<double>::builder()
                    .name("temperature")
                    .description("Temperature in Kelvin")
                    .min(0.0)
                    .max(10000.0)
                    .build();

    auto error = meta.validate(3300.0);
    REQUIRE_FALSE(error.has_value());
  }

  SECTION("Value below minimum") {
    auto meta =
        ParameterMetadata<double>::builder().name("temperature").min(0.0).build();

    auto error = meta.validate(-100.0);
    REQUIRE(error.has_value());
    REQUIRE(error->find("below minimum") != std::string::npos);
    REQUIRE(error->find("temperature") != std::string::npos);
  }

  SECTION("Value above maximum") {
    auto meta = ParameterMetadata<double>::builder()
                    .name("temperature")
                    .max(10000.0)
                    .build();

    auto error = meta.validate(15000.0);
    REQUIRE(error.has_value());
    REQUIRE(error->find("exceeds maximum") != std::string::npos);
    REQUIRE(error->find("temperature") != std::string::npos);
  }

  SECTION("Value exactly at boundary") {
    auto meta = ParameterMetadata<double>::builder()
                    .name("density")
                    .range(-1.0, 0.0)
                    .build();

    REQUIRE_FALSE(meta.validate(-1.0).has_value());
    REQUIRE_FALSE(meta.validate(0.0).has_value());
    REQUIRE(meta.validate(-1.1).has_value());
    REQUIRE(meta.validate(0.1).has_value());
  }
}

TEST_CASE("ParameterMetadata integer validation", "[parameter_validation][unit]") {
  SECTION("Valid integer value") {
    auto meta =
        ParameterMetadata<int>::builder().name("num_steps").range(1, 1000).build();

    REQUIRE_FALSE(meta.validate(100).has_value());
  }

  SECTION("Integer out of range") {
    auto meta =
        ParameterMetadata<int>::builder().name("num_steps").range(1, 1000).build();

    REQUIRE(meta.validate(0).has_value());
    REQUIRE(meta.validate(1001).has_value());
  }
}

TEST_CASE("ParameterMetadata format_info", "[parameter_validation][unit]") {
  auto meta = ParameterMetadata<double>::builder()
                  .name("temperature")
                  .description("Effective temperature")
                  .required(true)
                  .range(0.0, 10000.0)
                  .typical(3300.0)
                  .units("K")
                  .category("Thermodynamics")
                  .build();

  std::string info = meta.format_info();

  REQUIRE(info.find("temperature") != std::string::npos);
  REQUIRE(info.find("Effective temperature") != std::string::npos);
  REQUIRE(info.find("K") != std::string::npos);
  REQUIRE(info.find("0") != std::string::npos);
  REQUIRE(info.find("10000") != std::string::npos);
  REQUIRE(info.find("3300") != std::string::npos);
  REQUIRE(info.find("yes") != std::string::npos);
}

TEST_CASE("ParameterValidator with required parameters",
          "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("temperature")
                             .description("Temperature in Kelvin")
                             .required(true)
                             .range(0.0, 10000.0)
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("density")
                             .description("Material density")
                             .required(true)
                             .range(-1.0, 0.0)
                             .build());

  SECTION("All required parameters present and valid") {
    json config = {{"temperature", 3300.0}, {"density", -0.10}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.errors.empty());
    REQUIRE(result.validated_params.size() == 2);
  }

  SECTION("Missing required parameter") {
    json config = {
        {"temperature", 3300.0}
        // density is missing
    };

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("density") != std::string::npos);
    REQUIRE(result.errors[0].find("missing") != std::string::npos);
  }

  SECTION("Parameter out of bounds") {
    json config = {{"temperature", -100.0}, // Invalid: below minimum
                   {"density", -0.10}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("temperature") != std::string::npos);
    REQUIRE(result.errors[0].find("below minimum") != std::string::npos);
  }

  SECTION("Multiple errors collected") {
    json config = {
        {"temperature", -100.0}, // Invalid: below minimum
        {"density", 1.0}         // Invalid: above maximum
    };

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
  }

  SECTION("Wrong type") {
    json config = {{"temperature", "3300"}, // String instead of number
                   {"density", -0.10}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
  }
}

TEST_CASE("ParameterValidator with optional parameters",
          "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("temperature")
                             .description("Temperature")
                             .required(true)
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("optional_param")
                             .description("Optional parameter")
                             .required(false)
                             .range(0.0, 1.0)
                             .build());

  SECTION("Optional parameter not provided is OK") {
    json config = {
        {"temperature", 300.0}
        // optional_param not provided
    };

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.size() == 1);
  }

  SECTION("Optional parameter provided and valid") {
    json config = {{"temperature", 300.0}, {"optional_param", 0.5}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.size() == 2);
  }

  SECTION("Optional parameter provided but invalid") {
    json config = {
        {"temperature", 300.0}, {"optional_param", 2.0} // Out of range
    };

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}

TEST_CASE("ParameterValidator with default values", "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("param_with_default")
                             .description("Parameter with default value")
                             .required(false)
                             .default_val(42.0)
                             .build());

  SECTION("Uses default when not provided") {
    json config = {};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.size() == 1);
    REQUIRE(result.validated_params["param_with_default"].find("42") !=
            std::string::npos);
    REQUIRE(result.validated_params["param_with_default"].find("default") !=
            std::string::npos);
  }

  SECTION("Overrides default when provided") {
    json config = {{"param_with_default", 100.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params["param_with_default"].find("100") !=
            std::string::npos);
    REQUIRE(result.validated_params["param_with_default"].find("default") ==
            std::string::npos);
  }
}

TEST_CASE("ValidationResult error formatting", "[parameter_validation][unit]") {
  ValidationResult result;
  result.valid = false;
  result.errors.push_back("Error 1: Missing parameter 'x'");
  result.errors.push_back("Error 2: Parameter 'y' out of range");

  std::string formatted = result.format_errors();

  REQUIRE(formatted.find("VALIDATION FAILED") != std::string::npos);
  REQUIRE(formatted.find("Found 2 error(s)") != std::string::npos);
  REQUIRE(formatted.find("Error 1") != std::string::npos);
  REQUIRE(formatted.find("Error 2") != std::string::npos);
  REQUIRE(formatted.find("ABORTING") != std::string::npos);
}

TEST_CASE("ValidationResult summary formatting", "[parameter_validation][unit]") {
  ValidationResult result;
  result.valid = true;
  result.validated_params["temperature"] = "3300.0  [range: 0, 10000]";
  result.validated_params["density"] = "-0.10  [range: -1, 0]";

  std::string summary = result.format_summary("Test Model");

  REQUIRE(summary.find("Test Model") != std::string::npos);
  REQUIRE(summary.find("Validated 2 parameter(s)") != std::string::npos);
  REQUIRE(summary.find("temperature") != std::string::npos);
  REQUIRE(summary.find("density") != std::string::npos);
  REQUIRE(summary.find("3300") != std::string::npos);
  REQUIRE(summary.find("-0.10") != std::string::npos);
}

TEST_CASE("Integer parameter validation", "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<int>::builder()
                             .name("num_iterations")
                             .description("Number of iterations")
                             .required(true)
                             .range(1, 10000)
                             .typical(1000)
                             .build());

  SECTION("Valid integer") {
    json config = {{"num_iterations", 100}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Float instead of integer") {
    json config = {{"num_iterations", 100.5}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[0].find("integer") != std::string::npos);
  }

  SECTION("Integer out of range") {
    json config = {{"num_iterations", 0}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors[0].find("below minimum") != std::string::npos);
  }
}
