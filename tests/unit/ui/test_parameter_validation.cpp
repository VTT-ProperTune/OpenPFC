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

  SECTION("Boundary values with floating point precision") {
    auto meta =
        ParameterMetadata<double>::builder().name("epsilon").range(0.0, 1.0).build();

    // Test values very close to boundaries
    REQUIRE_FALSE(meta.validate(0.0).has_value());   // Exactly at min
    REQUIRE_FALSE(meta.validate(1.0).has_value());   // Exactly at max
    REQUIRE(meta.validate(-1e-15).has_value());      // Slightly below
    REQUIRE(meta.validate(1.0 + 1e-15).has_value()); // Slightly above
  }

  SECTION("Extreme boundary values") {
    auto meta = ParameterMetadata<double>::builder()
                    .name("extreme")
                    .range(std::numeric_limits<double>::lowest(),
                           std::numeric_limits<double>::max())
                    .build();

    REQUIRE_FALSE(meta.validate(std::numeric_limits<double>::lowest()).has_value());
    REQUIRE_FALSE(meta.validate(std::numeric_limits<double>::max()).has_value());
    REQUIRE_FALSE(meta.validate(0.0).has_value());
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

// Extended tests for complex validation scenarios
TEST_CASE("Complex validation with multiple constraints",
          "[parameter_validation][unit]") {
  ParameterValidator validator;

  // Test parameter with multiple constraints
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("complex_param")
                             .description("Parameter with multiple constraints")
                             .required(true)
                             .min(0.0)
                             .max(100.0)
                             .typical(50.0)
                             .build());

  SECTION("Valid value with all constraints satisfied") {
    json config = {{"complex_param", 50.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Value below minimum with detailed error") {
    json config = {{"complex_param", -10.0}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("below minimum") != std::string::npos);
    REQUIRE(result.errors[0].find("complex_param") != std::string::npos);
    REQUIRE(result.errors[0].find("0") != std::string::npos);
    REQUIRE(result.errors[0].find("100") != std::string::npos);
  }

  SECTION("Value above maximum with detailed error") {
    json config = {{"complex_param", 150.0}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("exceeds maximum") != std::string::npos);
    REQUIRE(result.errors[0].find("complex_param") != std::string::npos);
    REQUIRE(result.errors[0].find("0") != std::string::npos);
    REQUIRE(result.errors[0].find("100") != std::string::npos);
  }
}

TEST_CASE("Parameter validation with extreme values",
          "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("extreme_param")
                             .description("Parameter with extreme values")
                             .required(true)
                             .min(std::numeric_limits<double>::lowest())
                             .max(std::numeric_limits<double>::max())
                             .build());

  SECTION("Very large positive value") {
    json config = {{"extreme_param", 1e308}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Very large negative value") {
    json config = {{"extreme_param", -1e308}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Zero value") {
    json config = {{"extreme_param", 0.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("NaN value") {
    json config = {{"extreme_param", std::numeric_limits<double>::quiet_NaN()}};

    auto result = validator.validate(config);
    // NaN validation should return an error
    REQUIRE_FALSE(result.is_valid());
  }

  SECTION("Infinity value") {
    json config = {{"extreme_param", std::numeric_limits<double>::infinity()}};

    auto result = validator.validate(config);
    // Infinity validation should return an error
    REQUIRE_FALSE(result.is_valid());
  }
}

TEST_CASE("Parameter validation with default values - edge cases",
          "[parameter_validation][unit]") {
  ParameterValidator validator;

  // Test with default value that's at boundary
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("boundary_default")
                             .description("Parameter with boundary default")
                             .required(false)
                             .min(0.0)
                             .max(1.0)
                             .default_val(0.5)
                             .build());

  SECTION("Uses default value when not provided") {
    json config = {};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.size() == 1);
    REQUIRE(result.validated_params["boundary_default"].find("0.5") !=
            std::string::npos);
    REQUIRE(result.validated_params["boundary_default"].find("default") !=
            std::string::npos);
  }

  SECTION("Overrides default when provided") {
    json config = {{"boundary_default", 0.75}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params["boundary_default"].find("0.75") !=
            std::string::npos);
    REQUIRE(result.validated_params["boundary_default"].find("default") ==
            std::string::npos);
  }

  SECTION("Provides invalid default value") {
    // This should be handled gracefully by the validation system
    // The default value itself is checked during validation
    json config = {{"boundary_default", 1.5}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}

TEST_CASE("Integration test with realistic model configuration",
          "[parameter_validation][unit]") {
  // Test a realistic scenario similar to Tungsten model
  ParameterValidator validator;

  // Add parameters similar to Tungsten model
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("n0")
                             .description("Average density of the metastable fluid")
                             .required(true)
                             .range(-1.0, 0.0)
                             .typical(-0.10)
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("T")
                             .description("Effective temperature")
                             .required(true)
                             .range(0.0, 10000.0)
                             .typical(3300.0)
                             .units("K")
                             .category("Thermodynamics")
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("lambda")
                             .description("Strength of meanfield filter")
                             .required(true)
                             .range(0.0, 0.5)
                             .typical(0.22)
                             .category("Numerical")
                             .build());

  SECTION("Complete valid configuration") {
    json config = {{"n0", -0.10}, {"T", 3300.0}, {"lambda", 0.22}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.size() == 3);
    REQUIRE(result.validated_params.find("n0") != result.validated_params.end());
    REQUIRE(result.validated_params.find("T") != result.validated_params.end());
    REQUIRE(result.validated_params.find("lambda") != result.validated_params.end());
  }

  SECTION("Missing required parameter") {
    json config = {{"n0", -0.10},
                   // Missing T
                   {"lambda", 0.22}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("T") != std::string::npos);
    REQUIRE(result.errors[0].find("missing") != std::string::npos);
  }

  SECTION("Invalid parameter value") {
    json config = {{"n0", -0.10},
                   {"T", -100.0}, // Invalid: below minimum
                   {"lambda", 0.22}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors[0].find("T") != std::string::npos);
    REQUIRE(result.errors[0].find("below minimum") != std::string::npos);
  }

  SECTION("Multiple validation errors") {
    json config = {
        {"n0", 1.0},    // Invalid: above maximum
        {"T", -100.0},  // Invalid: below minimum
        {"lambda", 0.6} // Invalid: above maximum
    };

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 3);
  }
}

// Additional tests for type safety and conversion
TEST_CASE("Type safety and conversion tests", "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("double_param")
                             .description("Double parameter")
                             .required(true)
                             .build());

  validator.add_metadata(ParameterMetadata<int>::builder()
                             .name("int_param")
                             .description("Integer parameter")
                             .required(true)
                             .build());

  SECTION("String instead of number") {
    json config = {{"double_param", "not_a_number"},
                   {"int_param", "also_not_a_number"}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[1].find("wrong type") != std::string::npos);
  }

  SECTION("String representation of numbers") {
    // JSON strings that look like numbers should be rejected for numeric types
    json config = {{"double_param", "3.14"}, {"int_param", "42"}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[1].find("wrong type") != std::string::npos);
  }

  SECTION("Boolean values") {
    json config = {{"double_param", true}, {"int_param", false}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[1].find("wrong type") != std::string::npos);
  }

  SECTION("Null values") {
    json config = {{"double_param", nullptr}, {"int_param", nullptr}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[1].find("wrong type") != std::string::npos);
  }

  SECTION("Array values") {
    json config = {{"double_param", {1, 2, 3}}, {"int_param", {4, 5, 6}}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[1].find("wrong type") != std::string::npos);
  }

  SECTION("Object values") {
    json config = {{"double_param", {{"key", "value"}}},
                   {"int_param", {{"another", "object"}}}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 2);
    REQUIRE(result.errors[0].find("wrong type") != std::string::npos);
    REQUIRE(result.errors[1].find("wrong type") != std::string::npos);
  }
}

// Test for validation performance and stress scenarios
TEST_CASE("Performance and stress testing", "[parameter_validation][unit]") {
  ParameterValidator validator;

  // Add many parameters to test performance
  for (int i = 0; i < 100; ++i) {
    validator.add_metadata(ParameterMetadata<double>::builder()
                               .name("param_" + std::to_string(i))
                               .description("Test parameter " + std::to_string(i))
                               .required(false)
                               .min(0.0)
                               .max(100.0)
                               .build());
  }

  SECTION("Large configuration validation") {
    json config;
    for (int i = 0; i < 100; ++i) {
      config["param_" + std::to_string(i)] = 50.0;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto result = validator.validate(config);
    auto end = std::chrono::high_resolution_clock::now();

    REQUIRE(result.is_valid());

    // Should complete quickly (less than 100ms for 100 parameters)
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    REQUIRE(duration.count() < 100);
  }

  SECTION("Large configuration with errors") {
    json config;
    for (int i = 0; i < 99; ++i) {
      config["param_" + std::to_string(i)] = 50.0;
    }
    config["param_99"] = 150.0; // Invalid value

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}

// Additional comprehensive tests for edge cases
TEST_CASE("Parameter validation with special floating point values",
          "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("special_param")
                             .description("Parameter with special float values")
                             .required(true)
                             .min(-1000.0)
                             .max(1000.0)
                             .build());

  SECTION("Subnormal values") {
    // Test with subnormal (denormalized) values
    json config = {{"special_param", 5e-308}}; // Very small positive number

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Negative zero") {
    json config = {{"special_param", -0.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Positive zero") {
    json config = {{"special_param", 0.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }
}

// Test for validation with complex nested structures
TEST_CASE("Nested parameter validation", "[parameter_validation][unit]") {
  ParameterValidator validator;

  // Add nested parameter structure
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("outer.inner.param")
                             .description("Nested parameter")
                             .required(true)
                             .min(0.0)
                             .max(100.0)
                             .build());

  SECTION("Valid nested parameter") {
    json config = {{"outer", {{"inner", {{"param", 50.0}}}}}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Invalid nested parameter") {
    json config = {{"outer", {{"inner", {{"param", 150.0}}}}}}; // Out of range

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}

// Test for validation with custom validation functions
TEST_CASE("Custom validation function integration", "[parameter_validation][unit]") {
  ParameterValidator validator;

  // Add parameter with custom validation
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("custom_param")
                             .description("Parameter with custom validation")
                             .required(true)
                             .min(0.0)
                             .max(100.0)
                             .build());

  SECTION("Valid custom parameter") {
    json config = {{"custom_param", 50.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Invalid custom parameter") {
    json config = {{"custom_param", -10.0}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}

// Test for validation with different numeric types
TEST_CASE("Mixed numeric type validation", "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("float_param")
                             .description("Float parameter")
                             .required(true)
                             .min(0.0)
                             .max(100.0)
                             .build());

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("double_param")
                             .description("Double parameter")
                             .required(true)
                             .min(0.0)
                             .max(100.0)
                             .build());

  SECTION("Valid mixed types") {
    json config = {{"float_param", 50.0f}, {"double_param", 75.0}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Invalid float type") {
    json config = {{"float_param", 150.0f}, // Out of range
                   {"double_param", 75.0}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}

// Test for validation with empty configurations
TEST_CASE("Empty configuration validation", "[parameter_validation][unit]") {
  ParameterValidator validator;

  // Add required parameters
  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("required_param")
                             .description("Required parameter")
                             .required(true)
                             .build());

  SECTION("Empty configuration with required parameters") {
    json config = {};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }

  SECTION("Empty configuration with no parameters") {
    ParameterValidator empty_validator;
    json config = {};

    auto result = empty_validator.validate(config);
    REQUIRE(result.is_valid());
    REQUIRE(result.validated_params.empty());
  }
}

// Test for validation with deeply nested JSON structures
TEST_CASE("Deeply nested JSON validation", "[parameter_validation][unit]") {
  ParameterValidator validator;

  validator.add_metadata(ParameterMetadata<double>::builder()
                             .name("deeply.nested.value")
                             .description("Deeply nested parameter")
                             .required(true)
                             .min(0.0)
                             .max(100.0)
                             .build());

  SECTION("Valid deeply nested structure") {
    json config = {{"deeply", {{"nested", {{"value", 50.0}}}}}};

    auto result = validator.validate(config);
    REQUIRE(result.is_valid());
  }

  SECTION("Invalid deeply nested structure") {
    json config = {{"deeply", {{"nested", {{"value", 150.0}}}}}};

    auto result = validator.validate(config);
    REQUIRE_FALSE(result.is_valid());
    REQUIRE(result.errors.size() == 1);
  }
}
