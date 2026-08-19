// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include <openpfc/kernel/simulation/parameter_schema.hpp>
#include <openpfc/kernel/simulation/physics_concepts.hpp>

using json = nlohmann::json;
using pfc::sim::HasParameterSchema;
using pfc::sim::ParameterSchema;
using pfc::sim::SchemaSpec;
using pfc::sim::schema_config_error;
using pfc::sim::schema_json_value_string;

namespace {

struct SHParams {
  double epsilon{0.0};
  double r0{1.0};
  int seed{0};
};

ParameterSchema<SHParams> make_sh_schema() {
  ParameterSchema<SHParams> schema;
  schema.model_name("SwiftHohenberg")
      .real(&SHParams::epsilon,
            SchemaSpec{.name = "epsilon",
                       .description = "reduced undercooling",
                       .required = true,
                       .min = 0.0,
                       .max = 1.0,
                       .typical = 0.3,
                       .default_value = {},
                       .units = {},
                       .example = "\"epsilon\": 0.3"})
      .real(&SHParams::r0, SchemaSpec{.name = "r0",
                                      .description = "peak wavenumber",
                                      .required = false,
                                      .min = 0.0,
                                      .default_value = 1.0,
                                      .units = "1"})
      .integer(&SHParams::seed, SchemaSpec{.name = "seed",
                                           .description = "RNG seed",
                                           .required = false,
                                           .min = 0.0,
                                           .default_value = 0.0});
  return schema;
}

struct SHPhysics {
  using parameters_type = SHParams;
  static ParameterSchema<SHParams> schema() { return make_sh_schema(); }
};

} // namespace

static_assert(HasParameterSchema<SHPhysics>);
static_assert(pfc::sim::HasParameters<SHPhysics>);

TEST_CASE("schema_config_error matches format_config_error snapshots",
          "[parameter_schema][unit]") {
  const auto missing = schema_config_error(
      "Lx", "number of grid points in X direction", "positive integer",
      "missing");
  REQUIRE(missing.find("Field 'Lx' is missing") != std::string::npos);
  REQUIRE(missing.find("positive integer") != std::string::npos);
  REQUIRE(missing.find("number of grid points") != std::string::npos);

  const auto invalid = schema_config_error(
      "Lx", "number of grid points", "integer", "256.5 (type: float)");
  REQUIRE(invalid.find("has invalid value") != std::string::npos);
  REQUIRE(invalid.find("256.5") != std::string::npos);
  REQUIRE(invalid.find("integer") != std::string::npos);

  const auto with_example = schema_config_error(
      "Lx", "grid points", "integer", "missing", "\"Lx\": 256");
  REQUIRE(with_example.find("Example: \"Lx\": 256") != std::string::npos);

  const auto n_newlines =
      std::count(missing.begin(), missing.end(), '\n');
  REQUIRE(1 + static_cast<int>(n_newlines) <= 10);
}

TEST_CASE("schema_json_value_string reports missing",
          "[parameter_schema][unit]") {
  json j = {{"foo", 42}};
  REQUIRE(schema_json_value_string(j, "bar") == "missing");
}

TEST_CASE("ParameterSchema parse round-trip", "[parameter_schema][unit]") {
  const auto schema = make_sh_schema();
  const json config = {{"epsilon", 0.25}, {"r0", 1.5}, {"seed", 7}};
  const SHParams p = schema.parse(config);
  REQUIRE(p.epsilon == Catch::Approx(0.25));
  REQUIRE(p.r0 == Catch::Approx(1.5));
  REQUIRE(p.seed == 7);
}

TEST_CASE("ParameterSchema applies defaults for missing optional keys",
          "[parameter_schema][unit]") {
  const auto schema = make_sh_schema();
  const json config = {{"epsilon", 0.1}};
  const SHParams p = schema.parse(config);
  REQUIRE(p.epsilon == Catch::Approx(0.1));
  REQUIRE(p.r0 == Catch::Approx(1.0));
  REQUIRE(p.seed == 0);
  const auto result = schema.validate(config);
  REQUIRE(result.is_valid());
  REQUIRE(result.validated_params.at("r0").find("default") !=
          std::string::npos);
}

TEST_CASE("ParameterSchema missing required key matches config-error quality",
          "[parameter_schema][unit]") {
  const auto schema = make_sh_schema();
  const json config = {};
  const auto result = schema.validate(config);
  REQUIRE_FALSE(result.is_valid());
  REQUIRE(result.errors.size() == 1);
  REQUIRE(result.errors.front().find("Field 'epsilon' is missing") !=
          std::string::npos);
  REQUIRE(result.errors.front().find("reduced undercooling") !=
          std::string::npos);
  REQUIRE(result.errors.front().find("Expected: number") != std::string::npos);
  REQUIRE(result.errors.front().find("Got: missing") != std::string::npos);
  REQUIRE_THROWS_AS(schema.parse(config), std::invalid_argument);
}

TEST_CASE("ParameterSchema wrong type matches config-error quality",
          "[parameter_schema][unit]") {
  const auto schema = make_sh_schema();
  const json config = {{"epsilon", "hot"}};
  const auto result = schema.validate(config);
  REQUIRE_FALSE(result.is_valid());
  REQUIRE(result.errors.front().find("has invalid value") != std::string::npos);
  REQUIRE(result.errors.front().find("Expected: number") != std::string::npos);
  REQUIRE(result.errors.front().find("hot") != std::string::npos);
}

TEST_CASE("ParameterSchema range errors name the parameter",
          "[parameter_schema][unit]") {
  const auto schema = make_sh_schema();
  const auto low = schema.validate({{"epsilon", -0.1}});
  REQUIRE_FALSE(low.is_valid());
  REQUIRE(low.errors.front().find("below minimum") != std::string::npos);
  REQUIRE(low.errors.front().find("epsilon") != std::string::npos);

  const auto high = schema.validate({{"epsilon", 2.0}});
  REQUIRE_FALSE(high.is_valid());
  REQUIRE(high.errors.front().find("exceeds maximum") != std::string::npos);
}

TEST_CASE("ParameterSchema integer rejects floats",
          "[parameter_schema][unit]") {
  const auto schema = make_sh_schema();
  const auto result = schema.validate({{"epsilon", 0.2}, {"seed", 1.5}});
  REQUIRE_FALSE(result.is_valid());
  REQUIRE(result.errors.front().find("Expected: integer") != std::string::npos);
}

TEST_CASE("ParameterSchema docs_table lists bound fields",
          "[parameter_schema][unit]") {
  const auto table = make_sh_schema().docs_table();
  REQUIRE(table.find("| epsilon |") != std::string::npos);
  REQUIRE(table.find("reduced undercooling") != std::string::npos);
  REQUIRE(table.find("| seed |") != std::string::npos);
}
