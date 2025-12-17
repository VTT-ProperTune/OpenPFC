// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/ui/parameter_metadata.hpp"
#include <catch2/catch_all.hpp>

using namespace pfc::ui;

TEST_CASE("ParameterMetadata double validation", "[parameter_metadata][unit]") {
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

TEST_CASE("ParameterMetadata integer validation", "[parameter_metadata][unit]") {
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

TEST_CASE("ParameterMetadata format_info", "[parameter_metadata][unit]") {
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
