// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file precision_json.hpp
 * @brief The session-level `model.precision` key (ADR 0006).
 *
 * A spectral session's field scalar is a compile-time choice
 * (`Physics::real_type`), so an app that offers both precisions builds two
 * session types and selects one from the input file. `precision_from_json`
 * reads `model.precision` (`"double"`, the default when the key is absent,
 * or `"float"`) and fails closed on anything else; a session checks that the
 * key, when present, names its own scalar.
 */

#include <stdexcept>
#include <string>
#include <type_traits>

#include <nlohmann/json.hpp>

namespace pfc::ui {

/// Field scalar requested by an input file.
enum class Precision { Double, Float };

[[nodiscard]] inline const char *to_string(Precision p) noexcept {
  return p == Precision::Float ? "float" : "double";
}

/// `Precision` of a scalar type (`double` or `float`).
template <class Real> [[nodiscard]] constexpr Precision precision_of() noexcept {
  static_assert(std::is_same_v<Real, double> || std::is_same_v<Real, float>,
                "precision_of: Real must be double or float");
  return std::is_same_v<Real, float> ? Precision::Float : Precision::Double;
}

/// True when @p settings carries `model.precision` (whatever its value).
[[nodiscard]] inline bool has_precision_key(const nlohmann::json &settings) {
  return settings.is_object() && settings.contains("model") &&
         settings["model"].is_object() && settings["model"].contains("precision");
}

/**
 * @brief Read `model.precision` from @p settings.
 *
 * @return `Precision::Double` when the key is absent, otherwise the named
 *         precision.
 * @throws std::invalid_argument when the key is present but is not the
 *         string `"double"` or `"float"`.
 */
[[nodiscard]] inline Precision precision_from_json(const nlohmann::json &settings) {
  if (!has_precision_key(settings)) {
    return Precision::Double;
  }
  const nlohmann::json &value = settings["model"]["precision"];
  if (!value.is_string()) {
    throw std::invalid_argument(
        "model.precision must be the string \"double\" or \"float\"");
  }
  const std::string name = value.get<std::string>();
  if (name == "double") {
    return Precision::Double;
  }
  if (name == "float") {
    return Precision::Float;
  }
  throw std::invalid_argument("model.precision '" + name +
                              "' is not supported; use \"double\" or \"float\"");
}

} // namespace pfc::ui
