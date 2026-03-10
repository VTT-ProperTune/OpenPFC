// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file parameter_metadata.hpp
 * @brief Parameter metadata system for configuration validation
 *
 * @details
 * This file provides a metadata system for describing model parameters with
 * validation rules and documentation. It enables:
 * - Required parameter checking (fail if missing)
 * - Bounds validation (min/max constraints)
 * - Helpful error messages with descriptions and typical values
 * - Parameter summary printing for reproducibility
 *
 * The system prevents the OpenFOAM problem: missing parameters causing silent
 * failures hours into a simulation. Instead, validation fails immediately at
 * startup with clear, actionable error messages.
 *
 * Usage:
 * @code
 * // Define parameter metadata
 * auto params = ParameterMetadata<double>::builder()
 *   .name("temperature")
 *   .description("Effective temperature in Kelvin")
 *   .required(true)
 *   .min(0.0)
 *   .max(10000.0)
 *   .typical(3300.0)
 *   .build();
 *
 * // Validate a value
 * double T = 3300.0;
 * if (auto error = params.validate(T)) {
 *   std::cerr << error.value() << std::endl;
 *   throw std::invalid_argument("Invalid temperature");
 * }
 * @endcode
 *
 * @see parameter_validator.hpp for validation orchestration
 * @see tungsten_input.hpp for usage example
 */

#ifndef PFC_UI_PARAMETER_METADATA_HPP
#define PFC_UI_PARAMETER_METADATA_HPP

#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <limits>

namespace pfc {
namespace ui {

/**
 * @brief Metadata for a single model parameter
 *
 * Contains all information needed to validate, document, and provide helpful
 * error messages for a parameter.
 *
 * @tparam T Parameter type (double, int, etc.)
 */
template <typename T>
struct ParameterMetadata {
  std::string name;                        ///< Parameter name (as in config file)
  std::string description;                 ///< Human-readable description
  bool required = false;                   ///< Must be present in config?
  std::optional<T> min_value;             ///< Minimum valid value (inclusive)
  std::optional<T> max_value;             ///< Maximum valid value (inclusive)
  std::optional<T> typical_value;         ///< Typical/recommended value
  std::optional<T> default_value;         ///< Default if not specified
  std::string physical_units;              ///< Physical units (e.g., "K", "m/s")
  std::string category;                    ///< Parameter category (e.g., "Thermodynamics")

  /**
   * @brief Validate a parameter value against constraints
   *
   * Checks if value satisfies min/max bounds.
   *
   * @param value Value to validate
   * @return Error message if invalid, empty optional if valid
   */
  std::optional<std::string> validate(T value) const {
    if (min_value && value < *min_value) {
      std::ostringstream msg;
      msg << "Parameter '" << name << "' = " << value 
          << " is below minimum " << *min_value;
      if (min_value && max_value) {
        msg << " (valid range: [" << *min_value << ", " << *max_value << "])";
      }
      return msg.str();
    }

    if (max_value && value > *max_value) {
      std::ostringstream msg;
      msg << "Parameter '" << name << "' = " << value 
          << " exceeds maximum " << *max_value;
      if (min_value && max_value) {
        msg << " (valid range: [" << *min_value << ", " << *max_value << "])";
      }
      return msg.str();
    }

    return std::nullopt;  // Valid
  }

  /**
   * @brief Format parameter info for error messages
   *
   * @return Multi-line string describing the parameter
   */
  std::string format_info() const {
    std::ostringstream info;
    info << "Parameter: " << name << "\n";
    info << "  Description: " << description;
    if (!physical_units.empty()) {
      info << " (" << physical_units << ")";
    }
    info << "\n";
    
    if (min_value || max_value) {
      info << "  Valid range: [";
      if (min_value) {
        info << *min_value;
      } else {
        info << "-inf";
      }
      info << ", ";
      if (max_value) {
        info << *max_value;
      } else {
        info << "+inf";
      }
      info << "]\n";
    }
    
    if (typical_value) {
      info << "  Typical value: " << *typical_value << "\n";
    }
    
    if (default_value) {
      info << "  Default value: " << *default_value << "\n";
    }
    
    info << "  Required: " << (required ? "yes" : "no");
    
    return info.str();
  }

  /**
   * @brief Builder pattern for fluent parameter construction
   */
  class Builder {
  private:
    ParameterMetadata<T> param_;

  public:
    Builder& name(const std::string& n) {
      param_.name = n;
      return *this;
    }

    Builder& description(const std::string& desc) {
      param_.description = desc;
      return *this;
    }

    Builder& required(bool req = true) {
      param_.required = req;
      return *this;
    }

    Builder& optional(bool opt = true) {
      param_.required = !opt;
      return *this;
    }

    Builder& min(T min_val) {
      param_.min_value = min_val;
      return *this;
    }

    Builder& max(T max_val) {
      param_.max_value = max_val;
      return *this;
    }

    Builder& range(T min_val, T max_val) {
      param_.min_value = min_val;
      param_.max_value = max_val;
      return *this;
    }

    Builder& typical(T typ_val) {
      param_.typical_value = typ_val;
      return *this;
    }

    Builder& default_val(T def_val) {
      param_.default_value = def_val;
      return *this;
    }

    Builder& units(const std::string& u) {
      param_.physical_units = u;
      return *this;
    }

    Builder& category(const std::string& cat) {
      param_.category = cat;
      return *this;
    }

    ParameterMetadata<T> build() const {
      return param_;
    }
  };

  /**
   * @brief Start building parameter metadata
   */
  static Builder builder() {
    return Builder();
  }
};

} // namespace ui
} // namespace pfc

#endif // PFC_UI_PARAMETER_METADATA_HPP
