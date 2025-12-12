// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file parameter_validator.hpp
 * @brief Parameter validation orchestration and reporting
 *
 * @details
 * This file provides the validation engine that checks all parameters against
 * their metadata and generates comprehensive error reports. It implements the
 * "fail fast" philosophy: catch configuration errors immediately at startup
 * rather than hours into a simulation.
 *
 * Key features:
 * - Validates all parameters in one pass
 * - Collects all errors (not just first one)
 * - Generates formatted validation reports
 * - Prints parameter summaries for reproducibility
 *
 * Usage:
 * @code
 * // Create validator with metadata
 * ParameterValidator validator;
 * validator.add_metadata(temp_metadata);
 * validator.add_metadata(density_metadata);
 *
 * // Validate configuration
 * auto result = validator.validate(json_config);
 * if (!result.is_valid()) {
 *   std::cerr << result.format_errors() << std::endl;
 *   throw std::invalid_argument("Invalid configuration");
 * }
 *
 * // Print summary
 * std::cout << result.format_summary() << std::endl;
 * @endcode
 *
 * @see parameter_metadata.hpp for individual parameter metadata
 * @see tungsten_input.hpp for usage example
 */

#ifndef PFC_UI_PARAMETER_VALIDATOR_HPP
#define PFC_UI_PARAMETER_VALIDATOR_HPP

#include "parameter_metadata.hpp"
#include <nlohmann/json.hpp>
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>

namespace pfc {
namespace ui {

using json = nlohmann::json;

/**
 * @brief Result of parameter validation
 *
 * Contains all validation errors and the validated values.
 */
struct ValidationResult {
  bool valid = true;
  std::vector<std::string> errors;
  std::map<std::string, std::string> validated_params;

  /**
   * @brief Check if validation passed
   */
  bool is_valid() const { return valid && errors.empty(); }

  /**
   * @brief Format error report
   */
  std::string format_errors() const {
    if (errors.empty()) {
      return "No errors";
    }

    std::ostringstream msg;
    msg << "\n" << std::string(80, '=') << "\n";
    msg << "CONFIGURATION VALIDATION FAILED\n";
    msg << std::string(80, '=') << "\n\n";
    msg << "Found " << errors.size() << " error(s):\n\n";

    for (size_t i = 0; i < errors.size(); ++i) {
      msg << (i + 1) << ". " << errors[i] << "\n\n";
    }

    msg << "ABORTING: Fix configuration errors before running simulation.\n";
    msg << std::string(80, '=') << "\n";

    return msg.str();
  }

  /**
   * @brief Format parameter summary
   */
  std::string format_summary(const std::string& model_name = "Model") const {
    std::ostringstream msg;
    msg << "\n" << std::string(80, '=') << "\n";
    msg << "Configuration Validation Summary - " << model_name << "\n";
    msg << std::string(80, '=') << "\n";

    if (validated_params.empty()) {
      msg << "No parameters validated\n";
    } else {
      msg << "Validated " << validated_params.size() << " parameter(s):\n";
      msg << std::string(80, '-') << "\n";

      // Find max key length for formatting
      size_t max_key_len = 0;
      for (const auto& [key, _] : validated_params) {
        max_key_len = std::max(max_key_len, key.length());
      }

      for (const auto& [key, value] : validated_params) {
        msg << "  " << std::left << std::setw(max_key_len + 2) << key 
            << " = " << value << "\n";
      }
    }

    msg << std::string(80, '=') << "\n";

    return msg.str();
  }
};

/**
 * @brief Parameter validator for double-valued parameters
 *
 * Orchestrates validation of all parameters against their metadata.
 */
class ParameterValidator {
private:
  std::vector<ParameterMetadata<double>> double_params_;
  std::vector<ParameterMetadata<int>> int_params_;
  std::string model_name_ = "Model";

public:
  /**
   * @brief Set model name for reporting
   */
  void set_model_name(const std::string& name) {
    model_name_ = name;
  }

  /**
   * @brief Add parameter metadata for validation
   */
  void add_metadata(const ParameterMetadata<double>& meta) {
    double_params_.push_back(meta);
  }

  /**
   * @brief Add integer parameter metadata for validation
   */
  void add_metadata(const ParameterMetadata<int>& meta) {
    int_params_.push_back(meta);
  }

  /**
   * @brief Validate parameters from JSON configuration
   *
   * Checks all registered parameters against the provided configuration.
   * Collects all errors rather than stopping at the first one.
   *
   * @param config JSON configuration object
   * @return ValidationResult with errors and validated values
   */
  ValidationResult validate(const json& config) const {
    ValidationResult result;

    // Validate double parameters
    for (const auto& meta : double_params_) {
      validate_parameter(config, meta, result);
    }

    // Validate int parameters
    for (const auto& meta : int_params_) {
      validate_parameter(config, meta, result);
    }

    result.valid = result.errors.empty();
    return result;
  }

private:
  /**
   * @brief Validate a single double parameter
   */
  void validate_parameter(const json& config, 
                         const ParameterMetadata<double>& meta,
                         ValidationResult& result) const {
    // Check if parameter exists
    if (!config.contains(meta.name)) {
      if (meta.required && !meta.default_value) {
        std::ostringstream err;
        err << "Required parameter '" << meta.name << "' is missing\n"
            << "  " << meta.format_info();
        result.errors.push_back(err.str());
        return;
      } else if (meta.default_value) {
        // Use default value
        std::ostringstream val_str;
        val_str << *meta.default_value << " (default)";
        result.validated_params[meta.name] = val_str.str();
        return;
      } else {
        // Optional parameter, not provided
        return;
      }
    }

    // Check type
    if (!config[meta.name].is_number()) {
      std::ostringstream err;
      err << "Parameter '" << meta.name << "' has wrong type\n"
          << "  Expected: number\n"
          << "  Got: " << config[meta.name].type_name() << "\n"
          << "  Value: " << config[meta.name].dump() << "\n"
          << "  " << meta.format_info();
      result.errors.push_back(err.str());
      return;
    }

    // Get value and validate bounds
    double value = config[meta.name].get<double>();
    if (auto error = meta.validate(value)) {
      std::ostringstream err;
      err << *error << "\n"
          << "  " << meta.format_info();
      result.errors.push_back(err.str());
    } else {
      // Valid parameter
      std::ostringstream val_str;
      val_str << value;
      if (meta.min_value && meta.max_value) {
        val_str << "  [range: " << *meta.min_value << ", " << *meta.max_value << "]";
      }
      result.validated_params[meta.name] = val_str.str();
    }
  }

  /**
   * @brief Validate a single int parameter
   */
  void validate_parameter(const json& config, 
                         const ParameterMetadata<int>& meta,
                         ValidationResult& result) const {
    // Check if parameter exists
    if (!config.contains(meta.name)) {
      if (meta.required && !meta.default_value) {
        std::ostringstream err;
        err << "Required parameter '" << meta.name << "' is missing\n"
            << "  " << meta.format_info();
        result.errors.push_back(err.str());
        return;
      } else if (meta.default_value) {
        // Use default value
        std::ostringstream val_str;
        val_str << *meta.default_value << " (default)";
        result.validated_params[meta.name] = val_str.str();
        return;
      } else {
        // Optional parameter, not provided
        return;
      }
    }

    // Check type
    if (!config[meta.name].is_number_integer()) {
      std::ostringstream err;
      err << "Parameter '" << meta.name << "' has wrong type\n"
          << "  Expected: integer\n"
          << "  Got: " << config[meta.name].type_name() << "\n"
          << "  Value: " << config[meta.name].dump() << "\n"
          << "  " << meta.format_info();
      result.errors.push_back(err.str());
      return;
    }

    // Get value and validate bounds
    int value = config[meta.name].get<int>();
    if (auto error = meta.validate(value)) {
      std::ostringstream err;
      err << *error << "\n"
          << "  " << meta.format_info();
      result.errors.push_back(err.str());
    } else {
      // Valid parameter
      std::ostringstream val_str;
      val_str << value;
      if (meta.min_value && meta.max_value) {
        val_str << "  [range: " << *meta.min_value << ", " << *meta.max_value << "]";
      }
      result.validated_params[meta.name] = val_str.str();
    }
  }
};

} // namespace ui
} // namespace pfc

#endif // PFC_UI_PARAMETER_VALIDATOR_HPP
