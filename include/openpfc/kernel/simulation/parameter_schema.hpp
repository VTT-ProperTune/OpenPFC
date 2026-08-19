// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file parameter_schema.hpp
 * @brief Declarative parameter list: validate, parse, and document one model.
 *
 * @details
 * One schema per physics model (not per backend). Member-pointer bindings
 * generate `from_json` into the model's `parameters_type`. Validation
 * messages match `pfc::ui::format_config_error` for missing / wrong-type
 * keys so frontend snapshots stay comparable. This header lives in the
 * kernel and does not include the frontend `ParameterValidator`.
 *
 * @see physics_concepts.hpp `HasParameters`
 */

#include <cmath>
#include <concepts>
#include <functional>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/simulation/physics_concepts.hpp>

namespace pfc::sim {

/**
 * @brief One schema field (name, bounds, docs). Type is chosen by
 *        `ParameterSchema::real` vs `integer`.
 */
struct SchemaSpec {
  std::string name;
  std::string description;
  bool required{true};
  std::optional<double> min{};
  std::optional<double> max{};
  std::optional<double> typical{};
  std::optional<double> default_value{};
  std::string units{};
  std::string example{};
};

/**
 * @brief Collected validation errors and the values that passed.
 */
struct SchemaValidationResult {
  std::vector<std::string> errors;
  std::map<std::string, std::string> validated_params;

  [[nodiscard]] bool is_valid() const { return errors.empty(); }

  [[nodiscard]] std::string format_errors() const {
    if (errors.empty()) {
      return "No errors";
    }
    std::ostringstream msg;
    msg << "\n" << std::string(80, '=') << "\n";
    msg << "CONFIGURATION VALIDATION FAILED\n";
    msg << std::string(80, '=') << "\n\n";
    msg << "Found " << errors.size() << " error(s):\n\n";
    for (std::size_t i = 0; i < errors.size(); ++i) {
      msg << (i + 1) << ". " << errors[i] << "\n\n";
    }
    msg << "ABORTING: Fix configuration errors before running simulation.\n";
    msg << std::string(80, '=') << "\n";
    return msg.str();
  }
};

/**
 * @brief Error text matching `pfc::ui::format_config_error`.
 */
[[nodiscard]] inline std::string
schema_config_error(const std::string &field_name,
                    const std::string &description,
                    const std::string &expected_type,
                    const std::string &actual_value,
                    const std::string &example = {}) {
  std::ostringstream oss;
  oss << "Invalid configuration: Field '" << field_name << "' ";
  if (actual_value == "missing") {
    oss << "is missing.\n";
  } else {
    oss << "has invalid value.\n";
  }
  oss << "  Description: " << description << "\n";
  oss << "  Expected: " << expected_type << "\n";
  oss << "  Got: " << actual_value << "\n";
  if (!example.empty()) {
    oss << "  Example: " << example;
  }
  return oss.str();
}

[[nodiscard]] inline std::string
schema_json_value_string(const nlohmann::json &j, const std::string &field) {
  if (!j.contains(field)) {
    return "missing";
  }
  const auto &value = j[field];
  if (value.is_string()) {
    return std::string("\"") + value.get<std::string>() + "\"";
  }
  return value.dump() + " (type: " + std::string(value.type_name()) + ")";
}

/**
 * @brief Declarative schema bound to a parameter struct @p Params.
 *
 * @tparam Params Default-constructible aggregate / struct of scalars.
 */
template <class Params> class ParameterSchema {
public:
  using parameters_type = Params;

  ParameterSchema &model_name(std::string name) {
    m_model_name = std::move(name);
    return *this;
  }

  [[nodiscard]] const std::string &model_name() const { return m_model_name; }

  ParameterSchema &real(double Params::*member, SchemaSpec spec) {
    add_field(member, std::move(spec), /*integer=*/false);
    return *this;
  }

  ParameterSchema &integer(int Params::*member, SchemaSpec spec) {
    add_field(member, std::move(spec), /*integer=*/true);
    return *this;
  }

  [[nodiscard]] SchemaValidationResult
  validate(const nlohmann::json &config) const {
    SchemaValidationResult result;
    Params scratch{};
    apply(config, scratch, result);
    return result;
  }

  /**
   * @brief Parse JSON into @p Params, applying defaults.
   * @throws std::invalid_argument if validation fails.
   */
  [[nodiscard]] Params parse(const nlohmann::json &config) const {
    SchemaValidationResult result;
    Params out{};
    apply(config, out, result);
    if (!result.is_valid()) {
      throw std::invalid_argument(result.format_errors());
    }
    return out;
  }

  [[nodiscard]] std::string docs_table() const {
    std::ostringstream oss;
    oss << "| Name | Type | Required | Range | Typical | Units | "
           "Description |\n";
    oss << "|------|------|----------|-------|---------|-------|"
           "-------------|\n";
    for (const auto &f : m_fields) {
      oss << "| " << f.spec.name << " | " << f.expected_type << " | "
          << (f.spec.required ? "yes" : "no") << " | ";
      if (f.spec.min || f.spec.max) {
        oss << "[";
        if (f.spec.min) {
          oss << *f.spec.min;
        } else {
          oss << "-inf";
        }
        oss << ", ";
        if (f.spec.max) {
          oss << *f.spec.max;
        } else {
          oss << "+inf";
        }
        oss << "]";
      } else {
        oss << "—";
      }
      oss << " | ";
      if (f.spec.typical) {
        oss << *f.spec.typical;
      } else {
        oss << "—";
      }
      oss << " | " << (f.spec.units.empty() ? "—" : f.spec.units) << " | "
          << f.spec.description << " |\n";
    }
    return oss.str();
  }

private:
  struct Field {
    SchemaSpec spec;
    std::string expected_type;
    bool integer{false};
    std::function<void(Params &, double)> assign_real;
    std::function<void(Params &, int)> assign_int;
  };

  template <class T>
  void add_field(T Params::*member, SchemaSpec spec, bool integer) {
    Field f;
    f.spec = std::move(spec);
    f.integer = integer;
    f.expected_type = integer ? "integer" : "number";
    if constexpr (std::is_same_v<T, double>) {
      f.assign_real = [member](Params &p, double v) { p.*member = v; };
    } else {
      f.assign_int = [member](Params &p, int v) { p.*member = v; };
    }
    m_fields.push_back(std::move(f));
  }

  void apply(const nlohmann::json &config, Params &out,
             SchemaValidationResult &result) const {
    for (const auto &f : m_fields) {
      apply_one(config, out, f, result);
    }
  }

  void apply_one(const nlohmann::json &config, Params &out, const Field &f,
                 SchemaValidationResult &result) const {
    const auto &name = f.spec.name;
    if (!config.contains(name)) {
      if (f.spec.default_value) {
        assign_default(out, f, *f.spec.default_value);
        std::ostringstream val;
        val << *f.spec.default_value << " (default)";
        result.validated_params[name] = val.str();
        return;
      }
      if (f.spec.required) {
        result.errors.push_back(schema_config_error(
            name, f.spec.description, f.expected_type, "missing",
            f.spec.example));
      }
      return;
    }

    const nlohmann::json &val = config.at(name);
    const std::string got = schema_json_value_string(config, name);

    if (f.integer) {
      if (!val.is_number_integer()) {
        result.errors.push_back(schema_config_error(
            name, f.spec.description, f.expected_type, got, f.spec.example));
        return;
      }
      const int iv = val.template get<int>();
      if (auto err = bounds_error(f, static_cast<double>(iv))) {
        result.errors.push_back(*err);
        return;
      }
      f.assign_int(out, iv);
      std::ostringstream recorded;
      recorded << iv;
      result.validated_params[name] = recorded.str();
      return;
    }

    if (!val.is_number() || !std::isfinite(val.template get<double>())) {
      result.errors.push_back(schema_config_error(
          name, f.spec.description, f.expected_type, got, f.spec.example));
      return;
    }
    const double dv = val.template get<double>();
    if (auto err = bounds_error(f, dv)) {
      result.errors.push_back(*err);
      return;
    }
    f.assign_real(out, dv);
    std::ostringstream recorded;
    recorded << dv;
    result.validated_params[name] = recorded.str();
  }

  void assign_default(Params &out, const Field &f, double def) const {
    if (f.integer) {
      f.assign_int(out, static_cast<int>(def));
    } else {
      f.assign_real(out, def);
    }
  }

  [[nodiscard]] static std::optional<std::string>
  bounds_error(const Field &f, double value) {
    if (f.spec.min && value < *f.spec.min) {
      std::ostringstream msg;
      msg << "Parameter '" << f.spec.name << "' = " << value
          << " is below minimum " << *f.spec.min;
      if (f.spec.min && f.spec.max) {
        msg << " (valid range: [" << *f.spec.min << ", " << *f.spec.max
            << "])";
      }
      return msg.str();
    }
    if (f.spec.max && value > *f.spec.max) {
      std::ostringstream msg;
      msg << "Parameter '" << f.spec.name << "' = " << value
          << " exceeds maximum " << *f.spec.max;
      if (f.spec.min && f.spec.max) {
        msg << " (valid range: [" << *f.spec.min << ", " << *f.spec.max
            << "])";
      }
      return msg.str();
    }
    return std::nullopt;
  }

  std::string m_model_name{"Model"};
  std::vector<Field> m_fields;
};

/**
 * @brief Physics that exposes `schema()` for its `parameters_type`.
 */
template <class Physics>
concept HasParameterSchema =
    HasParameters<Physics> && requires {
      {
        Physics::schema()
      } -> std::convertible_to<
          ParameterSchema<typename Physics::parameters_type>>;
    };

} // namespace pfc::sim
