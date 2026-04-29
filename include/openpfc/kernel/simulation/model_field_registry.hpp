// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file model_field_registry.hpp
 * @brief Named real/complex field registration for simulation models
 *
 * @details
 * Holds references to real and complex fields keyed by name. `Model` composes a
 * `ModelFieldRegistry` so field bookkeeping is separate from FFT, world, and
 * physics (`initialize` / `step`).
 *
 * @see Model for the full simulation abstraction
 */

#ifndef PFC_KERNEL_SIMULATION_MODEL_FIELD_REGISTRY_HPP
#define PFC_KERNEL_SIMULATION_MODEL_FIELD_REGISTRY_HPP

#include <openpfc/kernel/data/model_types.hpp>

#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace pfc {

/**
 * @brief Registry of named real and complex fields (references, not owned)
 */
class ModelFieldRegistry {
private:
  RealFieldSet m_real_fields;
  ComplexFieldSet m_complex_fields;

  [[nodiscard]] std::string list_field_names() const {
    std::vector<std::string> names;
    for (const auto &[name, _] : m_real_fields) {
      names.push_back(name);
    }
    for (const auto &[name, _] : m_complex_fields) {
      names.push_back(name);
    }
    if (names.empty()) {
      return "(none)";
    }
    return std::accumulate(
        std::next(names.begin()), names.end(), names[0],
        [](const std::string &a, const std::string &b) { return a + ", " + b; });
  }

public:
  [[nodiscard]] bool has_real_field(std::string_view field_name) const noexcept {
    return m_real_fields.count(std::string(field_name)) > 0;
  }

  void add_real_field(std::string_view name, RealField &field) {
    m_real_fields.emplace(std::string(name), field);
  }

  [[nodiscard]] bool has_complex_field(std::string_view field_name) const noexcept {
    return m_complex_fields.count(std::string(field_name)) > 0;
  }

  void add_complex_field(std::string_view name, ComplexField &field) {
    m_complex_fields.emplace(std::string(name), field);
  }

  [[nodiscard]] RealField &get_real_field(std::string_view name) {
    auto it = m_real_fields.find(std::string(name));
    if (it == m_real_fields.end()) {
      throw std::out_of_range("Real field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  [[nodiscard]] const RealField &get_real_field(std::string_view name) const {
    auto it = m_real_fields.find(std::string(name));
    if (it == m_real_fields.end()) {
      throw std::out_of_range("Real field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  [[nodiscard]] ComplexField &get_complex_field(std::string_view name) {
    auto it = m_complex_fields.find(std::string(name));
    if (it == m_complex_fields.end()) {
      throw std::out_of_range("Complex field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  [[nodiscard]] const ComplexField &get_complex_field(std::string_view name) const {
    auto it = m_complex_fields.find(std::string(name));
    if (it == m_complex_fields.end()) {
      throw std::out_of_range("Complex field '" + std::string(name) +
                              "' not found. "
                              "Available fields: " +
                              list_field_names());
    }
    return it->second;
  }

  void add_field(const std::string &name, RealField &field) {
    add_real_field(name, field);
  }

  void add_field(const std::string &name, ComplexField &field) {
    add_complex_field(name, field);
  }

  [[nodiscard]] bool has_field(std::string_view field_name) const noexcept {
    return has_real_field(field_name) || has_complex_field(field_name);
  }
};

} // namespace pfc

#endif // PFC_KERNEL_SIMULATION_MODEL_FIELD_REGISTRY_HPP
