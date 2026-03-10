// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file toml_to_json.hpp
 * @brief Convert TOML data to nlohmann::json format
 *
 * This utility provides conversion from TOML format to JSON format, allowing
 * TOML configuration files to be used with existing JSON parsing code.
 *
 * @note This is a zero-cost abstraction - TOML files are converted to JSON
 *       at load time, then all existing JSON parsing code works unchanged.
 */

#ifndef PFC_TOML_TO_JSON_HPP
#define PFC_TOML_TO_JSON_HPP

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

namespace pfc {
namespace utils {

// Forward declaration
inline nlohmann::json toml_to_json(const toml::node &node_ref);

/**
 * @brief Convert a TOML table to nlohmann::json
 *
 * Recursively converts TOML data structures to JSON format, preserving
 * all data types and structure.
 *
 * @param table TOML table to convert
 * @return JSON object representing the TOML data
 */
inline nlohmann::json toml_to_json(const toml::table &table) {
  nlohmann::json j = nlohmann::json::object();

  for (const auto &[key, value] : table) {
    std::string key_str(key.str());
    // value is a const toml::node& when iterating over table
    j[key_str] = toml_to_json(value);
  }

  return j;
}

/**
 * @brief Convert a TOML node to nlohmann::json
 *
 * Handles all TOML node types (table, array, string, integer, float, boolean,
 * date, time, etc.) and converts them to appropriate JSON types.
 *
 * @param node TOML node to convert
 * @return JSON value representing the TOML node
 */
inline nlohmann::json toml_to_json(const toml::node &node_ref) {
  switch (node_ref.type()) {
  case toml::node_type::table: {
    return toml_to_json(*node_ref.as_table());
  }
  case toml::node_type::array: {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &elem : *node_ref.as_array()) {
      // elem is a node_view, convert it directly
      arr.push_back(toml_to_json(elem));
    }
    return arr;
  }
  case toml::node_type::string: {
    return std::string(node_ref.as_string()->get());
  }
  case toml::node_type::integer: {
    return node_ref.as_integer()->get();
  }
  case toml::node_type::floating_point: {
    return node_ref.as_floating_point()->get();
  }
  case toml::node_type::boolean: {
    return node_ref.as_boolean()->get();
  }
  case toml::node_type::date: {
    auto date = node_ref.as_date()->get();
    // Convert date to ISO 8601 string format
    std::string date_str = std::to_string(date.year) + "-" +
                           (date.month < 10 ? "0" : "") +
                           std::to_string(date.month) + "-" +
                           (date.day < 10 ? "0" : "") + std::to_string(date.day);
    return date_str;
  }
  case toml::node_type::time: {
    auto time = node_ref.as_time()->get();
    // Convert time to ISO 8601 string format
    std::string time_str =
        (time.hour < 10 ? "0" : "") + std::to_string(time.hour) + ":" +
        (time.minute < 10 ? "0" : "") + std::to_string(time.minute) + ":" +
        (time.second < 10 ? "0" : "") + std::to_string(time.second);
    if (time.nanosecond > 0) {
      time_str += "." + std::to_string(time.nanosecond);
    }
    return time_str;
  }
  case toml::node_type::date_time: {
    auto dt = node_ref.as_date_time()->get();
    // Convert date-time to ISO 8601 string format
    std::string dt_str =
        std::to_string(dt.date.year) + "-" + (dt.date.month < 10 ? "0" : "") +
        std::to_string(dt.date.month) + "-" + (dt.date.day < 10 ? "0" : "") +
        std::to_string(dt.date.day) + "T" + (dt.time.hour < 10 ? "0" : "") +
        std::to_string(dt.time.hour) + ":" + (dt.time.minute < 10 ? "0" : "") +
        std::to_string(dt.time.minute) + ":" + (dt.time.second < 10 ? "0" : "") +
        std::to_string(dt.time.second);
    if (dt.time.nanosecond > 0) {
      dt_str += "." + std::to_string(dt.time.nanosecond);
    }
    if (dt.offset.has_value()) {
      auto offset = dt.offset.value();
      int offset_hours = offset.minutes / 60;
      int offset_minutes = offset.minutes % 60;
      dt_str += std::string(offset_hours >= 0 ? "+" : "-") +
                std::string(std::abs(offset_hours) < 10 ? "0" : "") +
                std::to_string(std::abs(offset_hours)) + ":" +
                std::string(std::abs(offset_minutes) < 10 ? "0" : "") +
                std::to_string(std::abs(offset_minutes));
    }
    return dt_str;
  }
  default: {
    // For unknown types, try to convert to string or return null
    if (node_ref.is_string()) {
      return std::string(node_ref.as_string()->get());
    }
    return nlohmann::json(nullptr);
  }
  }
}

/**
 * @brief Convert a TOML node_view to nlohmann::json
 *
 * Overload for node_view (used when accessing array elements)
 *
 * @param node_view TOML node_view to convert
 * @return JSON value representing the TOML node
 */
inline nlohmann::json
toml_to_json(const toml::node_view<const toml::node> &node_view) {
  const toml::node *node = node_view.node();
  if (!node) {
    return nlohmann::json(nullptr);
  }
  return toml_to_json(*node);
}

} // namespace utils
} // namespace pfc

#endif // PFC_TOML_TO_JSON_HPP
