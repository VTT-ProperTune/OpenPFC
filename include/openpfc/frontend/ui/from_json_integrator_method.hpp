// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file from_json_integrator_method.hpp
 * @brief `from_json` specialization for `sim::steppers::RKIntegratorMethod`
 */

#ifndef PFC_UI_FROM_JSON_INTEGRATOR_METHOD_HPP
#define PFC_UI_FROM_JSON_INTEGRATOR_METHOD_HPP

#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/frontend/ui/errors_config_format.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>
#include <openpfc/kernel/simulation/steppers/integrator_method.hpp>

namespace pfc::ui {

[[nodiscard]] inline const std::vector<std::string> &rk_integrator_method_tokens() {
  static const std::vector<std::string> tokens{
      "euler", "rk2_midpoint", "rk2_heun", "rk4_classical", "bogacki_shampine32"};
  return tokens;
}

/**
 * @brief Deserialize RKIntegratorMethod from JSON
 *
 * Specialization of from_json<T> for RKIntegratorMethod. Parses lowercase
 * string values (e.g., "rk4_classical") and returns corresponding enum value.
 *
 * Supported strings:
 * - "euler"
 * - "rk2_midpoint"
 * - "rk2_heun"
 * - "rk4_classical"
 * - "bogacki_shampine32"
 *
 * @param j JSON value containing string representation
 * @return RKIntegratorMethod enum value
 *
 * @throws std::invalid_argument if string does not match any known method
 *
 * @note Follows from_json<Time> pattern in from_json_world_time.hpp
 */
template <>
[[nodiscard]] inline pfc::sim::steppers::RKIntegratorMethod
from_json<pfc::sim::steppers::RKIntegratorMethod>(const json &j) {
  if (!j.is_string()) {
    throw std::invalid_argument(format_config_error(
        "method", "Runge-Kutta integrator method", "string", "non-string",
        rk_integrator_method_tokens(), "\"method\": \"euler\""));
  }
  const std::string s = j.get<std::string>();

  if (s == "euler") return pfc::sim::steppers::RKIntegratorMethod::Euler;
  if (s == "rk2_midpoint")
    return pfc::sim::steppers::RKIntegratorMethod::RK2_Midpoint;
  if (s == "rk2_heun") return pfc::sim::steppers::RKIntegratorMethod::RK2_Heun;
  if (s == "rk4_classical")
    return pfc::sim::steppers::RKIntegratorMethod::RK4_Classical;
  if (s == "bogacki_shampine32")
    return pfc::sim::steppers::RKIntegratorMethod::BogackiShampine32;

  throw std::invalid_argument(format_config_error(
      "method", "Runge-Kutta integrator method", "one of the valid RK method tokens",
      s, rk_integrator_method_tokens(), "\"method\": \"euler\""));
}

} // namespace pfc::ui

#endif // PFC_UI_FROM_JSON_INTEGRATOR_METHOD_HPP
