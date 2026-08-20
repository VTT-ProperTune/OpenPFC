// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file integrator_method.hpp
 * @brief Type-safe Runge-Kutta integrator method selection with validation
 *
 * @details
 * This header provides a centralized enum for explicit Runge-Kutta method
 * selection with validation and ButcherTableau factory functions. The design
 * enables configuration-driven method selection while maintaining compile-time
 * safety and runtime error checking. JSON deserialization lives in the
 * frontend header `from_json_integrator_method.hpp`.
 *
 * Key features:
 * - RKIntegratorMethod enum: five explicit RK methods plus IMEX/ETD identity
 * - Validation function for adaptive step-size control requirements
 * - ButcherTableau factory for explicit RK methods (`make_tableau`)
 * - String conversion for debugging, logging, and JSON tokens
 *
 * `Time` stores this same enum (there is no separate method-identity type
 * on `Time`). IMEX/ETD tokens are identity only; they have no Butcher tableau.
 *
 * ## Usage
 * @code
 * auto method = pfc::sim::steppers::RKIntegratorMethod::RK4_Classical;
 *
 * // Validate for adaptive step-size control
 * if (auto error = pfc::sim::steppers::validate_method(method, true)) {
 *   throw std::runtime_error(error.value());
 * }
 *
 * // Get ButcherTableau for the method
 * auto tableau = pfc::sim::steppers::make_tableau(method);
 * @endcode
 *
 * For JSON configuration parsing, see
 * `include/openpfc/frontend/ui/from_json_integrator_method.hpp`.
 *
 * @see butcher_tableau.hpp for ButcherTableau class and factory functions
 * @see from_json_integrator_method.hpp for from_json<RKIntegratorMethod>
 */

#ifndef PFC_SIM_STEPPERS_INTEGRATOR_METHOD_HPP
#define PFC_SIM_STEPPERS_INTEGRATOR_METHOD_HPP

#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <optional>
#include <stdexcept>
#include <string>

namespace pfc {
namespace sim {
namespace steppers {

/**
 * @brief Integrator method identity stored on `Time`
 *
 * Explicit Runge-Kutta methods plus IMEX/ETD tokens. JSON
 * `simulator.integrator.method` / `timestepping.integrator.method` use the
 * same strings as `to_string`. `make_tableau` is defined only for the RK
 * values (`is_runge_kutta`).
 *
 * Values:
 * - Euler: Forward Euler (1st order, 1 stage)
 * - RK2_Midpoint: Second-order RK midpoint method (2 stages)
 * - RK2_Heun: Second-order RK Heun's method (2 stages)
 * - RK4_Classical: Classical fourth-order RK (4 stages)
 * - BogackiShampine32: Embedded 3(2) adaptive method (4 stages, with error
 * estimator)
 * - ImexEuler: implicit-explicit Euler identity (`"imex_euler"`)
 * - ETD1: first-order exponential time differencing identity (`"etd1"`)
 */
enum class RKIntegratorMethod {
  Euler,             ///< Forward Euler (1st order, 1 stage)
  RK2_Midpoint,      ///< Second-order RK midpoint (2 stages)
  RK2_Heun,          ///< Second-order RK Heun's method (2 stages)
  RK4_Classical,     ///< Classical fourth-order RK (4 stages)
  BogackiShampine32, ///< Embedded 3(2) adaptive method (4 stages, with error
                     ///< estimator)
  ImexEuler,         ///< IMEX Euler identity token (no Butcher tableau)
  ETD1               ///< ETD1 identity token (no Butcher tableau)
};

/**
 * @brief Convert RKIntegratorMethod to string
 *
 * Returns lowercase string representation matching JSON format.
 *
 * @param method Method to convert
 * @return String representation (e.g., "rk4_classical")
 */
inline std::string to_string(RKIntegratorMethod method) {
  switch (method) {
  case RKIntegratorMethod::Euler: return "euler";
  case RKIntegratorMethod::RK2_Midpoint: return "rk2_midpoint";
  case RKIntegratorMethod::RK2_Heun: return "rk2_heun";
  case RKIntegratorMethod::RK4_Classical: return "rk4_classical";
  case RKIntegratorMethod::BogackiShampine32: return "bogacki_shampine32";
  case RKIntegratorMethod::ImexEuler: return "imex_euler";
  case RKIntegratorMethod::ETD1: return "etd1";
  }
  // Unreachable with complete switch, but prevent compiler warning
  return "unknown";
}

/**
 * @brief Check if method has embedded error estimator
 *
 * Embedded methods provide two sets of output weights (b and b_hat) for
 * adaptive step-size control via error estimation.
 *
 * @param method Method to check
 * @return true if method has embedded error estimator, false otherwise
 */
inline bool is_embedded(RKIntegratorMethod method) {
  return method == RKIntegratorMethod::BogackiShampine32;
}

/** @brief True for explicit RK values that have a Butcher tableau. */
inline bool is_runge_kutta(RKIntegratorMethod method) {
  switch (method) {
  case RKIntegratorMethod::Euler:
  case RKIntegratorMethod::RK2_Midpoint:
  case RKIntegratorMethod::RK2_Heun:
  case RKIntegratorMethod::RK4_Classical:
  case RKIntegratorMethod::BogackiShampine32: return true;
  case RKIntegratorMethod::ImexEuler:
  case RKIntegratorMethod::ETD1: return false;
  }
  return false;
}

/**
 * @brief Validate RK integrator method against requirements
 *
 * Checks if the method satisfies specified requirements (e.g., adaptive
 * step-size control needs an embedded method with error estimator).
 *
 * @param method Method to validate
 * @param requires_adaptive Whether method must support adaptive step-size control
 * @return Error message if invalid, empty optional if valid
 *
 * @note Follows ParameterMetadata<T>::validate() pattern: returns
 *       std::optional<std::string> where empty means valid.
 */
inline std::optional<std::string> validate_method(RKIntegratorMethod method,
                                                  bool requires_adaptive = false) {
  if (requires_adaptive && !is_embedded(method)) {
    return "Adaptive step-size control requires an embedded method with error "
           "estimator, but " +
           to_string(method) + " does not provide one";
  }
  return std::nullopt; // Valid
}

namespace detail {

/**
 * @brief Create Euler ButcherTableau
 *
 * Constructs a 1-stage explicit Euler tableau with coefficients:
 * - a_ij = [0] (single stage, no dependencies)
 * - b_i = [1] (output weight)
 * - c_i = [0] (stage time)
 *
 * @return ButcherTableau<double> configured for forward Euler
 *
 * @note This is a local helper since butcher_tableau.hpp cannot be modified
 *       per non-scope constraints.
 */
inline ButcherTableau<double> make_euler_tableau() {
  // 1-stage explicit Euler: a_ij=[0], b_i=[1], c_i=[0]
  return ButcherTableau<double>(1,       // stage count s
                                {0.0},   // a_ij - flat 1x1 matrix
                                {1.0},   // b_i
                                {0.0},   // c_i
                                {},      // b_hat_i (empty for non-embedded)
                                "Euler", // name
                                1        // order
  );
}

} // namespace detail

/**
 * @brief Create ButcherTableau for specified RK method
 *
 * Factory function that returns the appropriate ButcherTableau for each
 * method. Euler uses local implementation; others delegate to existing
 * factory functions in butcher_tableau.hpp.
 *
 * @param method RK integrator method
 * @return ButcherTableau<double> configured for the specified method
 *
 * @throws TableauValidationError if tableau construction fails (should not
 *         occur with predefined methods)
 */
inline ButcherTableau<double> make_tableau(RKIntegratorMethod method) {
  switch (method) {
  case RKIntegratorMethod::Euler: return detail::make_euler_tableau();
  case RKIntegratorMethod::RK2_Midpoint: return make_rk2_midpoint<double>();
  case RKIntegratorMethod::RK2_Heun: return make_rk2_heun<double>();
  case RKIntegratorMethod::RK4_Classical: return make_rk4_classical<double>();
  case RKIntegratorMethod::BogackiShampine32: return make_embedded_rk23<double>();
  case RKIntegratorMethod::ImexEuler:
  case RKIntegratorMethod::ETD1:
    throw std::invalid_argument("make_tableau: " + to_string(method) +
                                " is not an explicit Runge-Kutta method");
  }
  throw std::runtime_error("Unknown RKIntegratorMethod value");
}

} // namespace steppers
} // namespace sim
} // namespace pfc

#endif // PFC_SIM_STEPPERS_INTEGRATOR_METHOD_HPP
