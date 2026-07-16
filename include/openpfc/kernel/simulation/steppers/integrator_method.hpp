// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file integrator_method.hpp
 * @brief Type-safe Runge-Kutta integrator method selection with validation
 *
 * @details
 * This header provides a centralized enum for explicit Runge-Kutta method
 * selection with JSON deserialization support, validation, and ButcherTableau
 * factory functions. The design enables configuration-driven method selection
 * while maintaining compile-time safety and runtime error checking.
 *
 * Key features:
 * - RKIntegratorMethod enum class with 5 explicit RK methods
 * - JSON deserialization via from_json<RKIntegratorMethod> specialization
 * - Validation function for adaptive step-size control requirements
 * - ButcherTableau factory for each method
 * - String conversion for debugging and logging
 *
 * The enum is named RKIntegratorMethod to avoid collision with the existing
 * IntegratorMethod enum in time.hpp, which serves a different purpose for
 * Time class internal state tracking.
 *
 * ## Usage
 * @code
 * // Parse from JSON configuration
 * nlohmann::json config = {{"method", "rk4_classical"}};
 * auto method = pfc::ui::from_json<pfc::sim::steppers::RKIntegratorMethod>(config["method"]);
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
 * @see butcher_tableau.hpp for ButcherTableau class and factory functions
 * @see from_json_world_time.hpp for Time class JSON parsing (separate enum)
 */

#ifndef PFC_SIM_STEPPERS_INTEGRATOR_METHOD_HPP
#define PFC_SIM_STEPPERS_INTEGRATOR_METHOD_HPP

#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <optional>
#include <string>

namespace pfc {
namespace sim {
namespace steppers {

/**
 * @brief Explicit Runge-Kutta integrator method selection
 *
 * Enum class defining supported explicit Runge-Kutta methods for time
 * integration. This enum is separate from IntegratorMethod in time.hpp,
 * which serves Time class internal state tracking.
 *
 * Values:
 * - Euler: Forward Euler (1st order, 1 stage)
 * - RK2_Midpoint: Second-order RK midpoint method (2 stages)
 * - RK2_Heun: Second-order RK Heun's method (2 stages)
 * - RK4_Classical: Classical fourth-order RK (4 stages)
 * - BogackiShampine32: Embedded 3(2) adaptive method (4 stages, with error estimator)
 */
enum class RKIntegratorMethod {
    Euler,             ///< Forward Euler (1st order, 1 stage)
    RK2_Midpoint,      ///< Second-order RK midpoint (2 stages)
    RK2_Heun,          ///< Second-order RK Heun's method (2 stages)
    RK4_Classical,     ///< Classical fourth-order RK (4 stages)
    BogackiShampine32  ///< Embedded 3(2) adaptive method (4 stages, with error estimator)
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
inline std::optional<std::string> validate_method(RKIntegratorMethod method, bool requires_adaptive = false) {
    if (requires_adaptive && !is_embedded(method)) {
        return "Adaptive step-size control requires an embedded method with error estimator, but " + 
               to_string(method) + " does not provide one";
    }
    return std::nullopt;  // Valid
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
    return ButcherTableau<double>(
        1,           // stage count s
        {0.0},       // a_ij - flat 1x1 matrix
        {1.0},       // b_i
        {0.0},       // c_i
        {},          // b_hat_i (empty for non-embedded)
        "Euler",     // name
        1            // order
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
        case RKIntegratorMethod::Euler:
            return detail::make_euler_tableau();
        case RKIntegratorMethod::RK2_Midpoint:
            return make_rk2_midpoint<double>();
        case RKIntegratorMethod::RK2_Heun:
            return make_rk2_heun<double>();
        case RKIntegratorMethod::RK4_Classical:
            return make_rk4_classical<double>();
        case RKIntegratorMethod::BogackiShampine32:
            return make_embedded_rk23<double>();
    }
    // Unreachable with complete switch, but prevent compiler warning
    throw std::runtime_error("Unknown RKIntegratorMethod value");
}

} // namespace pfc::sim::steppers
} // namespace pfc::sim
} // namespace pfc

// ============================================================================
// JSON deserialization specialization
// ============================================================================

#include <nlohmann/json.hpp>
#include <openpfc/frontend/ui/from_json_fwd.hpp>

namespace pfc::ui {

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
 * @throws std::runtime_error if string does not match any known method
 *
 * @note Follows from_json<Time> pattern in from_json_world_time.hpp
 */
template<>
[[nodiscard]] inline pfc::sim::steppers::RKIntegratorMethod 
from_json<pfc::sim::steppers::RKIntegratorMethod>(const json& j) {
    const std::string s = j.get<std::string>();
    
    if (s == "euler") return pfc::sim::steppers::RKIntegratorMethod::Euler;
    if (s == "rk2_midpoint") return pfc::sim::steppers::RKIntegratorMethod::RK2_Midpoint;
    if (s == "rk2_heun") return pfc::sim::steppers::RKIntegratorMethod::RK2_Heun;
    if (s == "rk4_classical") return pfc::sim::steppers::RKIntegratorMethod::RK4_Classical;
    if (s == "bogacki_shampine32") return pfc::sim::steppers::RKIntegratorMethod::BogackiShampine32;
    
    throw std::runtime_error("Unknown RK integrator method: '" + s + 
                             "'. Valid methods are: euler, rk2_midpoint, rk2_heun, rk4_classical, bogacki_shampine32");
}

} // namespace pfc::ui

#endif // PFC_SIM_STEPPERS_INTEGRATOR_METHOD_HPP
