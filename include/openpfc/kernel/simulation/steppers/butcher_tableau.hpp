// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cmath>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pfc::sim::steppers {

/**
 * @file butcher_tableau.hpp
 * @brief Validated explicit Runge-Kutta coefficient tableau infrastructure
 *
 * This header provides the ButcherTableau class template for representing and
 * validating explicit Runge-Kutta method coefficients. Tableaus define the
 * a_ij weights, b_i output weights, and c_i stage times used by explicit
 * RK integrators to advance time-dependent systems.
 *
 * Validation ensures explicit lower-triangular structure, row-sum consistency,
 * finite coefficient values, and correct array dimensions before numerical
 * integration begins. Failed validation yields descriptive errors identifying
 * the specific component and condition that failed.
 */

/**
 * @brief Exception thrown when ButcherTableau validation fails
 *
 * Describes why a tableau definition is invalid, including the error type
 * and a human-readable message with component names and indices.
 */
class TableauValidationError : public std::runtime_error {
public:
  /**
   * @brief Validation error categories
   *
   * Each type corresponds to a specific validation condition that must be
   * satisfied for a valid explicit Runge-Kutta tableau.
   */
  enum class ErrorType {
    StageCountMismatch,        ///< Coefficient array dimensions do not match stage count
    ExplicitStructureViolation, ///< Non-zero a_ij found on or above diagonal (must be zero for explicit methods)
    RowSumInconsistency,       ///< Row sum of a_ij does not match c_i values
    NonFiniteCoefficient,      ///< NaN or Inf detected in coefficient arrays
    InvalidStageCount          ///< Stage count s is less than 1
  };

  /**
   * @brief Construct validation error
   *
   * @param type Category of validation failure
   * @param message Descriptive error message with component details
   */
  TableauValidationError(ErrorType type, const std::string& message);

  /**
   * @brief Get error type
   *
   * @return ErrorType category indicating which validation condition failed
   */
  ErrorType error_type() const noexcept;

private:
  ErrorType type_;
};

/**
 * @brief Validated explicit Runge-Kutta coefficient tableau
 *
 * Stores immutable coefficient arrays for explicit Runge-Kutta methods and
 * validates them upon construction. Coefficients follow the Butcher tableau
 * convention:
 *
 *   - a_ij: explicit lower-triangular weight matrix (s×s, a_ij = 0 for i ≤ j)
 *   - b_i: output weights for final solution (s-vector)
 *   - c_i: stage times (s-vector, satisfying ∑_{j=0}^{s-1} a_ij = c_i for each row i)
 *   - b_hat_i: optional embedded error-control weights (s-vector for adaptive step-size methods)
 *
 * Type parameter T must be a real floating-point type (float or double).
 * Complex state types can use real-valued tableaus; coefficients are applied
 * element-wise without requiring complex coefficient definitions.
 *
 * @tparam T Real floating-point coefficient type (float or double)
 *
 * @note Coefficients are immutable after construction; accessors return
 *       values, not references, guaranteeing tableau stability during integrator use.
 *
 * @note Row-sum validation uses type-aware tolerances scaled by machine epsilon
 *       for the coefficient type. For float, tolerances are ~10× larger than double.
 */
template <typename T>
class ButcherTableau {
  static_assert(std::is_floating_point_v<T>, "ButcherTableau<T> requires T to be a real floating-point type");

public:
  /**
   * @brief Construct and validate a Butcher tableau
   *
   * @param s Stage count (number of stages, s ≥ 1)
   * @param a_ij Lower-triangular weight matrix in row-major order (s×s)
   * @param b_i Output weights (s elements)
   * @param c_i Stage times (s elements)
   * @param b_hat_i Optional embedded error-control weights (s elements, or empty)
   * @param name Method name for display and diagnostics (optional)
   * @param order Declared order of convergence (optional, informational)
   * @param embedded_order Order of embedded method (if b_hat_i provided, optional)
   *
   * @throws TableauValidationError if any validation condition fails
   *
   * @post All coefficient arrays are stored as const and cannot be modified after construction
   * @post Tableau object represents a valid explicit RK method configuration
   */
  ButcherTableau(unsigned int s,
                 std::vector<T> a_ij,
                 std::vector<T> b_i,
                 std::vector<T> c_i,
                 std::vector<T> b_hat_i = {},
                 std::string_view name = "",
                 unsigned int order = 0,
                 unsigned int embedded_order = 0);

  /**
   * @brief Get stage count
   *
   * @return Number of stages in the Runge-Kutta method
   */
  unsigned int stage_count() const noexcept { return s_; }

  /**
   * @brief Get a_ij coefficient
   *
   * @param i Row index (0-based, must be < stage_count())
   * @param j Column index (0-based, must be < stage_count())
   * @return Weight coefficient for a_ij[i][j]
   * @throws std::out_of_range if indices are out of bounds
   */
  T a(unsigned int i, unsigned int j) const;

  /**
   * @brief Get b_i coefficient
   *
   * @param i Index (0-based, must be < stage_count())
   * @return Output weight b_i[i]
   * @throws std::out_of_range if index is out of bounds
   */
  T b(unsigned int i) const;

  /**
   * @brief Get c_i coefficient
   *
   * @param i Index (0-based, must be < stage_count())
   * @return Stage time c_i[i]
   * @throws std::out_of_range if index is out of bounds
   */
  T c(unsigned int i) const;

  /**
   * @brief Get embedded b_hat_i coefficient
   *
   * @param i Index (0-based, must be < stage_count())
   * @return Embedded error estimator weight b_hat_i[i]
   * @throws std::runtime_error if tableau has no embedded weights
   * @throws std::out_of_range if index is out of bounds
   */
  T b_hat(unsigned int i) const;

  /**
   * @brief Get method name
   *
   * @return Method name string (may be empty if not set)
   */
  std::string_view name() const noexcept { return name_; }

  /**
   * @brief Get declared method order
   *
   * @return Declared order of convergence (0 if not set)
   */
  unsigned int order() const noexcept { return order_; }

  /**
   * @brief Get embedded method order
   *
   * @return Order of embedded method (0 if not set or no embedded method)
   */
  unsigned int embedded_order() const noexcept { return embedded_order_; }

  /**
   * @brief Check if tableau has embedded error estimator
   *
   * @return true if b_hat_i weights are present
   */
  bool has_embedded() const noexcept { return !b_hat_.empty(); }

private:
  void validate() const;
  void check_finite(const T& value, const char* component, std::size_t index) const;
  T tolerance() const noexcept;

  const unsigned int s_;
  const std::vector<T> a_ij_;
  const std::vector<T> b_i_;
  const std::vector<T> c_i_;
  const std::vector<T> b_hat_;
  const std::string name_;
  const unsigned int order_;
  const unsigned int embedded_order_;
};

// ============================================================================
// TableauValidationError implementation
// ============================================================================

inline TableauValidationError::TableauValidationError(ErrorType type, const std::string& message)
  : std::runtime_error(message), type_(type)
{}

inline TableauValidationError::ErrorType TableauValidationError::error_type() const noexcept
{
  return type_;
}

// ============================================================================
// ButcherTableau implementation
// ============================================================================

template <typename T>
inline ButcherTableau<T>::ButcherTableau(unsigned int s,
                                         std::vector<T> a_ij,
                                         std::vector<T> b_i,
                                         std::vector<T> c_i,
                                         std::vector<T> b_hat_i,
                                         std::string_view name,
                                         unsigned int order,
                                         unsigned int embedded_order)
  : s_(s),
    a_ij_(std::move(a_ij)),
    b_i_(std::move(b_i)),
    c_i_(std::move(c_i)),
    b_hat_(std::move(b_hat_i)),
    name_(name),
    order_(order),
    embedded_order_(embedded_order)
{
  validate();
}

template <typename T>
inline void ButcherTableau<T>::validate() const
{
  if (s_ < 1) {
    throw TableauValidationError(
      TableauValidationError::ErrorType::InvalidStageCount,
      "stage_count " + std::to_string(s_) + " must be >= 1"
    );
  }

  if (a_ij_.size() != s_ * s_) {
    throw TableauValidationError(
      TableauValidationError::ErrorType::StageCountMismatch,
      "a_ij size " + std::to_string(a_ij_.size()) + " != stage_count^2 (" +
      std::to_string(s_) + ")"
    );
  }
  if (b_i_.size() != s_) {
    throw TableauValidationError(
      TableauValidationError::ErrorType::StageCountMismatch,
      "b_i size " + std::to_string(b_i_.size()) + " != stage_count (" +
      std::to_string(s_) + ")"
    );
  }
  if (c_i_.size() != s_) {
    throw TableauValidationError(
      TableauValidationError::ErrorType::StageCountMismatch,
      "c_i size " + std::to_string(c_i_.size()) + " != stage_count (" +
      std::to_string(s_) + ")"
    );
  }
  if (!b_hat_.empty() && b_hat_.size() != s_) {
    throw TableauValidationError(
      TableauValidationError::ErrorType::StageCountMismatch,
      "b_hat_i size " + std::to_string(b_hat_.size()) + " != stage_count (" +
      std::to_string(s_) + ")"
    );
  }

  for (std::size_t i = 0; i < a_ij_.size(); ++i) {
    check_finite(a_ij_[i], "a_ij", i);
  }
  for (std::size_t i = 0; i < b_i_.size(); ++i) {
    check_finite(b_i_[i], "b_i", i);
  }
  for (std::size_t i = 0; i < c_i_.size(); ++i) {
    check_finite(c_i_[i], "c_i", i);
  }
  for (std::size_t i = 0; i < b_hat_.size(); ++i) {
    check_finite(b_hat_[i], "b_hat_i", i);
  }

  const T tol = tolerance() * T(10);
  for (unsigned int i = 0; i < s_; ++i) {
    for (unsigned int j = i; j < s_; ++j) {
      if (std::fabs(a(i, j)) > tol) {
        throw TableauValidationError(
          TableauValidationError::ErrorType::ExplicitStructureViolation,
          "a_ij[" + std::to_string(i) + "][" + std::to_string(j) + "] = " +
          std::to_string(a(i, j)) + " != 0 for explicit method (must be zero for i <= j)"
        );
      }
    }
  }

  const T row_sum_tol = tolerance() * T(100);
  for (unsigned int i = 0; i < s_; ++i) {
    T row_sum = T(0);
    for (unsigned int j = 0; j < s_; ++j) {
      row_sum += a(i, j);
    }
    if (std::fabs(row_sum - c(i)) > row_sum_tol * (T(1) + std::fabs(c(i)))) {
      throw TableauValidationError(
        TableauValidationError::ErrorType::RowSumInconsistency,
        "sum_j a_ij[" + std::to_string(i) + "][j] = " + std::to_string(row_sum) +
        " != c_i[" + std::to_string(i) + "] = " + std::to_string(c(i))
      );
    }
  }
}

template <typename T>
inline void ButcherTableau<T>::check_finite(const T& value, const char* component, std::size_t index) const
{
  if (!std::isfinite(value)) {
    throw TableauValidationError(
      TableauValidationError::ErrorType::NonFiniteCoefficient,
      std::string(component) + "[" + std::to_string(index) + "] = " +
      std::to_string(value) + " is not finite"
    );
  }
}

template <typename T>
inline T ButcherTableau<T>::tolerance() const noexcept
{
  return std::numeric_limits<T>::epsilon();
}

template <typename T>
inline T ButcherTableau<T>::a(unsigned int i, unsigned int j) const
{
  if (i >= s_ || j >= s_) {
    throw std::out_of_range(
      "a(" + std::to_string(i) + "," + std::to_string(j) + ") out of range [0," +
      std::to_string(s_) + ")"
    );
  }
  return a_ij_[i * s_ + j];
}

template <typename T>
inline T ButcherTableau<T>::b(unsigned int i) const
{
  if (i >= s_) {
    throw std::out_of_range(
      "b(" + std::to_string(i) + ") out of range [0," +
      std::to_string(s_) + ")"
    );
  }
  return b_i_[i];
}

template <typename T>
inline T ButcherTableau<T>::c(unsigned int i) const
{
  if (i >= s_) {
    throw std::out_of_range(
      "c(" + std::to_string(i) + ") out of range [0," +
      std::to_string(s_) + ")"
    );
  }
  return c_i_[i];
}

template <typename T>
inline T ButcherTableau<T>::b_hat(unsigned int i) const
{
  if (b_hat_.empty()) {
    throw std::runtime_error("b_hat not available for this tableau");
  }
  if (i >= s_) {
    throw std::out_of_range(
      "b_hat(" + std::to_string(i) + ") out of range [0," +
      std::to_string(s_) + ")"
    );
  }
  return b_hat_[i];
}

// ============================================================================
// Factory functions for standard tableaus
// ============================================================================

/**
 * @brief Create RK2 midpoint tableau
 *
 * Second-order explicit Runge-Kutta method using midpoint rule.
 * Coefficients:
 *   - a_ij: [[0, 0], [0.5, 0]]
 *   - b_i: [0, 1]
 *   - c_i: [0, 0.5]
 *
 * @tparam T Real floating-point type (float or double)
 * @return ButcherTableau<T> configured for RK2 midpoint
 */
template <typename T>
inline ButcherTableau<T> make_rk2_midpoint()
{
  return ButcherTableau<T>(
    2,
    {T(0), T(0), T(0.5), T(0)},
    {T(0), T(1)},
    {T(0), T(0.5)},
    {},
    "RK2 midpoint",
    2
  );
}

/**
 * @brief Create RK2 Heun tableau
 *
 * Second-order explicit Runge-Kutta method using Heun's method.
 * Coefficients:
 *   - a_ij: [[0, 0], [1, 0]]
 *   - b_i: [0.5, 0.5]
 *   - c_i: [0, 1]
 *
 * @tparam T Real floating-point type (float or double)
 * @return ButcherTableau<T> configured for RK2 Heun
 */
template <typename T>
inline ButcherTableau<T> make_rk2_heun()
{
  return ButcherTableau<T>(
    2,
    {T(0), T(0), T(1), T(0)},
    {T(0.5), T(0.5)},
    {T(0), T(1)},
    {},
    "RK2 Heun",
    2
  );
}

/**
 * @brief Create classical RK4 tableau
 *
 * Fourth-order explicit Runge-Kutta method using classical coefficients.
 * Coefficients:
 *   - a_ij: [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]
 *   - b_i: [1/6, 1/3, 1/3, 1/6]
 *   - c_i: [0, 0.5, 0.5, 1]
 *
 * @tparam T Real floating-point type (float or double)
 * @return ButcherTableau<T> configured for classical RK4
 */
template <typename T>
inline ButcherTableau<T> make_rk4_classical()
{
  return ButcherTableau<T>(
    4,
    {T(0), T(0), T(0), T(0),
     T(0.5), T(0), T(0), T(0),
     T(0), T(0.5), T(0), T(0),
     T(0), T(0), T(1), T(0)},
    {T(1.0/6.0), T(1.0/3.0), T(1.0/3.0), T(1.0/6.0)},
    {T(0), T(0.5), T(0.5), T(1)},
    {},
    "RK4 classical",
    4
  );
}

/**
 * @brief Create Bogacki-Shampine 3(2) embedded tableau
 *
 * Third-order explicit Runge-Kutta method with embedded second-order
 * error estimator (Bogacki-Shampine 3(2)). The primary output (b) is
 * third-order accurate, while b_hat provides the second-order estimate
 * for adaptive step-size control.
 *
 * Coefficients:
 *   - a_ij: [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.75, 0, 0], [2/9, 1/3, 4/9, 0]]
 *   - b_i (3rd order): [2/9, 1/3, 4/9, 0]
 *   - b_hat_i (2nd order): [7/24, 1/4, 1/3, 1/8]
 *   - c_i: [0, 0.5, 0.75, 1]
 *
 * @tparam T Real floating-point type (float or double)
 * @return ButcherTableau<T> configured for Bogacki-Shampine 3(2)
 */
template <typename T>
inline ButcherTableau<T> make_embedded_rk23()
{
  return ButcherTableau<T>(
    4,
    {T(0), T(0), T(0), T(0),
     T(0.5), T(0), T(0), T(0),
     T(0), T(0.75), T(0), T(0),
     T(2.0/9.0), T(1.0/3.0), T(4.0/9.0), T(0)},
    {T(2.0/9.0), T(1.0/3.0), T(4.0/9.0), T(0)},
    {T(0), T(0.5), T(0.75), T(1)},
    {T(7.0/24.0), T(1.0/4.0), T(1.0/3.0), T(1.0/8.0)},
    "Bogacki-Shampine 3(2)",
    3,
    2
  );
}

} // namespace pfc::sim::steppers
