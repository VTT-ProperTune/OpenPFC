// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stepper_validation.hpp
 * @brief Validation utilities for stepper-model compatibility.
 *
 * @details
 * This header provides compile-time and runtime validation functions for
 * ensuring that stepper and model types are compatible. It addresses three
 * key validation scenarios:
 *
 * 1. **RHS signature compatibility**: Ensures that a model's `rhs(double t, const G&)`
 *    method is callable with the grad type `G` returned by the evaluator's
 *    `operator()(int, int, int)`, and that the return type is either scalar
 *    or tuple-protocol compatible.
 *
 * 2. **Spatial discretization compatibility**: Validates that the evaluator
 *    type is a recognized gradient backend (FDGradient, SpectralGradient, or
 *    CompositeGradient).
 *
 * 3. **Multi-field field count**: For multi-field steppers, ensures that the
 *    stepper's field count matches the arity of the model's RHS return type.
 *
 * All validation failures throw `StepperValidationError` with descriptive
 * error messages and specific error types for programmatic handling.
 *
 * @see openpfc/kernel/simulation/steppers/euler.hpp for stepper factories
 *      that use these validation functions
 */

#include <stdexcept>
#include <string>
#include <type_traits>

#include <openpfc/kernel/field/fd_gradient.hpp>
#include <openpfc/kernel/field/spectral_gradient.hpp>
#include <openpfc/kernel/field/composite_gradient.hpp>
#include <openpfc/kernel/field/tuple_protocol.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Exception thrown when stepper-model validation fails.
 *
 * @details
 * This exception provides specific error types for different validation
 * failures, enabling programmatic handling and user-friendly error messages.
 */
class StepperValidationError : public std::runtime_error {
public:
  /**
   * @brief Error type categorization.
   */
  enum class ErrorType {
    SignatureMismatch,   ///< Model::rhs signature incompatible with stepper
    IncompatibleBackend, ///< Evaluator spatial discretization not supported
    FieldCountMismatch   ///< Multi-field stepper receives wrong field count
  };

  /**
   * @brief Construct a validation error with type and message.
   * @param type The specific error type.
   * @param message Human-readable error description.
   */
  StepperValidationError(ErrorType type, const std::string& message);

  /**
   * @brief Get the error type.
   * @return The error type enum value.
   */
  ErrorType error_type() const noexcept;

private:
  ErrorType type_;
};

namespace detail {

/**
 * @brief Type trait to detect FD gradient evaluator types.
 */
template <class T> struct is_fd_gradient : std::false_type {};

template <class G>
struct is_fd_gradient<pfc::gradient::FDGradient<G>> : std::true_type {};

/**
 * @brief Type trait to detect spectral gradient evaluator types.
 */
template <class T> struct is_spectral_gradient : std::false_type {};

template <class G>
struct is_spectral_gradient<pfc::field::SpectralGradient<G>> : std::true_type {};

/**
 * @brief Type trait to detect composite gradient evaluator types.
 */
template <class T> struct is_composite_gradient : std::false_type {};

template <class Composite, class... PerField>
struct is_composite_gradient<pfc::field::CompositeGradient<Composite, PerField...>>
    : std::true_type {};

/**
 * @brief Concept for known evaluator types.
 */
template <class T>
concept is_known_evaluator = is_fd_gradient<T>::value ||
                           is_spectral_gradient<T>::value ||
                           is_composite_gradient<T>::value;

} // namespace detail

/**
 * @brief Validate RHS signature compatibility at compile time.
 *
 * @details
 * This function checks that:
 * 1. `Model::rhs(double t, const G&)` is callable, where `G` is the grad type
 *    returned by `Eval::operator()(int, int, int)`.
 * 2. The return type is either `double` or tuple-protocol compatible.
 *
 * @tparam Model The model type with an `rhs` method.
 * @tparam Eval The evaluator type with an `operator()(int, int, int)`.
 *
 * @pre `Eval::operator()(int, int, int)` must be valid and return the grad type.
 */
template <class Model, class Eval>
constexpr void validate_rhs_signature() {
  using namespace pfc::field::detail;

  // Extract the actual grad type G from Eval::operator()
  using G = decltype(std::declval<const Eval&>()(0, 0, 0));

  // Check that Model::rhs is callable with (double, const G&)
  static_assert(
    requires(double t, const G& g) {
      { std::declval<const Model>().rhs(t, g) };
    },
    "Model::rhs(double t, const G&) must be callable with the grad type returned by Eval::operator()"
  );

  // Check that the return type is either scalar or tuple-protocol compatible
  using RhsReturn = decltype(std::declval<const Model>().rhs(0.0, std::declval<const G&>()));

  static_assert(
    std::is_convertible_v<RhsReturn, double> ||
    has_as_tuple<RhsReturn> ||
    is_tuple<RhsReturn>,
    "Model::rhs must return double or a tuple-protocol compatible type (has as_tuple() or is std::tuple)"
  );
}

/**
 * @brief Validate spatial discretization compatibility at runtime.
 *
 * @details
 * This function checks that the evaluator type is a recognized gradient
 * backend. If not, it throws `StepperValidationError` with
 * `ErrorType::IncompatibleBackend`.
 *
 * @tparam Eval The evaluator type to validate.
 *
 * @throws StepperValidationError if the evaluator is not a known type.
 */
template <class Eval>
void validate_spatial_compatibility() {
  using namespace pfc::sim::steppers::detail;

  if constexpr (!is_known_evaluator<Eval>) {
    throw StepperValidationError(
      StepperValidationError::ErrorType::IncompatibleBackend,
      "Evaluator type is not a recognized gradient backend (FDGradient, SpectralGradient, or CompositeGradient)"
    );
  }
}

/**
 * @brief Validate multi-field field count compatibility at compile time.
 *
 * @details
 * This function checks that:
 * 1. `Stepper::field_count` exists and is accessible.
 * 2. `Stepper::RhsType` matches the return type of `Model::rhs`.
 * 3. The field count matches the arity of the model's RHS return type.
 *
 * @tparam Stepper The stepper type with `field_count` and `RhsType`.
 * @tparam Model The model type with an `rhs` method.
 * @tparam Eval The evaluator type with an `operator()(int, int, int)`.
 *
 * @pre `Eval::operator()(int, int, int)` must be valid and return the grad type.
 */
template <class Stepper, class Model, class Eval>
constexpr void validate_field_count() {
  if constexpr (requires { Stepper::field_count; }) {
    // Extract the actual grad type G from Eval::operator()
    using G = decltype(std::declval<const Eval&>()(0, 0, 0));

    // Extract the actual return type of Model::rhs
    using RhsReturn = decltype(std::declval<const Model>().rhs(0.0, std::declval<const G&>()));

    // Verify that Stepper::RhsType matches the model's return type
    static_assert(
      std::is_same_v<typename Stepper::RhsType, RhsReturn>,
      "Stepper::RhsType must match the return type of Model::rhs(double t, const G&)"
    );

    // Check field count
    using namespace pfc::field::detail;
    using RhsTuple = decltype(to_tuple(std::declval<RhsReturn&>()));
    constexpr std::size_t model_fields = std::tuple_size_v<std::remove_cvref_t<RhsTuple>>;

    static_assert(
      Stepper::field_count == model_fields,
      "Multi-field stepper expects N fields, but model.rhs returns M fields"
    );
  }
}

inline StepperValidationError::StepperValidationError(ErrorType type, const std::string& message)
  : std::runtime_error(message), type_(type)
{}

inline StepperValidationError::ErrorType StepperValidationError::error_type() const noexcept
{
  return type_;
}

} // namespace pfc::sim::steppers
