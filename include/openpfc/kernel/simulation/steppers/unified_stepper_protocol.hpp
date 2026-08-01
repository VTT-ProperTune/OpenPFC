// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file unified_stepper_protocol.hpp
 * @brief Unified stepper protocol for time integration from M6 consolidation
 *
 * @details
 * Defines the single attempt/commit step protocol that all OpenPFC steppers
 * must implement, replacing the fragmented step interfaces across different
 * stepper implementations. This is CPU-only interface design work that can
 * proceed independently of GPU cluster access.
 *
 * **Key Components:**
 *
 * 1. **StepAttemptResult**: Represents the outcome of a step attempt with
 *    error estimation and state management for adaptive control
 *
 * 2. **UnifiedStepProtocol**: Concept that all steppers must satisfy,
 *    requiring attempt/commit semantics and rollback capability
 *
 * 3. **MethodEnum**: Unified enumeration of all integration methods
 *    (Euler, RK2, RK3, RK4, embedded RK, IMEX, ETD)
 *
 * 4. **StateConcepts**: Field-based state abstractions for real/complex
 *    fields and multi-field packs, replacing raw `std::vector<double>` restrictions
 *
 * 5. **StageContext**: Single, unified stage execution context (replacing
 *    multiple duplicate context types)
 *
 * **Protocol Semantics:**
 * - `attempt_step(t, state)` → StepAttemptResult: Try a step, compute error estimate
 * - `commit_step()` → Apply the attempted step permanently
 * - `reject_step()` → Rollback to pre-attempt state
 * - The stepper manages internal state for rollback capability
 *
 * **State Generalization:**
 * - Steppers accept any `Field<T, MemorySpace>` where `T` is `double` or
 * `complex<double>`
 * - Multi-field steppers work with heterogeneous field packs
 * - No assumption about raw `std::vector<double>` storage
 *
 * **Adaptive Control Integration:**
 * - StepAttemptResult provides error_estimates for adaptive controllers
 * - AdaptiveTimeController can shrink/grow dt based on error evidence
 * - Steppers don't implement adaptive logic themselves (controller pattern)
 *
 * @see kernel/simulation/steppers/euler.hpp for stepper pattern being consolidated
 * @see kernel/simulation/steppers/explicit_rk.hpp for explicit RK steppers
 * @see OPENPFC_REFACTORING_EXECUTION_PLAN.md M6 section for consolidation
 * requirements
 * @author OpenPFC Development Team
 * @date 2026
 */

#pragma once

#include <complex>
#include <concepts>
#include <optional>
#include <tuple>
#include <variant>

#include <openpfc/kernel/data/grid_field.hpp>

namespace pfc::sim::steppers {

/**
 * @brief Outcome of a step attempt with error estimation
 *
 * Encapsulates the result of attempting a time step, including success/failure,
 * error estimates for adaptive control, and state management information.
 */
struct StepAttemptResult {
  /**
   * @brief Status of the step attempt
   */
  enum class Status {
    Accepted, ///< Step accepted (error within tolerance)
    Rejected, ///< Step rejected (error exceeds tolerance)
    Failed    ///< Step failed numerically (NaN, overflow, etc.)
  };

  Status status;                        ///< Status of the step attempt
  double new_time;                      ///< Time after the attempted step
  std::optional<double> error_estimate; ///< Error estimate (for adaptive methods)

  /**
   * @brief Check if the step attempt was successful
   */
  [[nodiscard]] bool is_successful() const noexcept {
    return status == Status::Accepted;
  }

  /**
   * @brief Check if the step can be committed
   */
  [[nodiscard]] bool can_commit() const noexcept {
    return status == Status::Accepted;
  }

  /**
   * @brief Check if the step should be rejected
   */
  [[nodiscard]] bool should_reject() const noexcept {
    return status == Status::Rejected || status == Status::Failed;
  }
};

/**
 * @brief Unified integrator method enumeration
 *
 * Consolidates all time integration methods into a single enum, replacing
 * multiple scattered method enums (RKIntegratorMethod, IntegrMethod, etc.)
 */
enum class IntegratorMethod {
  Euler,      ///< Forward Euler (1st order explicit)
  RK2Heun,    ///< Heun's method (2nd order explicit)
  RK3Heun,    ///< 3rd order Heun method
  RK4,        ///< Classical 4th order Runge-Kutta
  EmbeddedRK, ///< Embedded Runge-Kutta (adaptive)
  ImexEuler,  ///< IMEX Euler (explicit-implicit split)
  ETD1,       ///< 1st order Exponential Time Differencing
  Multi       ///< Placeholder for multi-field/multi-method variants
};

/**
 * @brief Get the formal order of accuracy for each integration method
 *
 * @param method Integration method
 * @return Order of accuracy (e.g., 4 for RK4)
 */
[[nodiscard]] inline int method_order(IntegratorMethod method) noexcept {
  switch (method) {
  case IntegratorMethod::Euler: return 1;
  case IntegratorMethod::RK2Heun: return 2;
  case IntegratorMethod::RK3Heun: return 3;
  case IntegratorMethod::RK4: return 4;
  case IntegratorMethod::EmbeddedRK: return 4; // Typically 4(5) embedded
  case IntegratorMethod::ImexEuler: return 1;
  case IntegratorMethod::ETD1: return 1;
  case IntegratorMethod::Multi: return 1; // Placeholder
  }
  return 1; // Unknown method
}

/**
 * @brief Check if a method supports adaptive error estimation
 */
[[nodiscard]] inline bool
method_supports_adaptive(IntegratorMethod method) noexcept {
  switch (method) {
  case IntegratorMethod::EmbeddedRK: return true;
  default: return false;
  }
}

/**
 * @brief Single, unified stage execution context
 *
 * Replaces duplicate StageContext types from stepper_base.hpp and
 * integrator_base.hpp. Provides information about the current RK/IMEX/ETD stage to
 * RHS callables.
 */
struct StageContext {
  double stage_time;       ///< Time at which to evaluate RHS
  double dt;               ///< Current time step
  int stage_number;        ///< Current stage index (0-based)
  int total_stages;        ///< Total number of stages in this step
  IntegratorMethod method; ///< Integration method being used

  /**
   * @brief Check if this is the final stage
   */
  [[nodiscard]] bool is_final_stage() const noexcept {
    return stage_number == total_stages - 1;
  }

  /**
   * @brief Check if this method is explicit (no implicit solves)
   */
  [[nodiscard]] bool is_explicit() const noexcept {
    return method == IntegratorMethod::Euler ||
           method == IntegratorMethod::RK2Heun ||
           method == IntegratorMethod::RK3Heun || method == IntegratorMethod::RK4 ||
           method == IntegratorMethod::EmbeddedRK;
  }

  /**
   * @brief Check if this method uses IMEX splitting
   */
  [[nodiscard]] bool is_imex() const noexcept {
    return method == IntegratorMethod::ImexEuler;
  }

  /**
   * @brief Check if this method uses ETD
   */
  [[nodiscard]] bool is_etd() const noexcept {
    return method == IntegratorMethod::ETD1;
  }
};

/**
 * @brief Single field state concept for stepper state access
 *
 * Unifies state access patterns across different field types (real/complex,
 * various memory spaces). Replaces raw std::vector<double> assumptions.
 */
template <class State>
concept SingleFieldState = requires(State state) {
  // State must provide a way to access field data
  { state.data() } -> std::convertible_to<double *>;
  { state.size() } -> std::convertible_to<std::size_t>;
  // State must support complex field access when applicable
  requires std::same_as<typename State::value_type, double> ||
               std::same_as<typename State::value_type, std::complex<double>>;
};

/**
 * @brief Multi-field state concept for heterogeneous field packs
 *
 * Supports steppers operating on multiple fields simultaneously (e.g.,
 * MultiEulerStepper, MultiExplicitRKStepper, MultiEtd1Stepper).
 * Replaces fixed-2 field count assumptions.
 */
template <class StatePack>
concept MultiFieldState = requires(StatePack pack) {
  // Must support tuple-like access
  { std::get<0>(pack) } -> SingleFieldState;
  { std::tuple_size_v<StatePack> } -> std::convertible_to<std::size_t>;
};

/**
 * @brief Unified stepper protocol concept
 *
 * All steppers must satisfy this concept, providing attempt/commit semantics
 * and the ability to work with generalized state types.
 */
template <class Stepper, class State>
concept UnifiedStepProtocol =
    requires(Stepper stepper, State state, double t, double dt) {
      // Attempt a step - must return StepAttemptResult
      { stepper.attempt_step(t, state) } -> std::same_as<StepAttemptResult>;

      // Commit the attempted step
      { stepper.commit_step() } -> std::same_as<void>;

      // Reject and rollback the attempted step
      { stepper.reject_step() } -> std::same_as<void>;

      // Access current time step
      { stepper.dt() } -> std::same_as<double>;

      // Check if method supports adaptive control
      { stepper.supports_adaptive() } -> std::same_as<bool>;

      // Get integration method
      { stepper.method() } -> std::same_as<IntegratorMethod>;
    };

/**
 * @brief Stepper workspace for temporary storage
 *
 * Single unified workspace type replacing multiple workspace implementations
 * (StageWorkspace, integrator::Workspace, etc.). Provides scratch storage
 * for intermediate computations during stepping.
 */
template <typename T = double, std::size_t Alignment = 64> class StepperWorkspace {
public:
  /**
   * @brief Construct workspace with given capacity
   */
  explicit StepperWorkspace(std::size_t capacity)
      : m_capacity(capacity), m_data(capacity) {}

  /**
   * @brief Access workspace data
   */
  [[nodiscard]] T *data() noexcept { return m_data.data(); }
  [[nodiscard]] const T *data() const noexcept { return m_data.data(); }

  /**
   * @brief Get workspace capacity
   */
  [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

  /**
   * @brief Resize workspace (if capacity needs to change)
   */
  void resize(std::size_t new_capacity) {
    m_capacity = new_capacity;
    m_data.resize(new_capacity);
  }

  /**
   * @brief Clear workspace to zero
   */
  void zero() { std::fill(m_data.begin(), m_data.end(), T{}); }

private:
  std::size_t m_capacity;
  std::vector<T> m_data;
};

/**
 * @brief Result of a commit operation
 *
 * Provides additional information about successful step commit,
 * potentially including performance metrics or solver statistics.
 */
struct StepCommitResult {
  double final_time; ///< Time after the committed step
  double actual_dt;  ///< Actual time step used (may differ from attempted)
  std::optional<double> solver_time; ///< Time spent in implicit solves (IMEX/ETD)

  /**
   * @brief Check if commit was successful
   */
  [[nodiscard]] bool is_successful() const noexcept { return true; }
};

/**
 * @brief Protocol validator for stepper implementations
 *
 * Provides compile-time validation that a stepper properly implements
 * the unified protocol. Used in tests to ensure all steppers conform.
 */
template <class Stepper, class State> struct ValidateUnifiedStepProtocol {
  static constexpr bool value = UnifiedStepProtocol<Stepper, State>;

  static_assert(value, "Stepper does not satisfy UnifiedStepProtocol concept");
};

} // namespace pfc::sim::steppers