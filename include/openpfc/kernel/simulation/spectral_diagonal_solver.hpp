// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file spectral_diagonal_solver.hpp
 * @brief CPU spectral diagonal scale-and-divide solver (`SolveFunction`)
 *
 * @details
 * Rank-local element-wise solve @f$ s_i = b_i / d_i @f$ for real or complex
 * spectral diagonal coefficients held in `LinearOperatorDesc::operator_context`.
 * Models `pfc::sim::SolveFunction` without a virtual `LinearSolver` base.
 *
 * Diagonals may be `std::vector<double>` or `std::vector<std::complex<double>>`;
 * RHS and target primary fields must match that scalar type.
 *
 * ## Nullspace policies
 *
 * Let @f$ \tau @f$ = `singular_threshold`. Mode @f$ i @f$ is singular when
 * @f$ |d_i| < \tau @f$ (`std::abs` for both real and complex).
 *
 * - **fail:** any singular mode → `ill_conditioned`, no write to `target_out`.
 * - **project:** singular → @f$ s_i = @f$ `null_mode_value` (real path) or
 *   `std::complex<double>(null_mode_value, 0.0)` (complex path); otherwise
 *   @f$ s_i = b_i / d_i @f$. Residual still uses the original @f$ d @f$:
 *   @f$ r_i = d_i s_i - b_i @f$ (for @f$ d_i=0 @f$, @f$ s_i=0 @f$ this yields
 *   @f$ r_i = -b_i @f$; keep @f$ |b_i| @f$ small on null modes when
 *   compatibility is required).
 * - **regularize:** require @f$ \lambda = @f$ `regularization` @f$ > 0 @f$;
 *   @f$ s_i = b_i / (d_i + \lambda) @f$ for every @f$ i @f$ (explicit additive
 *   shift in units of @f$ d @f$, not a silent epsilon inside a plain divide).
 *   Residual uses the **original** @f$ d @f$, so @f$ r @f$ may be nonzero.
 *
 * ## Residual and commit
 *
 * Candidate @f$ s @f$ is formed in solver-owned scratch. Local Euclidean
 * residual norm @f$ \|r\|_2 = \sqrt{\sum |r_i|^2} @f$ with
 * @f$ r_i = d_i s_i - b_i @f$ (real: @f$ |r_i|^2 = r_i^2 @f$). Absolute
 * threshold: `absolute_tolerance.value_or(tolerance)`. On success
 * (`converged`), copy into caller `target_out` (`iteration_count = 1`). On
 * failure, leave `target_out` unchanged.
 *
 * ## Workspace / checkpoint
 *
 * `residual_scratch_` / `complex_residual_scratch_` are recomputable
 * transient state for the candidate solution only. They are **not** part of
 * any checkpoint, `save_state`, or `restore_state` API.
 *
 * @see docs/reference/solver_contract.md
 * @see openpfc/kernel/simulation/solver_contract.hpp
 */

#ifndef OPENPFC_KERNEL_SIMULATION_SPECTRAL_DIAGONAL_SOLVER_HPP
#define OPENPFC_KERNEL_SIMULATION_SPECTRAL_DIAGONAL_SOLVER_HPP

#include "openpfc/kernel/simulation/solver_contract.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace pfc::sim {

namespace detail {

template <typename> constexpr bool spectral_diag_always_false_v = false;

} // namespace detail

/**
 * @brief Singular / near-singular handling for spectral diagonal modes
 */
enum class DiagonalNullspacePolicy {
  fail,       ///< Singular mode → do not mutate target; `ill_conditioned`
  project,    ///< @f$ |d_i| < \tau @f$ → @f$ s_i = @f$ `null_mode_value`
  regularize  ///< @f$ s_i = b_i / (d_i + \lambda) @f$ with @f$ \lambda > 0 @f$
};

/**
 * @brief Configuration for `SpectralDiagonalSolver`
 */
struct SpectralDiagonalConfig {
  DiagonalNullspacePolicy nullspace_policy = DiagonalNullspacePolicy::fail;
  /// @f$ \tau @f$: @f$ |d| < \tau @f$ is singular / near-zero
  double singular_threshold = 1e-14;
  /// Value written for projected null modes (imag part 0 for complex path)
  double null_mode_value = 0.0;
  /// @f$ \lambda @f$ for `regularize`; must be @f$ > 0 @f$ when that policy is selected
  double regularization = 0.0;
};

/**
 * @brief CPU spectral diagonal scale-and-divide solver
 *
 * Header-only value type that models `SolveFunction`. Primary field bundle:
 * a single `std::vector<double>` or `std::vector<std::complex<double>>` (or
 * `std::tie` / `std::ref` wrappers that `to_tuple` to one vector-like field).
 */
class SpectralDiagonalSolver {
public:
  explicit SpectralDiagonalSolver(SpectralDiagonalConfig config = {})
      : config_(std::move(config)) {}

  /**
   * @brief Solve @f$ D s = b @f$ by element-wise scale-and-divide
   *
   * Dispatches to the real or complex path from the primary RHS field type.
   * `target_out` is mutated only on `ConvergenceStatus::converged`.
   *
   * @tparam RHSFields    RHS field bundle (`tuple_protocol`)
   * @tparam TargetFields Caller-owned result storage (`tuple_protocol`)
   */
  template <tuple_protocol RHSFields, tuple_protocol TargetFields>
  auto operator()(const LinearOperatorDesc &op_desc, const RHSFields &rhs,
                  TargetFields &target_out, const SolveOptions &options,
                  const StageContext &ctx) const {
    decltype(auto) rhs_tuple = pfc::field::detail::to_tuple(rhs);
    using Primary = std::remove_cvref_t<
        decltype(unwrap_vector_field(std::get<0>(rhs_tuple)))>;

    if constexpr (std::is_same_v<Primary, std::vector<double>>) {
      return solve_real_(op_desc, rhs, target_out, options, ctx);
    } else if constexpr (std::is_same_v<Primary,
                                        std::vector<std::complex<double>>>) {
      return solve_complex_(op_desc, rhs, target_out, options, ctx);
    } else {
      static_assert(detail::spectral_diag_always_false_v<Primary>,
                    "SpectralDiagonalSolver: primary field must be "
                    "std::vector<double> or std::vector<std::complex<double>>");
    }
  }

private:
  template <tuple_protocol RHSFields, tuple_protocol TargetFields>
  SolveOutcome<std::vector<double> &>
  solve_real_(const LinearOperatorDesc &op_desc, const RHSFields &rhs,
              TargetFields &target_out, const SolveOptions &options,
              const StageContext &ctx) const {
    using Outcome = SolveOutcome<std::vector<double> &>;

    auto fail_outcome = [this](ConvergenceStatus status, double residual,
                               std::string cause) -> Outcome {
      return Outcome{residual_scratch_, status, 0, residual, std::move(cause)};
    };

    if (config_.nullspace_policy == DiagonalNullspacePolicy::regularize &&
        !(config_.regularization > 0.0)) {
      residual_scratch_.clear();
      return fail_outcome(ConvergenceStatus::unknown_failure, 0.0,
                          "SpectralDiagonalSolver: regularization must be > 0 "
                          "for DiagonalNullspacePolicy::regularize");
    }

    if (!op_desc.operator_identifier.empty() &&
        op_desc.operator_identifier != "spectral_diagonal") {
      residual_scratch_.clear();
      return fail_outcome(
          ConvergenceStatus::unknown_failure, 0.0,
          "SpectralDiagonalSolver: unsupported operator_identifier '" +
              op_desc.operator_identifier +
              "' (expected empty or \"spectral_diagonal\")");
    }

    const auto *diag_ptr =
        std::get_if<std::vector<double>>(&op_desc.operator_context);
    if (diag_ptr == nullptr) {
      residual_scratch_.clear();
      return fail_outcome(ConvergenceStatus::ill_conditioned, 0.0,
                          "SpectralDiagonalSolver: operator_context must hold "
                          "std::vector<double> diagonal coefficients");
    }
    const std::vector<double> &diag = *diag_ptr;

    decltype(auto) rhs_tuple = pfc::field::detail::to_tuple(rhs);
    decltype(auto) target_tuple = pfc::field::detail::to_tuple(target_out);
    const auto &rhs_vec = unwrap_vector_field(std::get<0>(rhs_tuple));
    auto &target_vec = unwrap_vector_field(std::get<0>(target_tuple));

    if (diag.size() != rhs_vec.size() || diag.size() != target_vec.size()) {
      residual_scratch_.clear();
      return fail_outcome(
          ConvergenceStatus::ill_conditioned, 0.0,
          "SpectralDiagonalSolver: diagonal, RHS, and target sizes must match");
    }

    const std::size_t n = diag.size();
    residual_scratch_.assign(n, 0.0);

    if (config_.nullspace_policy == DiagonalNullspacePolicy::fail) {
      for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(diag[i]) < config_.singular_threshold) {
          residual_scratch_.clear();
          return fail_outcome(
              ConvergenceStatus::ill_conditioned, 0.0,
              "SpectralDiagonalSolver: singular mode at index " +
                  std::to_string(i) + " (DiagonalNullspacePolicy::fail)");
        }
      }
    }

    const double lambda = config_.regularization;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = diag[i];
      const double b = rhs_vec[i];
      switch (config_.nullspace_policy) {
      case DiagonalNullspacePolicy::fail:
        residual_scratch_[i] = b / d;
        break;
      case DiagonalNullspacePolicy::project:
        if (std::abs(d) < config_.singular_threshold) {
          residual_scratch_[i] = config_.null_mode_value;
        } else {
          residual_scratch_[i] = b / d;
        }
        break;
      case DiagonalNullspacePolicy::regularize:
        residual_scratch_[i] = b / (d + lambda);
        break;
      }
    }

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double r = diag[i] * residual_scratch_[i] - rhs_vec[i];
      sum_sq += r * r;
    }
    // Seam exercise only: ExecutionService::global_reduce does not write back.
    ctx.execution_service.global_reduce({sum_sq}, MPI_SUM);
    const double residual_norm = std::sqrt(sum_sq);

    const double abs_tol =
        options.absolute_tolerance.value_or(options.tolerance);
    if (!(residual_norm <= abs_tol)) {
      return Outcome{residual_scratch_, ConvergenceStatus::max_iterations_reached,
                     1, residual_norm,
                     std::string("SpectralDiagonalSolver: residual norm ") +
                         std::to_string(residual_norm) +
                         " exceeds absolute tolerance " +
                         std::to_string(abs_tol)};
    }

    target_vec = residual_scratch_;
    return Outcome{target_vec, ConvergenceStatus::converged, 1, residual_norm,
                   std::nullopt};
  }

  template <tuple_protocol RHSFields, tuple_protocol TargetFields>
  SolveOutcome<std::vector<std::complex<double>> &>
  solve_complex_(const LinearOperatorDesc &op_desc, const RHSFields &rhs,
                 TargetFields &target_out, const SolveOptions &options,
                 const StageContext &ctx) const {
    using Complex = std::complex<double>;
    using Outcome = SolveOutcome<std::vector<Complex> &>;

    auto fail_outcome = [this](ConvergenceStatus status, double residual,
                               std::string cause) -> Outcome {
      return Outcome{complex_residual_scratch_, status, 0, residual,
                     std::move(cause)};
    };

    if (config_.nullspace_policy == DiagonalNullspacePolicy::regularize &&
        !(config_.regularization > 0.0)) {
      complex_residual_scratch_.clear();
      return fail_outcome(ConvergenceStatus::unknown_failure, 0.0,
                          "SpectralDiagonalSolver: regularization must be > 0 "
                          "for DiagonalNullspacePolicy::regularize");
    }

    if (!op_desc.operator_identifier.empty() &&
        op_desc.operator_identifier != "spectral_diagonal") {
      complex_residual_scratch_.clear();
      return fail_outcome(
          ConvergenceStatus::unknown_failure, 0.0,
          "SpectralDiagonalSolver: unsupported operator_identifier '" +
              op_desc.operator_identifier +
              "' (expected empty or \"spectral_diagonal\")");
    }

    const auto *diag_ptr =
        std::get_if<std::vector<Complex>>(&op_desc.operator_context);
    if (diag_ptr == nullptr) {
      complex_residual_scratch_.clear();
      return fail_outcome(
          ConvergenceStatus::ill_conditioned, 0.0,
          "SpectralDiagonalSolver: operator_context must hold "
          "std::vector<std::complex<double>> diagonal coefficients");
    }
    const std::vector<Complex> &diag = *diag_ptr;

    decltype(auto) rhs_tuple = pfc::field::detail::to_tuple(rhs);
    decltype(auto) target_tuple = pfc::field::detail::to_tuple(target_out);
    const auto &rhs_vec = unwrap_vector_field(std::get<0>(rhs_tuple));
    auto &target_vec = unwrap_vector_field(std::get<0>(target_tuple));

    if (diag.size() != rhs_vec.size() || diag.size() != target_vec.size()) {
      complex_residual_scratch_.clear();
      return fail_outcome(
          ConvergenceStatus::ill_conditioned, 0.0,
          "SpectralDiagonalSolver: diagonal, RHS, and target sizes must match");
    }

    const std::size_t n = diag.size();
    complex_residual_scratch_.assign(n, Complex{0.0, 0.0});

    if (config_.nullspace_policy == DiagonalNullspacePolicy::fail) {
      for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(diag[i]) < config_.singular_threshold) {
          complex_residual_scratch_.clear();
          return fail_outcome(
              ConvergenceStatus::ill_conditioned, 0.0,
              "SpectralDiagonalSolver: singular mode at index " +
                  std::to_string(i) + " (DiagonalNullspacePolicy::fail)");
        }
      }
    }

    const double lambda = config_.regularization;
    const Complex projected{config_.null_mode_value, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
      const Complex d = diag[i];
      const Complex b = rhs_vec[i];
      switch (config_.nullspace_policy) {
      case DiagonalNullspacePolicy::fail:
        complex_residual_scratch_[i] = b / d;
        break;
      case DiagonalNullspacePolicy::project:
        if (std::abs(d) < config_.singular_threshold) {
          complex_residual_scratch_[i] = projected;
        } else {
          complex_residual_scratch_[i] = b / d;
        }
        break;
      case DiagonalNullspacePolicy::regularize:
        complex_residual_scratch_[i] = b / (d + lambda);
        break;
      }
    }

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const Complex r =
          diag[i] * complex_residual_scratch_[i] - rhs_vec[i];
      sum_sq += std::norm(r);
    }
    // Seam exercise only: ExecutionService::global_reduce does not write back.
    ctx.execution_service.global_reduce({sum_sq}, MPI_SUM);
    const double residual_norm = std::sqrt(sum_sq);

    const double abs_tol =
        options.absolute_tolerance.value_or(options.tolerance);
    if (!(residual_norm <= abs_tol)) {
      return Outcome{
          complex_residual_scratch_, ConvergenceStatus::max_iterations_reached,
          1, residual_norm,
          std::string("SpectralDiagonalSolver: residual norm ") +
              std::to_string(residual_norm) +
              " exceeds absolute tolerance " + std::to_string(abs_tol)};
    }

    target_vec = complex_residual_scratch_;
    return Outcome{target_vec, ConvergenceStatus::converged, 1, residual_norm,
                   std::nullopt};
  }

  template <typename Field>
  static decltype(auto) unwrap_vector_field(Field &&field) {
    using U = std::remove_cvref_t<Field>;
    if constexpr (std::is_same_v<U, std::reference_wrapper<std::vector<double>>> ||
                  std::is_same_v<
                      U, std::reference_wrapper<const std::vector<double>>> ||
                  std::is_same_v<
                      U,
                      std::reference_wrapper<std::vector<std::complex<double>>>> ||
                  std::is_same_v<U, std::reference_wrapper<
                                        const std::vector<std::complex<double>>>>) {
      return field.get();
    } else {
      return std::forward<Field>(field);
    }
  }

  SpectralDiagonalConfig config_;
  /**
   * Candidate solution scratch (recomputable transient state).
   * Not part of any checkpoint / save_state / restore_state API.
   */
  mutable std::vector<double> residual_scratch_;
  mutable std::vector<std::complex<double>> complex_residual_scratch_;
};

} // namespace pfc::sim

#endif // OPENPFC_KERNEL_SIMULATION_SPECTRAL_DIAGONAL_SOLVER_HPP
