// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file operator_result.hpp
 * @brief Minimal error propagation contract for operator-evaluation functions.
 *
 * @details
 * This header provides lightweight status and diagnostics types for
 * operator-evaluation functions without prescribing a specific error-handling
 * strategy (exceptions, error codes, or assertions). Operators may use return
 * values or output parameters for status; exceptions are permitted but not
 * required.
 *
 * The contract is intentionally minimal:
 * - `Status`: enumeration of possible outcomes (success, failure, numerical
 *   issues, preparation/execution failures, stability warnings).
 * - `Diagnostics`: optional contextual information (message, error norm, residual
 *   metric) to aid diagnosis and logging.
 *
 * Future orchestration seams may combine these with step-attempt results
 * for integrator/controller decision-making, but operator evaluation itself does
 * not own MPI synchronization or retry logic.
 */

#include <optional>
#include <string>

namespace pfc::sim {

/** Operator-evaluation status codes (not exception-based, not error-code-based). */
enum class Status {
  /** Evaluation completed successfully. */
  success,
  /** General failure without specific classification. */
  failure,
  /** NaN detected in output or intermediate computation. */
  nan_detected,
  /** Numerical overflow detected. */
  overflow,
  /** Preparation phase failed (e.g., workspace allocation, factor precomputation). */
  preparation_failed,
  /** Execution phase failed (e.g., kernel launch, numerical error). */
  execution_failed,
  /** Stability warning (e.g., large gradient, stiff behavior) - not a failure. */
  stability_warning
};

/** Diagnostics associated with operator-evaluation outcomes. */
struct Diagnostics {
  /** Optional human-readable message describing the outcome. */
  std::optional<std::string> message;

  /** Optional error norm (e.g., L2 norm of residual, max absolute error). */
  std::optional<double> error_norm;

  /** Optional residual metric for iterative or consistency checks. */
  std::optional<double> residual_metric;
};

} // namespace pfc::sim
