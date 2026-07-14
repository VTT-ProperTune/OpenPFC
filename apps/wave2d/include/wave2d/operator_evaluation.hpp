// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file operator_evaluation.hpp
 * @brief Wave2D operator-evaluation contract (coupled multi-field system).
 *
 * @details
 * This file establishes the operator-evaluation contract for the 2D wave
 * equation as a coupled first-order system:
 * \f$\partial_t u = v\f$, \f$\partial_t v = c^2 \Delta u\f$, demonstrating the
 * observational non-mutation semantics for multi-field physics models.
 *
 * Key contract elements:
 * - `WaveOperator::evaluate()` accepts read-only `WaveLaplacian` input, the
 *   current velocity value `v_val`, and evaluation time `t`, returning a
 *   `WaveOperatorResult` wrapper containing coupled increments `(du, dv)`.
 * - Input state is never modified (const-correctness enforced throughout).
 * - Write mode is declared as `overwrite` (caller provides uninitialized storage).
 * - Multi-field support via tuple protocol: `WaveOperatorResult` provides
 *   `as_tuple()` methods for scattering `du` and `dv` into separate field buffers.
 * - Stage context includes evaluation time `t` (accepted even though Wave2D RHS
 *   is time-independent) and metric scaling via `inv_dx2` / `inv_dy2`.
 *
 * This contract enables future Runge-Kutta, IMEX, exponential-integration, and
 * adaptive-time-stepping implementations to work with unified multi-field physics
 * models through the orchestration seam.
 *
 * @see apps/wave2d/wave_model.hpp for the underlying WaveModel implementation.
 * @see include/openpfc/kernel/simulation/operator_result.hpp for the error
 *      propagation contract.
 */

#include <wave2d/wave_model.hpp>
#include <openpfc/kernel/simulation/operator_result.hpp>

namespace wave2d {

/** Write-mode declaration for WaveOperator output mutation behavior. */
enum class WriteMode { overwrite, accumulate, in_place_alias };

/** WaveOperator result wrapper for coupled multi-field increments (du, dv). */
struct WaveOperatorResult {
  /** Increments (du, dv) for the coupled u/v field system. */
  WaveIncrements increments;

  /**
   * @brief Tuple protocol for scattering increments into separate field buffers.
   *
   * @return `std::tuple<double&, double&>` providing mutable references to
   *         `du` and `dv`.
   *
   * @details This enables the orchestration seam to write coupled increments
   * into separate field storage without requiring a custom field bundle type.
   * The tuple protocol is already used by `WaveIncrements::as_tuple()`, so
   * `WaveOperatorResult` simply forwards the call for convenience.
   */
  auto as_tuple() { return increments.as_tuple(); }

  /** @overload Const overload for read-only access. */
  auto as_tuple() const { return increments.as_tuple(); }
};

/**
 * @brief Operator-evaluation wrapper for Wave2D physics (coupled u/v fields).
 *
 * @details
 * `WaveOperator::evaluate()` computes the RHS of the 2D wave equation as a
 * coupled first-order system while maintaining the operator-evaluation contract:
 *
 * - **Non-mutation**: The `const WaveLaplacian&` parameter guarantees input state
 *   is never modified.
 * - **Multi-field support**: Returns coupled increments `(du, dv)` where
 *   `du = v` and `dv = c² Δu`, using the tuple protocol for scattering.
 * - **Write mode**: Declared as `overwrite`, meaning output storage is written
 *   entirely (caller may provide uninitialized `WaveOperatorResult`).
 * - **Stage context**: Evaluation time `t` is accepted for compatibility with
 *   the orchestration seam, even though the wave equation RHS is
 *   time-independent. Metric scaling (`inv_dx2`, `inv_dy2`) is provided at
 *   construction time.
 * - **Observational**: The function does not allocate persistent storage or
 *   modify external state; it only reads inputs and computes outputs.
 *
 * The implementation delegates to the legacy `WaveModel::rhs()` method,
 * preserving numerical equivalence with existing simulations.
 */
struct WaveOperator {
  static constexpr WriteMode write_mode = WriteMode::overwrite;

  /**
   * @brief Construct WaveOperator with metric scaling factors.
   *
   * @param inv_dx2 Inverse squared grid spacing in x-direction (1/dx²).
   * @param inv_dy2 Inverse squared grid spacing in y-direction (1/dy²).
   *
   * @note Metric factors are stored at construction time and passed to the
   *       underlying `WaveModel` during each evaluation.
   */
  WaveOperator(double inv_dx2, double inv_dy2) noexcept
    : model_{.inv_dx2 = inv_dx2, .inv_dy2 = inv_dy2} {}

  /**
   * @brief Evaluate RHS: \f$du = v\f$, \f$dv = c^2 \Delta u\f$ at time t.
   *
   * @param lap Read-only Laplacian components (lxx, lyy unscaled 3-point sums).
   * @param v_val Current velocity value at the grid point.
   * @param t Evaluation time (accepted for stage context; Wave2D RHS is
   *          time-independent, so this parameter does not affect the result).
   * @return `WaveOperatorResult` containing coupled increments `(du, dv)`.
   *
   * @post The input `WaveLaplacian` object is unchanged (const-correctness).
   * @post The function allocates no persistent storage; temporaries are
   *       stack-allocated.
   * @note This function is `noexcept` for optimal inlining in hot loops.
   */
  [[nodiscard]] WaveOperatorResult evaluate(const WaveLaplacian& lap, double v_val,
                                            double t) const noexcept {
    // Call legacy WaveModel::rhs semantics (const-correct)
    return WaveOperatorResult{model_.rhs(t, v_val, lap)};
  }

private:
  /** Underlying WaveModel with metric scaling factors. */
  WaveModel model_;
};

/*
 * Legacy Model::step() call sites (excluded from migration scope):
 *
 * aluminumNew:
 *   - m.step(t) in apps/aluminumNew/Aluminum.hpp (via free function step())
 *   - aluminum.step(1.0) in apps/aluminumNew/aluminumTest.cpp
 *
 * tungsten:
 *   - model.step(t) in apps/tungsten/include/tungsten/cpu/tungsten.hpp (via step())
 *   - model.step(t) in apps/tungsten/include/tungsten/cuda/tungsten.hpp (via step())
 *   - model.step(t) in apps/tungsten/include/tungsten/hip/tungsten.hpp (via step())
 *   - model.step(t) in apps/tungsten/include/tungsten/common/run_tungsten_gpu_vtk.hpp
 *   - model.step(0.0) multiple times in apps/tungsten/src/tungsten_scalability.cpp
 *   - tungsten.step(1.0) in apps/tungsten/tests/test_tungsten.cpp
 *   - model_cpu.step(0.0), model_cuda.step(0.0) in apps/tungsten/tests/test_tungsten_cpu_vs_cuda.cpp
 *   - model_cpu.step(0.0), model_hip.step(0.0) in apps/tungsten/tests/test_tungsten_cpu_vs_hip.cpp
 *
 * Note: Heat3D and Wave2D use explicit-Euler stepper.step(), not Model::step().
 */

} // namespace wave2d
