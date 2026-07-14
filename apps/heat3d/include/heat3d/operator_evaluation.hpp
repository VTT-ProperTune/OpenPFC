// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file operator_evaluation.hpp
 * @brief Heat3D operator-evaluation contract (scalar field).
 *
 * @details
 * This file establishes the operator-evaluation contract for the 3D heat
 * equation \f$\partial_t u = D \nabla^2 u\f$, demonstrating the
 * observational non-mutation semantics required for physics-model-independent
 * time integration.
 *
 * Key contract elements:
 * - `HeatOperator::evaluate()` accepts read-only `HeatGrads` input and returns
 *   a `HeatOperatorResult` wrapper containing the RHS value \f$\partial_t u\f$.
 * - Input state is never modified (const-correctness enforced throughout).
 * - Write mode is declared as `overwrite` (caller provides uninitialized storage).
 * - Stage context includes evaluation time `t` (accepted even though Heat3D RHS
 *   is time-independent).
 *
 * This contract enables future Runge-Kutta, IMEX, exponential-integration, and
 * adaptive-time-stepping implementations to work with unified physics models
 * through the orchestration seam.
 *
 * @see apps/heat3d/heat_model.hpp for the underlying HeatModel implementation.
 * @see include/openpfc/kernel/simulation/operator_result.hpp for the error
 *      propagation contract.
 */

#include <heat3d/heat_model.hpp>
#include <openpfc/kernel/simulation/operator_result.hpp>

namespace heat3d {

/** Write-mode declaration for HeatOperator output mutation behavior. */
enum class WriteMode { overwrite, accumulate, in_place_alias };

/** HeatOperator result wrapper around d_u RHS value. */
struct HeatOperatorResult {
  /** Time derivative \f$\partial_t u = D \nabla^2 u\f$. */
  double d_u;
};

/**
 * @brief Operator-evaluation wrapper for Heat3D physics (scalar field).
 *
 * @details
 * `HeatOperator::evaluate()` computes the RHS of the 3D heat equation while
 * maintaining the operator-evaluation contract:
 *
 * - **Non-mutation**: The `const HeatGrads&` parameter guarantees input state
 *   is never modified.
 * - **Write mode**: Declared as `overwrite`, meaning output storage is written
 *   entirely (caller may provide uninitialized `HeatOperatorResult`).
 * - **Stage context**: Evaluation time `t` is accepted for compatibility with
 *   the orchestration seam, even though the heat equation RHS is
 *   time-independent.
 * - **Observational**: The function does not allocate persistent storage or
 *   modify external state; it only reads inputs and computes outputs.
 *
 * The implementation delegates to the legacy `HeatModel::rhs()` method,
 * preserving numerical equivalence with existing simulations.
 */
struct HeatOperator {
  static constexpr WriteMode write_mode = WriteMode::overwrite;

  /**
   * @brief Evaluate RHS: \f$\partial_t u = D \nabla^2 u\f$ at time t.
   *
   * @param g Read-only Laplacian components (xx, yy, zz second derivatives).
   * @param t Evaluation time (accepted for stage context; Heat3D RHS is
   *          time-independent, so this parameter does not affect the result).
   * @return `HeatOperatorResult` containing \f$\partial_t u\f$.
   *
   * @post The input `HeatGrads` object is unchanged (const-correctness).
   * @post The function allocates no persistent storage; temporaries are
   *       stack-allocated.
   * @note This function is `noexcept` for optimal inlining in hot loops.
   */
  [[nodiscard]] HeatOperatorResult evaluate(const HeatGrads& g, double t) const noexcept {
    // Call legacy HeatModel::rhs semantics (const-correct)
    HeatModel model; // Stateless: kD is compile-time constant
    return HeatOperatorResult{model.rhs(t, g)};
  }
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

} // namespace heat3d
