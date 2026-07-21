// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_context.hpp
 * @brief Stage context for MPI coordination and boundary conditions
 *
 * @details
 * This header provides `pfc::integrator::StageContext`, a simple struct that
 * carries information required for MPI halo exchange timing and boundary
 * condition application.
 *
 * Stage context enables orchestration to apply boundary conditions and perform
 * halo exchanges at required stages without understanding method internals.
 * Drivers should map the context through `requirements_from` and call
 * `pfc::communication::StagePreparationService::prepare` rather than embedding
 * ad-hoc MPI calls at each evaluation site.
 *
 * Name collision: `pfc::sim::StageContext` in
 * `include/openpfc/kernel/simulation/solver_contract.hpp` is a different type
 * (`evaluation_time` + `ExecutionService&`). Always use the fully qualified
 * name `pfc::integrator::StageContext` at call sites in this design slice.
 *
 * Design rationale:
 * - Explicit timing: Carries time, dt, and stage index for coordination
 * - Region requirements: Specifies interior vs boundary access patterns
 * - Boundary condition flags: Indicates when BCs need to be applied
 * - Halo exchange flags: Indicates when halo exchange is needed
 *
 * @see kernel/integrator/workspace.hpp for integrator-owned storage
 * @see openpfc/kernel/decomposition/stage_preparation.hpp for the executable
 *      prepare protocol (`StagePreparationService`)
 */

#include <openpfc/kernel/decomposition/stage_preparation.hpp>

namespace pfc::integrator {

/**
 * @brief Stage context for MPI coordination and boundary conditions
 *
 * @details
 * `pfc::integrator::StageContext` carries information from integrators to
 * drivers/orchestration about what needs to happen at each evaluation stage:
 *
 * - Time information: Current time and timestep being attempted (`time`, `dt`)
 * - Stage index: RK stage index or method-specific stage identifier
 * - Region requirements: Interior vs boundary vs all cells (`region_kind`)
 * - Boundary conditions: Whether BCs need to be applied (`needs_boundary_update`)
 * - Halo exchange: Whether halo exchange is needed (`needs_halo_exchange`)
 *
 * Prefer `requirements_from(*this)` + `StagePreparationService::prepare` for
 * CPU/MPI padded-brick stages. Post-evaluation BC enforcement (after writing
 * new owned values) remains a separate driver responsibility outside `prepare`.
 *
 * @note Distinct from `pfc::sim::StageContext` in `solver_contract.hpp`.
 * @see pfc::communication::StagePreparationService
 * @see requirements_from
 */
struct StageContext {
  /**
   * @brief Current evaluation time
   */
  double time;

  /**
   * @brief Timestep being attempted
   */
  double dt;

  /**
   * @brief Stage index (e.g., RK stage index)
   */
  int stage_index;

  /**
   * @brief Field region kind for this evaluation
   *
   * Specifies which region of the field will be accessed:
   * - Interior: Only interior cells (no boundary access)
   * - Boundary: Only boundary cells (for BC application)
   * - All: All cells (interior + boundary)
   *
   * Numeric values align with `pfc::communication::RegionKind`.
   */
  enum class RegionKind {
    Interior, ///< Interior cells only
    Boundary, ///< Boundary cells only
    All       ///< All cells (interior + boundary)
  } region_kind;

  /**
   * @brief Whether boundary conditions need preparation for this stage
   *
   * When drivers use `StagePreparationService`, a true value means: run the
   * injectable boundary hook inside `prepare` (pre-evaluation), ordered vs
   * halo exchange according to `BoundaryHaloOrder` (default:
   * boundary then halo). Post-evaluation BC enforcement after writing new
   * owned values remains a separate driver responsibility outside `prepare`.
   */
  bool needs_boundary_update;

  /**
   * @brief Whether halo exchange is needed
   *
   * If true, ghost layers must be consistent with neighbor owned cores
   * **before** this evaluation. `StagePreparationService::prepare` performs
   * the exchange on bound fields when this flag is set. Drivers that still
   * call exchangers directly should exchange before evaluation as well.
   */
  bool needs_halo_exchange;
};

/**
 * @brief Map integrator `StageContext` flags to stage-preparation requirements.
 *
 * Copies `needs_halo_exchange` / `needs_boundary_update` and maps
 * `RegionKind` onto `pfc::communication::RegionKind`. Ordering defaults to
 * `BoundaryThenHalo`.
 */
inline pfc::communication::StagePreparationRequirements
requirements_from(const StageContext &ctx) {
  pfc::communication::StagePreparationRequirements req;
  switch (ctx.region_kind) {
  case StageContext::RegionKind::Interior:
    req.region_kind = pfc::communication::RegionKind::Interior;
    break;
  case StageContext::RegionKind::Boundary:
    req.region_kind = pfc::communication::RegionKind::Boundary;
    break;
  case StageContext::RegionKind::All:
    req.region_kind = pfc::communication::RegionKind::All;
    break;
  }
  req.needs_halo_exchange = ctx.needs_halo_exchange;
  req.needs_boundary_update = ctx.needs_boundary_update;
  return req;
}

} // namespace pfc::integrator
