// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_context.hpp
 * @brief Single stage context for MPI coordination, BCs, and solver evaluation
 *
 * @details
 * `pfc::integrator::StageContext` is the one stage-context type. Integrators
 * fill timing and region/halo/BC flags; solvers bind an `ExecutionService`
 * for distributed reductions. `pfc::sim::StageContext` is an alias of this
 * type (`solver_contract.hpp`).
 *
 * Drivers should map the context through `requirements_from` and call
 * `pfc::communication::StagePreparationService::prepare` rather than embedding
 * ad-hoc MPI calls at each evaluation site.
 *
 * @see kernel/integrator/workspace.hpp for integrator-owned storage
 * @see openpfc/kernel/decomposition/stage_preparation.hpp for the executable
 *      prepare protocol (`StagePreparationService`)
 */

#include <stdexcept>

#include <openpfc/kernel/decomposition/stage_preparation.hpp>

namespace pfc::sim {
class ExecutionService;
}

namespace pfc::integrator {

/**
 * @brief Stage context for MPI coordination, boundary conditions, and solves
 *
 * @details
 * Carries information from integrators and implicit solvers about what needs
 * to happen at each evaluation stage:
 *
 * - Time information: evaluation time and timestep being attempted (`time`,
 *   `dt`). Solvers historically called `time` `evaluation_time`.
 * - Stage index: RK stage index or method-specific stage identifier
 * - Region requirements: Interior vs boundary vs all cells (`region_kind`)
 * - Boundary conditions: Whether BCs need to be applied (`needs_boundary_update`)
 * - Halo exchange: Whether halo exchange is needed (`needs_halo_exchange`)
 * - Execution service: optional driver hook for distributed solver work
 *
 * Prefer `requirements_from(*this)` + `StagePreparationService::prepare` for
 * CPU/MPI padded-brick stages. Post-evaluation BC enforcement (after writing
 * new owned values) remains a separate driver responsibility outside `prepare`.
 *
 * @see pfc::communication::StagePreparationService
 * @see requirements_from
 */
struct StageContext {
  /**
   * @brief Current evaluation time
   */
  double time = 0.0;

  /**
   * @brief Timestep being attempted
   */
  double dt = 0.0;

  /**
   * @brief Stage index (e.g., RK stage index)
   */
  int stage_index = 0;

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
  } region_kind = RegionKind::All;

  /**
   * @brief Whether boundary conditions need preparation for this stage
   *
   * When drivers use `StagePreparationService`, a true value means: run the
   * injectable boundary hook inside `prepare` (pre-evaluation), ordered vs
   * halo exchange according to `BoundaryHaloOrder` (default:
   * boundary then halo). Post-evaluation BC enforcement after writing new
   * owned values remains a separate driver responsibility outside `prepare`.
   */
  bool needs_boundary_update = false;

  /**
   * @brief Whether halo exchange is needed
   *
   * If true, ghost layers must be consistent with neighbor owned cores
   * **before** this evaluation. `StagePreparationService::prepare` performs
   * the exchange on bound fields when this flag is set. Drivers that still
   * call exchangers directly should exchange before evaluation as well.
   */
  bool needs_halo_exchange = false;

  /**
   * @brief Driver execution service for distributed solver operations
   *
   * Null when the context is used only for halo/BC flags. Solvers that call
   * `service()` require a non-null pointer.
   */
  pfc::sim::ExecutionService *execution_service = nullptr;

  /**
   * @brief Bound `ExecutionService` (throws if `execution_service` is null)
   */
  [[nodiscard]] pfc::sim::ExecutionService &service() const {
    if (execution_service == nullptr) {
      throw std::logic_error("StageContext: ExecutionService is not bound");
    }
    return *execution_service;
  }
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
