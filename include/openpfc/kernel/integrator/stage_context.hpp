// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file stage_context.hpp
 * @brief Stage context for MPI coordination and boundary conditions
 *
 * @details
 * This header provides StageContext, a simple struct that carries information
 * required for MPI halo exchange timing and boundary condition application.
 *
 * Stage context enables orchestration to apply boundary conditions and perform
 * halo exchanges at required stages without understanding method internals.
 * The driver reads the context and schedules MPI operations accordingly.
 *
 * Design rationale:
 * - Explicit timing: Carries time, dt, and stage index for coordination
 * - Region requirements: Specifies interior vs boundary access patterns
 * - Boundary condition flags: Indicates when BCs need to be applied
 * - Halo exchange flags: Indicates when halo exchange is needed
 *
 * @see kernel/integrator/workspace.hpp for integrator-owned storage
 */

namespace pfc::integrator {

/**
 * @brief Stage context for MPI coordination and boundary conditions
 *
 * @details
 * StageContext carries information from integrators to drivers/orchestration
 * about what needs to happen at each evaluation stage:
 *
 * - Time information: Current time and timestep being attempted
 * - Stage index: RK stage index or method-specific stage identifier
 * - Region requirements: Interior vs boundary vs all cells
 * - Boundary conditions: Whether BCs need to be applied
 * - Halo exchange: Whether halo exchange is needed
 *
 * This enables orchestration to schedule MPI operations and boundary condition
 * application without understanding method internals.
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
     */
    enum class RegionKind {
        Interior,  ///< Interior cells only
        Boundary,  ///< Boundary cells only
        All        ///< All cells (interior + boundary)
    } region_kind;

    /**
     * @brief Whether boundary conditions need to be applied
     *
     * If true, the driver should apply boundary conditions after this evaluation.
     */
    bool needs_boundary_update;

    /**
     * @brief Whether halo exchange is needed
     *
     * If true, the driver should perform halo exchange before this evaluation.
     * Can be overlapped with computation for non-blocking exchanges.
     */
    bool needs_halo_exchange;
};

} // namespace pfc::integrator
