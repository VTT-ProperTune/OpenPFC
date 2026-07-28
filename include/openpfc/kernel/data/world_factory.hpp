// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_factory.hpp
 * @brief World creation and factory functions (DEPRECATED - use pfc::domain::create)
 *
 * @details
 * This file contains legacy factory functions for creating World objects.
 * These functions are **deprecated forwarders** that delegate to the primary
 * `pfc::domain::create_world()` API. Do not use these in new code.
 *
 * **Use `pfc::domain::create_world()`** or related functions in `domain/create.hpp`
 * for all new world construction. These legacy forwarders exist only for backward
 * compatibility and will be removed in a future release as part of the
 * OPENPFC_REFACTORING_EXECUTION_PLAN M2 migration.
 *
 * @see pfc::domain::create_world for the primary world creation API
 * @see domain/create.hpp for the recommended world construction interface
 * @see world.hpp for the core World struct definition
 */

#pragma once

#include <openpfc/kernel/data/strong_types.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/data/world.hpp>

#include <openpfc/domain/create.hpp>

namespace pfc::world {

using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

/**
 * @brief Create a World object with the specified size and default offset
 * and spacing (DEPRECATED - LEGACY FORWARDER).
 *
 * @deprecated This is a **legacy compatibility forwarder**. Do NOT use in new code.
 * Call `pfc::domain::create_world()` directly instead. This function is provided
 * only for backward compatibility and delegates to the primary API.
 *
 * @param dimensions Dimensions of the world.
 * @return A World object.
 *
 * @see pfc::domain::create_world for the primary API
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world(const Int3&) instead")]]
inline CartesianWorld create(const Int3 &size) {
  return pfc::domain::create_world(size);
}

/**
 * @brief Create a World object with strong types for type safety (DEPRECATED - LEGACY FORWARDER).
 *
 * @deprecated This is a **legacy compatibility forwarder**. Do NOT use in new code.
 * Call `pfc::domain::create_world()` directly instead. This function is provided
 * only for backward compatibility and delegates to the primary API.
 *
 * The primary API, `pfc::domain::create_world()`, provides the same functionality and is the
 * recommended interface for creating World objects. Strong types (GridSize, PhysicalOrigin,
 * GridSpacing) make the API self-documenting and prevent parameter confusion at compile time.
 *
 * @param size Grid dimensions (number of points per dimension)
 * @param origin Physical origin of the coordinate system
 * @param spacing Physical spacing between grid points
 * @param periodic Per-axis periodicity flags (default: all periodic). Stored in
 *        the coordinate system and reported by `world::get_periodic` /
 *        `world::is_periodic`.
 * @return A World object with the specified geometry
 *
 * @see pfc::domain::create_world for the primary API
 * @see GridSize, PhysicalOrigin, GridSpacing in strong_types.hpp
 *
 * @code
 * // Primary recommended API
 * GridSize size({256, 256, 256});
 * PhysicalOrigin origin({-128.0, -128.0, -128.0});
 * GridSpacing spacing({1.0, 1.0, 1.0});
 * auto world = pfc::domain::create_world(size, origin, spacing);
 *
 * // Deprecated legacy forwarder (do not use in new code)
 * // auto world = world::create(size, origin, spacing);  // LEGACY - do not use
 * @endcode
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world(const GridSize&, const PhysicalOrigin&, const GridSpacing&, const Bool3&) instead")]]
inline CartesianWorld create(const GridSize &size,
                                    const PhysicalOrigin &origin,
                                    const GridSpacing &spacing,
                                    const Bool3 &periodic = {true, true, true}) {
  return pfc::domain::create_world(size, origin, spacing, periodic);
}

} // namespace pfc::world
