// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_factory.hpp
 * @brief World creation and factory functions
 *
 * @details
 * This file contains factory functions for creating World objects with various
 * construction patterns. It provides multiple overloads of create() to support
 * different use cases:
 *
 * - create(size) - Simple creation with defaults (`Int3` grid size)
 * - create(GridSize, PhysicalOrigin, GridSpacing) - Full specification (type-safe)
 *
 * The factory functions handle coordinate system construction and validation,
 * making World creation convenient and safe.
 *
 * @see world.hpp for the core World struct definition
 * @see world_helpers.hpp for convenience constructors like uniform(), from_bounds()
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
 * and spacing.
 * @param dimensions Dimensions of the world.
 * @return A World object.
 *
 * @deprecated Use pfc::domain::create_world(const Int3&) instead
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world(const Int3&) instead")]]
inline CartesianWorld create(const Int3 &size) {
  return pfc::domain::create_world(size);
}

/**
 * @brief Create a World object with strong types for type safety
 *
 * This function is **deprecated**. Use `pfc::domain::create_world()` instead.
 *
 * The new API in `pfc::domain::create_world()` provides the same functionality and is the
 * preferred API for creating World objects. Strong types (GridSize, PhysicalOrigin,
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
 * @deprecated Use pfc::domain::create_world(const GridSize&, const PhysicalOrigin&, const GridSpacing&, const Bool3&) instead
 *
 * @code
 * // Preferred new API
 * GridSize size({256, 256, 256});
 * PhysicalOrigin origin({-128.0, -128.0, -128.0});
 * GridSpacing spacing({1.0, 1.0, 1.0});
 * auto world = pfc::domain::create_world(size, origin, spacing);
 *
 * // Old deprecated API (still works but not recommended)
 * // auto world2 = world::create(size, origin, spacing);
 * @endcode
 *
 * @see pfc::domain::create_world for the new replacement API
 * @see GridSize, PhysicalOrigin, GridSpacing in strong_types.hpp
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world(const GridSize&, const PhysicalOrigin&, const GridSpacing&, const Bool3&) instead")]]
inline CartesianWorld create(const GridSize &size,
                                    const PhysicalOrigin &origin,
                                    const GridSpacing &spacing,
                                    const Bool3 &periodic = {true, true, true}) {
  return pfc::domain::create_world(size, origin, spacing, periodic);
}

} // namespace pfc::world
