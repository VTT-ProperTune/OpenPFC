// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file create.hpp
 * @brief `pfc::domain::create` — World and Domain factory functions (M1).
 *
 * @details
 * This file contains factory functions for creating World objects in the
 * `pfc::domain` namespace. These provide the canonical API for world creation,
 * replacing the deprecated `pfc::world::create` functions.
 *
 * For Domain-only creation (without World wrapper), see the overloads in
 * `domain.hpp` that return `pfc::Domain`.
 *
 * @see domain.hpp for Domain creation and query functions
 * @see world_factory.hpp for deprecated world::create forwarders
 */

#pragma once

#include <openpfc/kernel/data/domain.hpp>

// Forward declaration to avoid circular dependency
namespace pfc::world {
  class World;
}

namespace pfc {

using types::Bool3;
using types::Int3;
using types::Real3;

namespace domain {

/**
 * @brief Create a World object with the specified size and default offset
 * and spacing.
 *
 * @param dimensions Dimensions of the world.
 * @return A World object.
 */
[[nodiscard]] world::World create_world(const Int3 &size);

/**
 * @brief Create a World object with strong types for type safety
 *
 * This is the **preferred** API for creating World objects. Strong types
 * (GridSize, PhysicalOrigin, GridSpacing) make the API self-documenting
 * and prevent parameter confusion at compile time.
 *
 * @param size Grid dimensions (number of points per dimension)
 * @param origin Physical origin of the coordinate system
 * @param spacing Physical spacing between grid points
 * @param periodic Per-axis periodicity flags (default: all periodic). Stored in
 *        the coordinate system and reported by `world::get_periodic` /
 *        `world::is_periodic`.
 * @return A World object with the specified geometry
 *
 * @note Zero overhead - strong types compile away completely
 * @note Type-safe - compiler catches parameter order mistakes
 *
 * @code
 * // Clear and type-safe
 * GridSize size({256, 256, 256});
 * PhysicalOrigin origin({-128.0, -128.0, -128.0});
 * GridSpacing spacing({1.0, 1.0, 1.0});
 * auto world = domain::create_world(size, origin, spacing);
 *
 * // Won't compile if parameters are swapped
 * // auto bad = domain::create_world(spacing, size, origin);  // Compile error!
 * @endcode
 *
 * @see GridSize, PhysicalOrigin, GridSpacing in strong_types.hpp
 */
[[nodiscard]] world::World create_world(const GridSize &size,
                           const PhysicalOrigin &origin,
                           const GridSpacing &spacing,
                           const Bool3 &periodic = {true, true, true});

} // namespace domain

} // namespace pfc
