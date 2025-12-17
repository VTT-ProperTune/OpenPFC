// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
 * - create(size) - Simple creation with defaults
 * - create(size, origin, spacing) - Full specification (type-safe)
 * - create(size, offset, spacing) - Legacy API (deprecated)
 *
 * The factory functions handle coordinate system construction and validation,
 * making World creation convenient and safe.
 *
 * @see world.hpp for the core World struct definition
 * @see world_helpers.hpp for convenience constructors like uniform(), from_bounds()
 */

#pragma once

#include "csys.hpp"
#include "strong_types.hpp"
#include "types.hpp"
#include "world.hpp"

namespace pfc {
namespace world {

using pfc::csys::CartesianTag;
using pfc::csys::CoordinateSystem;
using pfc::types::Int3;
using pfc::types::Real3;

/**
 * @brief Create a World object with the specified size and default offset
 * and spacing.
 * @param dimensions Dimensions of the world.
 * @return A World object.
 */
CartesianWorld create(const Int3 &size);

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
 * auto world = world::create(size, origin, spacing);
 *
 * // Won't compile if parameters are swapped
 * // auto bad = world::create(spacing, size, origin);  // Compile error!
 * @endcode
 *
 * @see GridSize, PhysicalOrigin, GridSpacing in strong_types.hpp
 * @see create(Int3, Real3, Real3) for legacy API (deprecated)
 */
CartesianWorld create(const GridSize &size, const PhysicalOrigin &origin,
                      const GridSpacing &spacing);

/**
 * @brief Create a World object with raw arrays (DEPRECATED)
 *
 * @deprecated Use create(GridSize, PhysicalOrigin, GridSpacing) for type safety.
 * This overload is ambiguous - it's unclear which Real3 is offset vs spacing.
 * The strong-type API prevents parameter confusion at compile time.
 *
 * @param size Grid dimensions
 * @param offset Physical offset (origin) of coordinate system
 * @param spacing Physical spacing between grid points
 * @return A World object
 *
 * **Migration guide:**
 * @code
 * // Old (ambiguous):
 * auto world = world::create({256, 256, 256}, {0, 0, 0}, {1, 1, 1});
 *
 * // New (type-safe):
 * auto world = world::create(
 *     GridSize({256, 256, 256}),
 *     PhysicalOrigin({0, 0, 0}),
 *     GridSpacing({1, 1, 1})
 * );
 * @endcode
 *
 * @see create(GridSize, PhysicalOrigin, GridSpacing) for new API
 */
[[deprecated("Use create(GridSize, PhysicalOrigin, GridSpacing) for type safety. "
             "See migration guide in documentation.")]] CartesianWorld
create(const Int3 &size, const Real3 &offset, const Real3 &spacing);

} // namespace world
} // namespace pfc
