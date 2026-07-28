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
#include <openpfc/kernel/data/strong_types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

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
 * and spacing (PRIMARY API).
 *
 * @details This is the recommended interface for constructing World objects.
 * It uses pfc::data::Field (grid_field.hpp) internally and follows the
 * OPENPFC_REFACTORING_EXECUTION_PLAN M2 guidance.
 *
 * @param dimensions Dimensions of the world.
 * @return A World object.
 *
 * @see For more flexible construction, see create_world(const GridSize&, const PhysicalOrigin&, const GridSpacing&, const Bool3&)
 */
[[nodiscard]] world::World create_world(const Int3 &size);

/**
 * @brief Create a World object with strong types for type safety (PRIMARY API).
 *
 * @details This is the **recommended** interface for constructing World objects.
 * It uses pfc::data::Field (grid_field.hpp) internally and follows the
 * OPENPFC_REFACTORING_EXECUTION_PLAN M2 guidance. Strong types
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
[[nodiscard]] world::World create_world(const ::pfc::GridSize &size,
                           const ::pfc::PhysicalOrigin &origin,
                           const ::pfc::GridSpacing &spacing,
                           const Bool3 &periodic = {true, true, true});

// Convenience helpers (equivalent world_helpers.hpp functions moved to pfc::domain::)

/**
 * @brief Create uniform grid with unit spacing at origin.
 *
 * Most common case: N×N×N grid with spacing=1, origin=(0,0,0).
 *
 * @param size Grid dimensions (same in all directions)
 * @return World with uniform grid
 *
 * @throws std::invalid_argument if size <= 0
 *
 * @code
 * auto world = domain::create_world_uniform(64);  // 64³ grid, dx=1
 * @endcode
 */
[[nodiscard]] world::World create_world_uniform(int size);

/**
 * @brief Create uniform grid with specified spacing.
 *
 * @param size Grid dimensions (same in all directions)
 * @param spacing Grid spacing (same in all directions)
 * @return World with uniform grid and spacing
 *
 * @throws std::invalid_argument if size <= 0
 * @throws std::invalid_argument if spacing <= 0
 *
 * @code
 * auto world = domain::create_world_uniform(128, 0.5);  // 128³ grid, dx=0.5
 * @endcode
 */
[[nodiscard]] world::World create_world_uniform(int size, double spacing);

/**
 * @brief Create grid from physical bounds (automatically computes spacing).
 *
 * This is the World-returning version. Use pfc::domain::from_bounds() for Domain return.
 *
 * @param size Grid dimensions
 * @param lower Lower physical bounds
 * @param upper Upper physical bounds
 * @param periodic Periodicity flags (default: all periodic)
 * @return World with computed spacing
 *
 * @throws std::invalid_argument if any dimension size <= 0
 * @throws std::invalid_argument if any upper bound <= corresponding lower bound
 *
 * @note Spacing computed as: dx = (upper - lower) / size for periodic,
 *                               dx = (upper - lower) / (size - 1) for non-periodic
 *
 * @code
 * // 100 cells from 0 to 10 (periodic)
 * auto w1 = domain::create_world_from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10});
 *
 * // Non-periodic in x (different spacing formula)
 * auto w2 = domain::create_world_from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10},
 *                                           {false, true, true});
 * @endcode
 */
[[nodiscard]] world::World create_world_from_bounds(Int3 size, Real3 lower, Real3 upper,
                                                     Bool3 periodic = {true, true, true});

/**
 * @brief Create grid with default origin but custom spacing.
 *
 * @param size Grid dimensions
 * @param spacing Grid spacing
 * @return World with specified size and spacing, origin at (0,0,0)
 *
 * @throws std::invalid_argument if any size <= 0
 * @throws std::invalid_argument if any spacing <= 0
 *
 * @code
 * auto world = domain::create_world_with_spacing({64, 64, 128}, {0.1, 0.1, 0.05});
 * @endcode
 */
[[nodiscard]] world::World create_world_with_spacing(Int3 size, Real3 spacing);

/**
 * @brief Create grid with custom origin but unit spacing.
 *
 * @param size Grid dimensions
 * @param origin Physical origin
 * @return World with specified size and origin, spacing=1
 *
 * @throws std::invalid_argument if any size <= 0
 *
 * @code
 * auto world = domain::create_world_with_origin({64, 64, 64}, {-5.0, -5.0, 0.0});
 * @endcode
 */
[[nodiscard]] world::World create_world_with_origin(Int3 size, Real3 origin);

} // namespace domain

} // namespace pfc
