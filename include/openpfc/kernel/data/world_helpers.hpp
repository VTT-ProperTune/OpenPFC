// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world_helpers.hpp
 * @brief Convenience functions for World creation
 *
 * @details
 * This file contains convenience functions that simplify common World creation
 * patterns. These functions provide shortcuts for frequently-used configurations:
 *
 * - uniform(size) - Cubic grid with unit spacing
 * - uniform(size, spacing) - Cubic grid with custom spacing
 * - from_bounds() - Create from physical domain bounds
 * - with_spacing() - Grid with custom spacing at origin
 * - with_origin() - Grid with custom origin and unit spacing
 *
 * These helpers make World creation more ergonomic for common use cases while
 * delegating to the core factory functions in world_factory.hpp.
 *
 * @see world.hpp for the core World struct definition
 * @see world_factory.hpp for the fundamental create() functions
 */

#pragma once

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/domain/create.hpp>

namespace pfc::world {

// Convenience helper types for backward compatibility
using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

/**
 * @brief Periodicity flags for each spatial dimension.
 *
 * A 3-element array where each element indicates whether that dimension
 * is periodic (true) or non-periodic (false).
 */
using Periodicity = Bool3;

/**
 * @brief Create uniform grid with unit spacing at origin.
 *
 * Most common case: N×N×N grid with spacing=1, origin=(0,0,0).
 *
 * @deprecated Use pfc::domain::create_world_uniform(int, Bool3) instead
 *
 * @param size Grid dimensions (same in all directions)
 * @param periodic Per-axis periodicity flags (default: all non-periodic)
 * @return World with uniform grid
 *
 * @throws std::invalid_argument if size <= 0
 *
 * @code
 * auto world = pfc::domain::create_world_uniform(64);  // 64³ grid, dx=1, non-periodic
 * auto world_periodic = pfc::domain::create_world_uniform(64, {true, true, true});  // periodic
 * @endcode
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world_uniform(int, Bool3) instead")]] inline CartesianWorld uniform(int size, Periodicity periodic = {false, false, false}) {
  return pfc::domain::create_world_uniform(size, periodic);
}

/**
 * @brief Create uniform grid with specified spacing.
 *
 * @deprecated Use pfc::domain::create_world_uniform(int, double, Bool3) instead
 *
 * @param size Grid dimensions (same in all directions)
 * @param spacing Grid spacing (same in all directions)
 * @param periodic Per-axis periodicity flags (default: all non-periodic)
 * @return World with uniform grid and spacing
 *
 * @throws std::invalid_argument if size <= 0
 * @throws std::invalid_argument if spacing <= 0
 *
 * @code
 * auto world = pfc::domain::create_world_uniform(128, 0.5);  // 128³ grid, dx=0.5, non-periodic
 * auto world_periodic = pfc::domain::create_world_uniform(128, 0.5, {true, true, true});  // periodic
 * @endcode
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world_uniform(int, double, Bool3) instead")]] inline CartesianWorld uniform(int size, double spacing, Periodicity periodic = {false, false, false}) {
  return pfc::domain::create_world_uniform(size, spacing, periodic);
}

/**
 * @brief Create grid from physical bounds (automatically computes spacing).
 *
 * @deprecated Use pfc::domain::create_world_from_bounds(const Int3&, const Real3&, const Real3&, const Bool3&) instead
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
 * auto w1 = pfc::domain::create_world_from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10});
 *
 * // Non-periodic in x (different spacing formula)
 * auto w2 = pfc::domain::create_world_from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10},
 *                                               {false, true, true});
 * @endcode
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world_from_bounds(const Int3&, const Real3&, const Real3&, const Bool3&) instead")]] inline CartesianWorld from_bounds(Int3 size, Real3 lower, Real3 upper,
                                                Bool3 periodic = {true, true,
                                                                  true}) {
  return pfc::domain::create_world_from_bounds(size, lower, upper, periodic);
}

/**
 * @brief Create grid with default origin but custom spacing.
 *
 * @deprecated Use pfc::domain::create_world_with_spacing(const Int3&, const Real3&, const Bool3&) instead
 *
 * @param size Grid dimensions
 * @param spacing Grid spacing
 * @param periodic Per-axis periodicity flags (default: all non-periodic)
 * @return World with specified size and spacing, origin at (0,0,0)
 *
 * @throws std::invalid_argument if any size <= 0
 * @throws std::invalid_argument if any spacing <= 0
 *
 * @code
 * auto world = pfc::domain::create_world_with_spacing({64, 64, 128}, {0.1, 0.1, 0.05});  // non-periodic
 * auto world_periodic = pfc::domain::create_world_with_spacing({64, 64, 128}, {0.1, 0.1, 0.05}, {true, true, true});  // periodic
 * @endcode
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world_with_spacing(const Int3&, const Real3&, const Bool3&) instead")]] inline CartesianWorld with_spacing(Int3 size, Real3 spacing, Periodicity periodic = {false, false, false}) {
  return pfc::domain::create_world_with_spacing(size, spacing, periodic);
}

/**
 * @brief Create grid with custom origin but unit spacing.
 *
 * @deprecated Use pfc::domain::create_world_with_origin(const Int3&, const Real3&, const Bool3&) instead
 *
 * @param size Grid dimensions
 * @param origin Physical origin
 * @param periodic Per-axis periodicity flags (default: all non-periodic)
 * @return World with specified size and origin, spacing=1
 *
 * @throws std::invalid_argument if any size <= 0
 *
 * @code
 * auto world = pfc::domain::create_world_with_origin({64, 64, 64}, {-5.0, -5.0, 0.0});  // non-periodic
 * auto world_periodic = pfc::domain::create_world_with_origin({64, 64, 64}, {-5.0, -5.0, 0.0}, {true, true, true});  // periodic
 * @endcode
 */
[[nodiscard]] [[deprecated("Use pfc::domain::create_world_with_origin(const Int3&, const Real3&, const Bool3&) instead")]] inline CartesianWorld with_origin(Int3 size, Real3 origin, Periodicity periodic = {false, false, false}) {
  return pfc::domain::create_world_with_origin(size, origin, periodic);
}

} // namespace pfc::world
