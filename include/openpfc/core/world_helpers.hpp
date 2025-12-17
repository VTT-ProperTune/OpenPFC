// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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

#include "strong_types.hpp"
#include "types.hpp"
#include "world.hpp"
#include "world_factory.hpp"
#include <stdexcept>
#include <string>

namespace pfc {
namespace world {

using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

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
 * auto world = world::uniform(64);  // 64³ grid, dx=1
 * @endcode
 */
inline CartesianWorld uniform(int size) {
  if (size <= 0) {
    throw std::invalid_argument("Grid size must be positive, got: " +
                                std::to_string(size));
  }
  return create(GridSize({size, size, size}), PhysicalOrigin({0.0, 0.0, 0.0}),
                GridSpacing({1.0, 1.0, 1.0}));
}

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
 * auto world = world::uniform(128, 0.5);  // 128³ grid, dx=0.5
 * @endcode
 */
inline CartesianWorld uniform(int size, double spacing) {
  if (size <= 0) {
    throw std::invalid_argument("Grid size must be positive, got: " +
                                std::to_string(size));
  }
  if (spacing <= 0.0) {
    throw std::invalid_argument("Spacing must be positive, got: " +
                                std::to_string(spacing));
  }
  return create(GridSize({size, size, size}), PhysicalOrigin({0.0, 0.0, 0.0}),
                GridSpacing({spacing, spacing, spacing}));
}

/**
 * @brief Create grid from physical bounds (automatically computes spacing).
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
 * auto w1 = world::from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10});
 *
 * // Non-periodic in x (different spacing formula)
 * auto w2 = world::from_bounds({100, 100, 100}, {0, 0, 0}, {10, 10, 10},
 *                               {false, true, true});
 * @endcode
 */
inline CartesianWorld from_bounds(Int3 size, Real3 lower, Real3 upper,
                                  Bool3 periodic = {true, true, true}) {
  // Validate inputs
  for (int i = 0; i < 3; ++i) {
    if (size[i] <= 0) {
      throw std::invalid_argument("Grid size must be positive in all dimensions");
    }
    if (upper[i] <= lower[i]) {
      throw std::invalid_argument("Upper bound must be greater than lower bound");
    }
  }

  // Compute spacing based on periodicity
  Real3 spacing;
  for (int i = 0; i < 3; ++i) {
    if (periodic[i]) {
      spacing[i] = (upper[i] - lower[i]) / size[i];
    } else {
      spacing[i] = (upper[i] - lower[i]) / (size[i] - 1);
    }
  }

  return create(GridSize(size), PhysicalOrigin(lower), GridSpacing(spacing));
}

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
 * auto world = world::with_spacing({64, 64, 128}, {0.1, 0.1, 0.05});
 * @endcode
 */
inline CartesianWorld with_spacing(Int3 size, Real3 spacing) {
  // Validate
  for (int i = 0; i < 3; ++i) {
    if (size[i] <= 0) {
      throw std::invalid_argument("Grid size must be positive");
    }
    if (spacing[i] <= 0.0) {
      throw std::invalid_argument("Spacing must be positive");
    }
  }

  return create(GridSize(size), PhysicalOrigin({0.0, 0.0, 0.0}),
                GridSpacing(spacing));
}

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
 * auto world = world::with_origin({64, 64, 64}, {-5.0, -5.0, 0.0});
 * @endcode
 */
inline CartesianWorld with_origin(Int3 size, Real3 origin) {
  // Validate
  for (int i = 0; i < 3; ++i) {
    if (size[i] <= 0) {
      throw std::invalid_argument("Grid size must be positive");
    }
  }

  return create(GridSize(size), PhysicalOrigin(origin),
                GridSpacing({1.0, 1.0, 1.0}));
}

} // namespace world
} // namespace pfc
