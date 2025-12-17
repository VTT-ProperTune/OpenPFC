// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world.hpp
 * @brief World class definition and unified interface
 *
 * @details
 * The `World<CoordTag>` class defines the **global simulation domain** in
 * OpenPFC's computational physics framework. It provides a unified abstraction
 * for describing a discretized physical space in which fields are defined,
 * evolved, and coupled to solvers.
 *
 * ## Architecture
 *
 * World functionality is split across focused modules:
 * - **world.hpp** (this file) - Core World struct definition
 * - **world_factory.hpp** - Factory functions for creating World objects
 * - **world_queries.hpp** - Query functions and coordinate transformations
 * - **world_helpers.hpp** - Convenience constructors (uniform(), from_bounds(),
 * etc.)
 *
 * ## Quick Start
 *
 * @code
 * using namespace pfc;
 *
 * // Create Cartesian world with default settings
 * World world = world::create({100, 100, 100});
 *
 * // Query and transform
 * Real3 x = world::to_coords(world, {10, 20, 30});
 * Int3 i  = world::to_indices(world, {10.0, 20.0, 30.0});
 * double dx = world::get_spacing(world, 0);
 * @endcode
 *
 * ## Design Philosophy
 *
 * World follows OpenPFC's "Laboratory, Not Fortress" principles:
 * - **Immutable value-type**: No mutable state, thread-safe by design
 * - **Functional API**: Free functions for operations (not member methods)
 * - **Template-based extensibility**: Support custom coordinate systems via tags
 * - **Zero-overhead abstractions**: Inline functions, no runtime polymorphism
 * - **Explicit over implicit**: Clear, self-documenting APIs
 *
 * ## Extending with Custom Coordinate Systems
 *
 * Add custom coordinate systems (cylindrical, spherical, etc.) without modifying
 * OpenPFC source. See `examples/17_custom_coordinate_system.cpp` for complete
 * working examples and `docs/extending_openpfc/adl_extension_patterns.md` for
 * comprehensive guide.
 *
 * @see world_factory.hpp for World creation functions
 * @see world_queries.hpp for queries and coordinate transforms
 * @see world_helpers.hpp for convenience constructors
 * @see examples/17_custom_coordinate_system.cpp for extension example
 */

#pragma once

#include <array>
#include <ostream>

#include "csys.hpp"
#include "types.hpp"

namespace pfc {
namespace world {

using pfc::csys::CartesianTag;
using pfc::csys::CoordinateSystem;
using pfc::types::Int3;

/**
 * @brief Represents the global simulation domain (the "world").
 *
 * The World class defines the size of the global simulation domain and
 * coordinate system. It is a purely functional object with no mutable state,
 * constructed once and immutable thereafter. This design enhances correctness,
 * thread safety, testability, and reproducibility.
 *
 * Coordinate system is defined via a tag-based programming approach, allowing
 * different coordinate systems (Cartesian, Polar, Cylindrical) without creating
 * a separate class for each. We default to 3D Cartesian as it's most common.
 *
 * @tparam T Coordinate system tag (e.g., CartesianTag)
 *
 * @see world_factory.hpp for construction
 * @see world_queries.hpp for accessing properties
 */
template <typename T> struct World final {
  const Int3 m_lower;             ///< Lower index bounds
  const Int3 m_upper;             ///< Upper index bounds
  const Int3 m_size;              ///< Grid dimensions: {nx, ny, nz}
  const CoordinateSystem<T> m_cs; ///< Coordinate system

  /**
   * @brief Constructs a World object.
   * @param lower Lower index bounds of the world.
   * @param upper Upper index bounds of the world.
   * @param cs Coordinate system.
   */
  explicit World(const Int3 &lower, const Int3 &upper,
                 const CoordinateSystem<T> &cs);

  /**
   * @brief Equality operator.
   * @param other Another World object.
   * @return True if equal, false otherwise.
   */
  bool operator==(const World &other) const noexcept {
    return m_lower == other.m_lower && m_upper == other.m_upper &&
           m_size == other.m_size && m_cs == other.m_cs;
  }

  /**
   * @brief Inequality operator.
   * @param other Another World object.
   * @return True if not equal, false otherwise.
   */
  bool operator!=(const World &other) const noexcept { return !(*this == other); }

  /**
   * @brief Stream output operator.
   * @param os Output stream.
   * @param w World object.
   * @return Reference to the output stream.
   */
  template <typename T_>
  friend std::ostream &operator<<(std::ostream &os, const World<T_> &w) noexcept;
};

/// Type alias for Cartesian 3D World (most common usage)
using CartesianWorld = World<CartesianTag>;

} // namespace world

// Export World to pfc namespace for convenient usage
using World = world::CartesianWorld;

} // namespace pfc

// Include World functionality modules
#include "world_factory.hpp"
#include "world_helpers.hpp"
#include "world_queries.hpp"
