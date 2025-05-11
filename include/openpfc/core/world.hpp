// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * @file world.hpp
 * @brief World class definition and interface
 *
 * @details
 * The `World<CoordTag>` class defines the **global simulation domain** \(
 * \Omega \) in OpenPFC's computational physics framework. It provides a unified
 * abstraction for describing a discretized physical space in which fields are
 * defined, evolved, and coupled to solvers.
 *
 * The World object encapsulates:
 *
 * - the grid resolution (number of cells per dimension),
 * - a coordinate system specialization (e.g., Cartesian, Polar),
 * - periodicity information (optional),
 * - and generic support for coordinate transformations.
 *
 * Coordinate transformations are handled via `CoordinateSystem<CoordTag>`,
 * which maps discrete index space to physical space and vice versa. This
 * structure is **open and extensible**: users may define their own coordinate
 * systems and inject them into the simulation without modifying OpenPFC
 * internals.
 *
 * ---
 *
 * ## Roles and Responsibilities of World
 *
 * 1. **Defines the discrete computational domain**: A World instance defines a
 *    *regular structured grid* of size \( L_x \times L_y \times L_z \),
 *    anchored to a user-defined coordinate system.
 *
 * 2. **Separates geometry from data**: Fields and solvers operate on raw data,
 *    while World encapsulates the geometry and layout of the simulation space.
 *    This separation improves clarity, composability, and reuse.
 *
 * 3. **Performs index ↔ physical coordinate transforms**: Coordinate systems
 *    define mappings like:
 *    @code
 *      Real3 x = to_coords(world, {i, j, k});
 *      Int3  ijk = to_indices(world, x);
 *    @endcode
 *    These are **zero-overhead**, inline computations using the coordinate
 * system definition.
 *
 * 4. **Supports extensible coordinate systems**: Instead of hardcoding known
 *    systems, OpenPFC uses a template-based model:
 *      - `CoordinateSystem<CartesianTag>`
 *      - `CoordinateSystem<PolarTag>`
 *      - or any user-defined `MyCustomTag`
 *
 *    Users may specialize traits like `CoordinateSystemDefaults<MyCustomTag>`
 *    to define defaults (e.g., spacing, offset, periodicity) and overload
 *    `to_coords()` as needed.
 *
 * 5. **Supports periodic and non-periodic boundaries**: The periodicity of each
 *    dimension is stored in the coordinate system and respected by
 *    index-to-physical transforms. Grid spacing logic follows:
 *    @code
 *      spacing = (upper - lower) / (periodic ? size : size - 1)
 *    @endcode
 *
 * 6. **Allows multiple construction styles**: World creation is done through
 *    overloaded factory functions using:
 *      - Size + spacing + offset
 *      - Size + lower + upper
 *      - User-defined coordinate system instance
 *      - Defaults via traits for built-in tags
 *
 * 7. **Offers a functional API**: World is an immutable value-type. Operations
 *    like `to_coords()` and `get_spacing()` are implemented as free functions
 *    in the `pfc::world` namespace. This:
 *      - avoids mutation and inheritance
 *      - supports ADL-based extension
 *      - encourages clean, composable simulation code
 *
 * 8. **Stays minimal and explicit**: `World<CoordTag>` is a lightweight value
 *    class with minimal members:
 *      - `m_size`: grid dimensions
 *      - `m_cs`: coordinate system instance
 *
 *    There are no virtual methods, hidden ownership, or runtime polymorphism.
 *
 * ---
 *
 * ## Philosophical Note
 *
 * OpenPFC is built as a **laboratory**, not a fortress. The `World` class plays
 * a central role in this lab — it defines the geometric stage on which physics
 * unfolds. Its design emphasizes:
 *
 * - *Purity*: `World` is immutable and functional
 * - *Precision*: Spacing, bounds, and coordinates are rigorously defined
 * - *Openness*: Users can define new coordinate systems and behaviors
 * - *Clarity*: No hidden magic, just explicit composition
 *
 * This structure ensures that simulation domains are *safe*, *predictable*, and
 * *easy to reason about*.
 *
 * ---
 *
 * ## Usage Example
 *
 * @code
 * using namespace pfc;
 *
 * // Default Cartesian world with unit spacing and offset at (0,0,0)
 * World<CartesianTag> w = world::create({100, 100, 100});
 *
 * Real3 x = to_coords(w, {10, 20, 30});
 * Int3 i  = to_indices(w, {10.0, 20.0, 30.0});
 * double dx = get_spacing(w, 0);
 * @endcode
 */

#pragma once

#include <array>
#include <ostream>
#include <stdexcept>

#include "csys.hpp"
#include "types.hpp"

namespace pfc {

namespace world {

using pfc::csys::CartesianTag;
using pfc::csys::CoordinateSystem;
using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

/**
 * @brief Represents the global simulation domain (the "world").
 *
 * The World class defines the *size*of the global simulation domain and
 * coordinate system. It is a *purely functional* object, meaning it has no
 * mutable state and is immutable once constructed. This design follows the
 * principles of functional programming, where data structures are fixed and
 * behavior is implemented externally via free functions. This enhances
 * correctness, thread safety, testability, and reproducibility.
 *
 * Coordinate system is defined via a tag-based programming approach. This
 * allows us to define different coordinate systems (e.g., Cartesian, Polar,
 * Cylindrical) without creating a separate class for each. We default to 3D
 * Cartesian coordinate system as it's the most common in scientific computing.
 */
template <typename T> struct World final {
  const Int3 m_lower;             ///< Lower bounds of the world
  const Int3 m_upper;             ///< Upper bounds of the world
  const Int3 m_size;              ///< Dimensions of the world: {L1, L2, L3}
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

// Free function API for creating (Cartesian 3D) World objects

using CartesianWorld = World<CartesianTag>;

/**
 * @brief Create a World object with the specified size and default offset
 * and spacing.
 * @param dimensions Dimensions of the world.
 * @return A World object.
 */
CartesianWorld create(const Int3 &size);

/**
 * @brief Create a World object with the specified dimensions, offset, and
 * spacing.
 * @param dimensions Dimensions of the world.
 * @param offset Offset of the world.
 * @param spacing Spacing of the grid.
 * @return A World object.
 */
CartesianWorld create(const Int3 &size, const Real3 &offset, const Real3 &spacing);

/**
 * @brief Create a World object with the specified size, lower bounds, upper
 * bounds, spacing, and periodicity.
 * @param size Size of the world.
 * @param lower Lower bounds of the world.
 * @param upper Upper bounds of the world.
 * @param spacing Spacing of the grid.
 * @param periodic Periodicity flags.
 * @param cs Coordinate system type.
 * @return A World object.
 */
/*
 CartesianWorld create(const Size3 &size, const LowerBounds3 &lower,
                      const UpperBounds3 &upper, const Spacing3 &spacing,
                      const Periodic3 &periodic);
*/

/**
 * @brief Create a World object with the specified size, lower bounds, upper
 * bounds, periodicity, and coordinate system.
 * @param size Size of the world.
 * @param lower Lower bounds of the world.
 * @param upper Upper bounds of the world.
 * @param periodic Periodicity flags.
 * @param cs Coordinate system type.
 * @return A World object.
 */
/*
CartesianWorld create(const Size3 &size, const LowerBounds3 &lower,
                      const UpperBounds3 &upper, const Periodic3 &periodic);
*/

/**
 * @brief Create a World object with the specified size, lower bounds, spacing,
 * periodicity, and coordinate system.
 * @param size Size of the world.
 * @param lower Lower bounds of the world.
 * @param spacing Spacing of the grid.
 * @param periodic Periodicity flags.
 * @param cs Coordinate system type.
 * @return A World object.
 */
/*
CartesianWorld create(const Size3 &size, const LowerBounds3 &lower,
                      const Spacing3 &spacing, const Periodic3 &periodic);
*/

/**
 * @brief Create a World object with the specified size and upper bounds.
 * @param size Size of the world.
 * @param upper Upper bounds of the world.
 * @return A World object.
 */
/*
CartesianWorld create(const Size3 &size, const UpperBounds3 &upper);
*/

// Free function API for querying World properties

/**
 * @brief Get the size of the world.
 * @param w World object.
 * @return The size of the world.
 */
template <typename T> Int3 get_size(const World<T> &world) noexcept {
  return world.m_size;
}

/**
 * @brief Get the size of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The size in the specified dimension.
 */
template <typename T> int get_size(const World<T> &world, int index) {
  return get_size(world).at(index);
}

/**
 * @brief Get the total number of grid points in the world.
 * @param w World object.
 * @return The total number of grid points.
 */
template <typename T> size_t get_total_size(const World<T> &world) noexcept {
  return get_size(world, 0) * get_size(world, 1) * get_size(world, 2);
}

/**
 * @brief Get the lower bounds of the world
 * @param w World object.
 * @return The lower bounds of the world.
 */
inline const auto &get_lower(const CartesianWorld &world) noexcept {
  return world.m_lower;
}

/**
 * @brief Get the lower bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The lower bound in the specified dimension.
 */
inline const auto &get_lower(const CartesianWorld &world, int index) {
  return get_lower(world).at(index);
}

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @return The upper bounds of the world.
 */
inline const auto &get_upper(const CartesianWorld &world) noexcept {
  return world.m_upper;
}

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The upper bound in the specified dimension.
 */
inline const auto get_upper(const CartesianWorld &world, int index) {
  return get_upper(world).at(index);
}

/**
 * @brief Compute the physical coordinates corresponding to grid indices.
 * @param w World object.
 * @param indices Grid indices.
 * @return The physical coordinates.
 */
template <typename T>
inline const auto to_coords(const World<T> &world, const Int3 &indices) noexcept {
  return to_coords(get_coordinate_system(world), indices);
}

/**
 * @brief Compute the grid indices corresponding to physical coordinates.
 * @param w World object.
 * @param coordinates Physical coordinates.
 * @return The grid indices.
 */
template <typename T>
inline const auto to_indices(const World<T> &world, const Real3 &coords) noexcept {
  return to_index(get_coordinate_system(world), coords);
}

// Free function API for coordinate system and periodicity

/**
 * @brief Get the coordinate system of the world.
 * @param w World object.
 * @return The coordinate system of the world.
 */
template <typename T>
inline const auto &get_coordinate_system(const World<T> &world) noexcept {
  return world.m_cs;
}

// For backward compatibility, might be removed in the future

inline const Real3 &get_spacing(const CartesianWorld &world) noexcept {
  return get_spacing(get_coordinate_system(world));
}

inline double get_spacing(const CartesianWorld &world, int index) noexcept {
  return get_spacing(get_coordinate_system(world), index);
}

inline const Real3 &get_origin(const CartesianWorld &world) noexcept {
  return get_offset(get_coordinate_system(world));
}

inline double get_origin(const CartesianWorld &world, int index) noexcept {
  return get_offset(get_coordinate_system(world), index);
}

} // namespace world

// export World class to the pfc namespace, so we hopefully don't have to write
// `world::World world = world::create_world(...)` kind of things :D
using World = world::CartesianWorld;

} // namespace pfc
