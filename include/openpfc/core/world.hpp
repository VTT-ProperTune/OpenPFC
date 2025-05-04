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

namespace pfc {

/// Type aliases for clarity
using Int3 = std::array<int, 3>;
using Real3 = std::array<double, 3>;
using Bool3 = std::array<bool, 3>;

namespace world {

/*
Strong typedefs for constructor clarity. The idea of strong typedefs is to wrap
some base class to provide a more meaningful name and to prevent implicit
conversions. This is a common C++ idiom to improve code readability and
maintainability. Strong typedefs are often used in scientific computing
libraries to represent physical quantities with units, or in simulation
frameworks to represent simulation parameters with specific meanings. They can
also include e.g. bounds checking, etc.
*/

/**
 * @brief Represents the size of the simulation domain.
 */
struct Size3 {
  std::array<int, 3> value;
  explicit Size3(const std::array<int, 3> &v);
};

/**
 * @brief Represents the lower bounds of the simulation domain.
 */
struct LowerBounds3 {
  std::array<double, 3> value;
  explicit LowerBounds3(const std::array<double, 3> &v);
};

/**
 * @brief Represents the upper bounds of the simulation domain.
 */
struct UpperBounds3 {
  std::array<double, 3> value;
  explicit UpperBounds3(const std::array<double, 3> &v);
};

/**
 * @brief Represents the spacing of the simulation grid.
 */
struct Spacing3 {
  std::array<double, 3> value;
  explicit Spacing3(const std::array<double, 3> &v);
};

/**
 * @brief Represents the periodicity of the simulation domain.
 */
struct Periodic3 {
  std::array<bool, 3> value;
  explicit Periodic3(const std::array<bool, 3> &v);
};

/**
 * @brief Primary template for defining coordinate systems by tag.
 *
 * @details
 * The `CoordinateSystem<Tag>` template provides a mechanism to define coordinate
 * systems in a modular and extensible way, where each `Tag` corresponds to a
 * specific geometry (e.g., Cartesian, Polar, Cylindrical).
 *
 * Specializations of this template should define:
 * - The internal parameters of the coordinate system (e.g., offset, spacing)
 * - Methods to map between index space and physical space:
 *   - `to_physical(const Int3&) -> Real3`
 *   - `to_index(const Real3&) -> Int3`
 *
 * This design decouples geometry from logic and enables user-defined coordinate
 * systems to integrate cleanly with the simulation framework. It also avoids
 * inheritance or runtime polymorphism, favoring compile-time specialization and
 * inlining for performance-critical transformations.
 *
 * Example usage:
 * @code
 * struct CartesianTag {};
 *
 * template <>
 * struct CoordinateSystem<CartesianTag> {
 *   Real3 offset;
 *   Real3 spacing;
 *
 *   Real3 to_physical(const Int3& idx) const noexcept;
 *   Int3 to_index(const Real3& pos) const noexcept;
 * };
 * @endcode
 *
 * Coordinate systems are used by the `World<Tag>` class and related infrastructure
 * to define how grid indices map to real-world physical coordinates.
 *
 * @tparam Tag A user-defined type that uniquely identifies the coordinate system.
 */
template <typename Tag> struct CoordinateSystem;

/**
 * @brief Trait class for providing default parameters for coordinate systems.
 *
 * @details
 * This traits template defines default values (e.g., offset, spacing, periodicity)
 * for coordinate systems identified by their tag type `CoordTag`.
 *
 * The primary template is left undefined, and users are expected to specialize
 * this trait for their own coordinate system tags to provide meaningful defaults.
 *
 * This mechanism allows external extension of coordinate system behavior
 * without modifying the core library, enabling user-defined systems to
 * seamlessly integrate with generic `World` construction and other infrastructure.
 *
 * Example specialization:
 * @code
 * struct MyCoordTag {};
 *
 * template <>
 * struct CoordinateSystemDefaults<MyCoordTag> {
 *   static constexpr Real3 offset = {0.0, 0.0, 0.0};
 *   static constexpr Real3 spacing = {1.0, 1.0, 1.0};
 *   static constexpr Bool3 periodicity = {true, false, false};
 * };
 * @endcode
 */
template <typename CoordTag>
struct CoordinateSystemDefaults; // intentionally left undefined

// Coordinate system tags.
// struct LineTag {};
// struct PlaneTag {};
// struct PolarTag {};
// struct CylindricalTag {};
// struct SphericalTag {};
// struct Polar2DTag {};
// struct Toroidal2DTag {};
// struct LogPolar3DTag {};

/**
 * @brief Tag type for the 3D Cartesian coordinate system.
 *
 * @details
 * This tag represents a standard right-handed Cartesian coordinate system in 3D,
 * which is the default and most commonly used geometry in OpenPFC simulations.
 *
 * The associated coordinate system maps discrete grid indices (i, j, k) to
 * continuous physical space using a uniform, axis-aligned grid defined by:
 * - `offset`: the physical position corresponding to index (0, 0, 0)
 * - `spacing`: the distance between adjacent grid points along each axis
 *
 * This coordinate system assumes:
 * - The grid is regular (uniform spacing)
 * - Axes are orthogonal and aligned with the simulation dimensions
 *
 * It provides a simple and efficient mapping suitable for a wide range of
 * physical simulations where a Euclidean space is appropriate.
 */
struct CartesianTag {};

/**
 * @brief Default parameters for the 3D Cartesian coordinate system.
 *
 * @details
 * This specialization of `CoordinateSystemDefaults` provides the default values
 * used when constructing a 3D Cartesian coordinate system (`CartesianTag`)
 * without explicitly specifying offset, spacing, or periodicity.
 *
 * These defaults are consistent with standard simulation conventions:
 * - `offset = {0.0, 0.0, 0.0}` places the (0, 0, 0) index at the physical origin.
 * - `spacing = {1.0, 1.0, 1.0}` defines unit grid spacing in all dimensions.
 * - `periodicity = {true, true, true}` models a fully periodic domain.
 *
 * These values are used by factory functions and world constructors when
 * coordinate system parameters are not explicitly provided by the user.
 */
template <> struct CoordinateSystemDefaults<CartesianTag> {
  static constexpr Real3 offset = {0.0, 0.0, 0.0};
  static constexpr Real3 spacing = {1.0, 1.0, 1.0};
  static constexpr Bool3 periodic = {true, true, true};
  std::size_t dimensions = 3; ///< Number of dimensions in the coordinate system
};

/**
 * @brief Specialization of the coordinate system for 3D Cartesian space.
 *
 * @details
 * This structure defines a uniform, axis-aligned Cartesian coordinate system
 * in three dimensions. It maps discrete grid indices (i, j, k) to continuous
 * physical coordinates using a regular grid geometry.
 *
 * The coordinate system is defined by three key parameters:
 * - `m_offset`: the physical position corresponding to the grid index (0, 0, 0)
 * - `m_spacing`: the uniform distance between adjacent grid points in each dimension
 * - `m_periodic`: flags indicating periodicity along each axis
 *
 * This specialization is used internally by the `World<CartesianTag>` type
 * and can be constructed directly or via factory methods. It is designed to
 * support fast, inlined coordinate transformations for performance-critical
 * simulation kernels.
 *
 * @constructor
 * Constructs a Cartesian coordinate system with the specified offset, spacing,
 * and periodicity. Throws `std::invalid_argument` if any spacing component
 * is non-positive.
 *
 * @param offset     Physical position of the index (0, 0, 0)
 * @param spacing    Grid spacing in each dimension (must be > 0)
 * @param periodic   Periodicity flags for {x, y, z}
 */
template <> struct CoordinateSystem<CartesianTag> {
  const Real3 m_offset;   ///< Physical coordinate of grid index (0, 0, 0)
  const Real3 m_spacing;  ///< Physical spacing between grid points
  const Bool3 m_periodic; ///< Periodicity flags for each dimension

  /// Constructs a 3D Cartesian coordinate system
  CoordinateSystem(
      const Real3 &offset = CoordinateSystemDefaults<CartesianTag>::offset,
      const Real3 &spacing = CoordinateSystemDefaults<CartesianTag>::spacing,
      const Bool3 &periodic = CoordinateSystemDefaults<CartesianTag>::periodic);
};

using CartesianCS = CoordinateSystem<CartesianTag>;

/**
 * @brief Get the offset of the coordinate system.
 * @param cs Coordinate system object.
 * @return The offset of the coordinate system.
 */
const Real3 &get_offset(const CartesianCS &cs) noexcept;

/**
 * @brief Get the offset of the coordinate system in a specific dimension.
 * @param cs Coordinate system object.
 * @param i Dimension index.
 * @return The offset in the specified dimension.
 * @throws std::out_of_range if i is not in [0, 2].
 */
double get_offset(const CartesianCS &cs, int i);

/**
 * @brief Get the spacing of the coordinate system.
 * @param cs Coordinate system object.
 * @return The spacing of the coordinate system.
 */
const Real3 &get_spacing(const CartesianCS &cs) noexcept;

/**
 * @brief Get the spacing of the coordinate system in a specific dimension.
 * @param cs Coordinate system object.
 * @param i Dimension index.
 * @return The spacing in the specified dimension.
 * @throws std::out_of_range if i is not in [0, 2].
 */
double get_spacing(const CartesianCS &cs, int i);

/**
 * @brief Get the periodicity of the coordinate system.
 * @param cs Coordinate system object.
 * @return The periodicity flags.
 */
const Bool3 &get_periodicity(const CartesianCS &cs) noexcept;

/**
 * @brief Check if the coordinate system is periodic in a specific dimension.
 * @param cs Coordinate system object.
 * @param i Dimension index.
 * @return True if periodic, false otherwise.
 * @throws std::out_of_range if i is not in [0, 2].
 */
bool is_periodic(const CartesianCS &cs, int i);

/**
 * @brief Convert grid indices to physical coordinates.
 * @param cs Coordinate system object.
 * @param idx Grid indices.
 * @return The physical coordinates.
 */
const Real3 to_coords(const CartesianCS &cs, const Int3 &idx) noexcept;

/**
 * @brief Convert physical coordinates to grid indices.
 * @param cs Coordinate system object.
 * @param xyz Physical coordinates.
 * @return The grid indices.
 */
const Int3 to_index(const CartesianCS &cs, const Real3 &xyz) noexcept;

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
template <typename CoordTag> struct World final {
  const Int3 m_lower;                    ///< Lower bounds of the world
  const Int3 m_upper;                    ///< Upper bounds of the world
  const Int3 m_size;                     ///< Dimensions of the world: {L1, L2, L3}
  const CoordinateSystem<CoordTag> m_cs; ///< Coordinate system

  /**
   * @brief Constructs a World object.
   * @param lower Lower index bounds of the world.
   * @param upper Upper index bounds of the world.
   * @param cs Coordinate system.
   */
  explicit World(const Int3 &lower, const Int3 &upper,
                 const CoordinateSystem<CoordTag> &cs);

  /**
   * @brief Equality operator.
   * @param other Another World object.
   * @return True if equal, false otherwise.
   */
  bool operator==(const World &other) const noexcept;

  /**
   * @brief Inequality operator.
   * @param other Another World object.
   * @return True if not equal, false otherwise.
   */
  bool operator!=(const World &other) const noexcept;

  /**
   * @brief Stream output operator.
   * @param os Output stream.
   * @param w World object.
   * @return Reference to the output stream.
   */
  template <typename CS>
  friend std::ostream &operator<<(std::ostream &os, const World<CS> &w) noexcept;
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
template <typename CoordTag> Int3 get_size(const World<CoordTag> &w) noexcept {
  return w.m_size;
}

/**
 * @brief Get the size of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The size in the specified dimension.
 */
template <typename CoordTag> int get_size(const World<CoordTag> &w, int i) noexcept {
  return get_size(w)[i];
}

/**
 * @brief Get the total number of grid points in the world.
 * @param w World object.
 * @return The total number of grid points.
 */
template <typename CoordTag> int total_size(const World<CoordTag> &w) noexcept {
  return get_size(w, 0) * get_size(w, 1) * get_size(w, 2);
}

/**
 * @brief Get the lower bounds of the world in a specific dimension.
 * @param w World object.
 * @return The lower bounds of the world.
 */
Real3 get_lower(const CartesianWorld &w) noexcept;

/**
 * @brief Get the lower bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The lower bound in the specified dimension.
 */
double get_lower(const CartesianWorld &w, int i) noexcept;

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @return The upper bounds of the world.
 */
Real3 get_upper(const CartesianWorld &w) noexcept;

/**
 * @brief Get the upper bounds of the world in a specific dimension.
 * @param w World object.
 * @param i Dimension index.
 * @return The upper bound in the specified dimension.
 */
double get_upper(const CartesianWorld &w, int i) noexcept;

/**
 * @brief Compute the physical coordinates corresponding to grid indices.
 * @param w World object.
 * @param indices Grid indices.
 * @return The physical coordinates.
 */
template <typename CoordTag>
Real3 to_coords(const World<CoordTag> &w, const Int3 &indices) noexcept {
  return to_coords(get_coordinate_system(w), indices);
}

/**
 * @brief Compute the grid indices corresponding to physical coordinates.
 * @param w World object.
 * @param coordinates Physical coordinates.
 * @return The grid indices.
 */
template <typename CoordTag>
Int3 to_indices(const World<CoordTag> &w, const Real3 &coordinates) noexcept {
  return to_index(get_coordinate_system(w), coordinates);
}

// Free function API for coordinate system and periodicity

/**
 * @brief Get the coordinate system of the world.
 * @param w World object.
 * @return The coordinate system of the world.
 */
template <typename CoordTag>
CoordinateSystem<CoordTag> get_coordinate_system(const World<CoordTag> &w) noexcept {
  return w.m_cs;
}

// For backward compatibility, might be removed in the future

inline const Real3 &get_spacing(const CartesianWorld &w) noexcept {
  return get_spacing(get_coordinate_system(w));
}

inline double get_spacing(const CartesianWorld &w, int i) noexcept {
  return get_spacing(get_coordinate_system(w), i);
}

inline const Real3 &get_origin(const CartesianWorld &w) noexcept {
  return get_offset(get_coordinate_system(w));
}

inline double get_origin(const CartesianWorld &w, int i) noexcept {
  return get_offset(get_coordinate_system(w), i);
}

} // namespace world

// export World class to the pfc namespace, so we hopefully don't have to write
// `world::World world = world::create_world(...)` kind of things :D
using World = world::CartesianWorld;

} // namespace pfc
