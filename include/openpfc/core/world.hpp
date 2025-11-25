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
 *
 * ## Extending World with Custom Coordinate Systems
 *
 * World is designed to work with **any coordinate system** you define. You can
 * add custom coordinate systems (cylindrical, spherical, curvilinear, etc.)
 * without modifying OpenPFC source code. This is OpenPFC's "Laboratory, Not
 * Fortress" philosophy in action.
 *
 * ### Requirements
 *
 * Your coordinate system must provide:
 *
 * 1. A **tag type** (empty struct) for template specialization
 * 2. A **CoordinateSystem<YourTag>** specialization with coordinate parameters
 * 3. **ADL-findable free functions** for coordinate transformations:
 *    - `Real3 to_coords(const CoordinateSystem<YourTag>&, const Int3& indices)`
 *    - `Int3 to_indices(const CoordinateSystem<YourTag>&, const Real3& coords)`
 *
 * ### Extension Pattern
 *
 * @code
 * // Step 1: Define your tag in your own namespace
 * namespace my_project {
 *     struct CylindricalTag {};
 * }
 *
 * // Step 2: Specialize CoordinateSystem in pfc::csys namespace
 * namespace pfc::csys {
 *     template<>
 *     struct CoordinateSystem<my_project::CylindricalTag> {
 *         const double m_r_min, m_r_max;
 *         const double m_theta_min, m_theta_max;
 *         const double m_z_min, m_z_max;
 *
 *         CoordinateSystem(double r0, double r1, double th0, double th1,
 *                          double z0, double z1)
 *             : m_r_min(r0), m_r_max(r1)
 *             , m_theta_min(th0), m_theta_max(th1)
 *             , m_z_min(z0), m_z_max(z1)
 *         {}
 *     };
 *
 *     // Step 3: Implement coordinate transformations (ADL extension point)
 *     inline Real3 cylindrical_to_coords(
 *         const CoordinateSystem<my_project::CylindricalTag>& cs,
 *         const Int3& indices
 *     ) {
 *         // Your cylindrical → Cartesian transformation
 *         double r = cs.m_r_min + ...;
 *         double theta = cs.m_theta_min + ...;
 *         double z = cs.m_z_min + ...;
 *         return {r * cos(theta), r * sin(theta), z};
 *     }
 * }
 *
 * // Step 4: Use with World - ADL automatically finds your functions!
 * using CylindricalWorld = pfc::World<my_project::CylindricalTag>;
 * CylindricalWorld world(cs, {64, 128, 32});  // Works seamlessly!
 * @endcode
 *
 * ### Key Benefits
 *
 * - **No source modifications**: Your extensions live in your code
 * - **Type-safe**: Compile-time checking catches errors
 * - **Zero overhead**: ADL resolution is at compile-time
 * - **Composable**: Mix and match coordinate systems easily
 *
 * ### Examples
 *
 * - **Complete working example**: `examples/17_custom_coordinate_system.cpp`
 * - **Comprehensive guide**: `docs/extending_openpfc/adl_extension_patterns.md`
 *
 * ### Learn More
 *
 * See the [ADL Extension Patterns
 * Guide](../../docs/extending_openpfc/adl_extension_patterns.md) for comprehensive
 * documentation on extending OpenPFC with custom components.
 */

#pragma once

#include <array>
#include <ostream>
#include <stdexcept>

#include "csys.hpp"
#include "strong_types.hpp"
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
             "See migration guide in documentation.")]]
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
 * @brief Get the grid dimensions of the simulation domain
 *
 * Returns the number of grid points in each direction [nx, ny, nz].
 * This defines the discrete resolution of the computational domain.
 *
 * @tparam T Coordinate system tag (e.g., CartesianTag)
 * @param world World object to query
 * @return Grid dimensions as Int3 array [nx, ny, nz]
 *
 * @note This is the total global size before domain decomposition
 * @note For efficient FFT, dimensions should be powers of 2 or have small prime
 * factors
 *
 * @example
 * ```cpp
 * auto world = world::create({256, 128, 64});
 * Int3 size = get_size(world);
 * // size = {256, 128, 64}
 * std::cout << "Grid: " << size[0] << "×" << size[1] << "×" << size[2] << "\n";
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_total_size() to get the product nx*ny*nz
 * @see get_size(world, index) to get size in specific dimension
 */
template <typename T> Int3 get_size(const World<T> &world) noexcept {
  return world.m_size;
}

/**
 * @brief Get the grid size in a specific dimension
 *
 * Returns the number of grid points in the specified dimension.
 * Convenience function equivalent to get_size(world)[index].
 *
 * @tparam T Coordinate system tag
 * @param world World object to query
 * @param index Dimension index (0=x, 1=y, 2=z)
 * @return Number of grid points in dimension 'index'
 *
 * @throws std::out_of_range if index is not 0, 1, or 2
 *
 * @note Index 0 corresponds to x-direction, 1 to y, 2 to z
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 200, 300});
 * int nx = get_size(world, 0);  // Returns 100
 * int ny = get_size(world, 1);  // Returns 200
 * int nz = get_size(world, 2);  // Returns 300
 * ```
 *
 * Time complexity: O(1)
 *
 * @see get_size(world) to get all dimensions at once
 */
template <typename T> int get_size(const World<T> &world, int index) {
  return get_size(world).at(index);
}

/**
 * @brief Get the total number of grid points in the domain
 *
 * Computes the product nx × ny × nz, which is the total number of
 * discrete grid points in the computational domain.
 *
 * @tparam T Coordinate system tag
 * @param world World object to query
 * @return Total grid points (nx * ny * nz)
 *
 * @note This is the memory footprint for storing one scalar field
 * @note For parallel simulations, each MPI rank stores only a subset
 *
 * @example
 * ```cpp
 * auto world = world::create({128, 128, 128});
 * size_t total = get_total_size(world);
 * // total = 2,097,152 (128³)
 *
 * // Memory for double precision field:
 * size_t bytes = total * sizeof(double);
 * std::cout << "Field requires " << bytes / (1024*1024) << " MB\n";
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_size() for individual dimensions
 * @see physical_volume() for physical domain size (not grid points)
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
 * @brief Convert grid indices to physical coordinates
 *
 * Transforms discrete grid indices (i, j, k) to continuous physical
 * coordinates (x, y, z) based on the world's coordinate system.
 *
 * For Cartesian coordinates: x = origin[d] + i * spacing[d]
 *
 * @tparam T Coordinate system tag
 * @param world World object defining the coordinate system
 * @param indices Grid indices [i, j, k] where 0 ≤ i < nx, etc.
 * @return Physical coordinates [x, y, z]
 *
 * @note This is a zero-cost inline function (no overhead)
 * @note Indices are not bounds-checked for performance; ensure valid indices
 * @note For periodic domains, periodicity is handled by the coordinate system
 *
 * @warning Passing out-of-bounds indices results in undefined physical coordinates
 *
 * @example
 * ```cpp
 * // Create 100³ grid with 0.1 spacing starting at origin
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 *
 * // Grid point at center
 * Real3 center = to_coords(world, {50, 50, 50});
 * // center = {5.0, 5.0, 5.0}
 *
 * // Corner points
 * Real3 origin = to_coords(world, {0, 0, 0});     // {0.0, 0.0, 0.0}
 * Real3 far = to_coords(world, {99, 99, 99});     // {9.9, 9.9, 9.9}
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see to_indices() for inverse transformation (physical → grid)
 * @see get_spacing() to understand coordinate scaling
 * @see get_origin() for coordinate system offset
 */
template <typename T>
inline const auto to_coords(const World<T> &world, const Int3 &indices) noexcept {
  return to_coords(get_coordinate_system(world), indices);
}

/**
 * @brief Convert physical coordinates to grid indices
 *
 * Transforms continuous physical coordinates (x, y, z) to discrete grid
 * indices (i, j, k) based on the world's coordinate system.
 *
 * For Cartesian coordinates: i = round((x - origin[d]) / spacing[d])
 *
 * @tparam T Coordinate system tag
 * @param world World object defining the coordinate system
 * @param coords Physical coordinates [x, y, z]
 * @return Grid indices [i, j, k] (nearest grid point)
 *
 * @note Uses nearest-neighbor rounding (no interpolation)
 * @note For periodic domains, coordinates outside bounds wrap around
 * @note Returned indices may be outside [0, size) for non-periodic boundaries
 *
 * @warning For non-periodic domains, caller must check bounds validity
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 *
 * // Find grid point nearest to physical location
 * Real3 pos = {2.53, 4.78, 7.21};
 * Int3 idx = to_indices(world, pos);
 * // idx = {25, 48, 72} (nearest grid points)
 *
 * // Inverse transformation
 * Real3 exact = to_coords(world, idx);
 * // exact = {2.5, 4.8, 7.2} (snapped to grid)
 * ```
 *
 * @example
 * ```cpp
 * // Common usage: sample field at arbitrary position
 * auto world = world::create({64, 64, 64});
 * Field<double> density(world);
 *
 * Real3 sample_point = {31.5, 31.5, 31.5};
 * Int3 idx = to_indices(world, sample_point);
 * double value = density[idx];
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see to_coords() for inverse transformation (grid → physical)
 * @see DiscreteField::interpolate() for higher-order interpolation
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

/**
 * @brief Get the grid spacing in all dimensions
 *
 * Returns the physical distance between adjacent grid points in each
 * direction. For periodic boundaries, spacing is (upper - lower) / size.
 * For non-periodic, spacing is (upper - lower) / (size - 1).
 *
 * @param world World object to query
 * @return Grid spacing [dx, dy, dz]
 *
 * @note Spacing determines the resolution of spectral derivatives
 * @note Smaller spacing = higher resolution but smaller time steps for stability
 * @note For FFT efficiency, uniform spacing is recommended
 *
 * @example
 * ```cpp
 * // Uniform spacing
 * auto world1 = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * Real3 dx1 = get_spacing(world1);  // {0.1, 0.1, 0.1}
 *
 * // Non-uniform spacing (e.g., refined in one direction)
 * auto world2 = world::create({100, 100, 200}, {0, 0, 0}, {0.1, 0.1, 0.05});
 * Real3 dx2 = get_spacing(world2);  // {0.1, 0.1, 0.05}
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_origin() for coordinate system offset
 * @see to_coords() which uses spacing for coordinate transforms
 */
inline const Real3 &get_spacing(const CartesianWorld &world) noexcept {
  return get_spacing(get_coordinate_system(world));
}

/**
 * @brief Get the grid spacing in a specific dimension
 *
 * Returns the physical distance between adjacent grid points in the
 * specified dimension.
 *
 * @param world World object to query
 * @param index Dimension index (0=x, 1=y, 2=z)
 * @return Grid spacing in dimension 'index'
 *
 * @throws std::out_of_range if index is not 0, 1, or 2
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.2, 0.3});
 * double dx = get_spacing(world, 0);  // Returns 0.1
 * double dy = get_spacing(world, 1);  // Returns 0.2
 * double dz = get_spacing(world, 2);  // Returns 0.3
 * ```
 *
 * Time complexity: O(1)
 *
 * @see get_spacing(world) to get all spacings at once
 */
inline double get_spacing(const CartesianWorld &world, int index) noexcept {
  return get_spacing(get_coordinate_system(world), index);
}

/**
 * @brief Get the physical coordinates of the grid origin
 *
 * Returns the physical coordinates of the grid point at index (0, 0, 0).
 * This defines the offset of the coordinate system.
 *
 * @param world World object to query
 * @return Physical coordinates of origin [x0, y0, z0]
 *
 * @note The origin can be any value; common choices are (0, 0, 0) or domain center
 * @note Changing origin shifts all physical coordinates but doesn't affect
 * computation
 *
 * @example
 * ```cpp
 * // Origin at (0, 0, 0) - most common
 * auto world1 = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * Real3 origin1 = get_origin(world1);  // {0, 0, 0}
 *
 * // Centered domain: origin at negative coordinates
 * auto world2 = world::create({100, 100, 100}, {-5, -5, -5}, {0.1, 0.1, 0.1});
 * Real3 origin2 = get_origin(world2);  // {-5, -5, -5}
 * // Domain spans [-5, 4.9] in each direction
 * ```
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 *
 * @see get_lower_bounds() for equivalent function
 * @see to_coords() which uses origin for coordinate transforms
 */
inline const Real3 &get_origin(const CartesianWorld &world) noexcept {
  return get_offset(get_coordinate_system(world));
}

/**
 * @brief Get the origin coordinate in a specific dimension
 *
 * Returns the physical coordinate of the grid origin in the specified dimension.
 *
 * @param world World object to query
 * @param index Dimension index (0=x, 1=y, 2=z)
 * @return Origin coordinate in dimension 'index'
 *
 * @throws std::out_of_range if index is not 0, 1, or 2
 *
 * @example
 * ```cpp
 * auto world = world::create({100, 100, 100}, {-1, 0, 5}, {0.1, 0.1, 0.1});
 * double x0 = get_origin(world, 0);  // Returns -1.0
 * double y0 = get_origin(world, 1);  // Returns 0.0
 * double z0 = get_origin(world, 2);  // Returns 5.0
 * ```
 *
 * Time complexity: O(1)
 *
 * @see get_origin(world) to get all origin coordinates
 */
inline double get_origin(const CartesianWorld &world, int index) noexcept {
  return get_offset(get_coordinate_system(world), index);
}

// ============================================================================
// World Convenience Query Functions
// ============================================================================

/**
 * @brief Compute physical volume of domain
 *
 * Returns the total physical volume (or area in 2D, length in 1D) of the
 * simulation domain.
 *
 * @param world World instance
 * @return Physical volume V = Lx * Ly * Lz where L = spacing * size
 *
 * @note For Cartesian coordinates, this is the product of all physical dimensions
 * @note Works correctly for 1D, 2D, and 3D domains
 *
 * @code
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * double vol = world::physical_volume(world);  // Returns 1000.0
 * @endcode
 *
 * Time complexity: O(1)
 * Space complexity: O(1)
 */
template <typename T> inline double physical_volume(const World<T> &world) noexcept {
  const auto spacing = get_spacing(world);
  const auto size = get_size(world);
  return spacing[0] * spacing[1] * spacing[2] * size[0] * size[1] * size[2];
}

/**
 * @brief Check if domain is 1D (only x-direction has > 1 point)
 *
 * A domain is considered 1D if only the first dimension has more than
 * one grid point.
 *
 * @param world World instance
 * @return true if only x-direction is active (nx > 1, ny = 1, nz = 1)
 *
 * @code
 * auto world1d = world::create({100, 1, 1});
 * bool is_1d = world::is_1d(world1d);  // Returns true
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline bool is_1d(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return (size[0] > 1) && (size[1] == 1) && (size[2] == 1);
}

/**
 * @brief Check if domain is 2D (x and y have > 1 point, z has 1)
 *
 * A domain is considered 2D if the first two dimensions have more than
 * one grid point and the third has exactly one.
 *
 * @param world World instance
 * @return true if x and y directions are active (nx > 1, ny > 1, nz = 1)
 *
 * @code
 * auto world2d = world::create({64, 64, 1});
 * bool is_2d = world::is_2d(world2d);  // Returns true
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline bool is_2d(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return (size[0] > 1) && (size[1] > 1) && (size[2] == 1);
}

/**
 * @brief Check if domain is 3D (all dimensions have > 1 point)
 *
 * A domain is considered 3D if all three dimensions have more than
 * one grid point.
 *
 * @param world World instance
 * @return true if all three directions are active (nx > 1, ny > 1, nz > 1)
 *
 * @code
 * auto world3d = world::create({32, 32, 32});
 * bool is_3d = world::is_3d(world3d);  // Returns true
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline bool is_3d(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return (size[0] > 1) && (size[1] > 1) && (size[2] > 1);
}

/**
 * @brief Get dimensionality as integer
 *
 * Returns 1, 2, or 3 based on how many dimensions have more than one
 * grid point. Returns 0 for degenerate case where all dimensions have
 * size 1.
 *
 * @param world World instance
 * @return 1, 2, 3, or 0 (degenerate) based on active dimensions
 *
 * @code
 * auto world2d = world::create({64, 64, 1});
 * int dim = world::dimensionality(world2d);  // Returns 2
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline int dimensionality(const World<T> &world) noexcept {
  if (is_3d(world)) return 3;
  if (is_2d(world)) return 2;
  if (is_1d(world)) return 1;
  return 0; // Degenerate case (all dimensions size 1)
}

/**
 * @brief Get physical lower bounds (origin corner)
 *
 * Returns the physical coordinates of the grid point at index (0, 0, 0).
 * This is the lower corner of the domain.
 *
 * @param world World instance
 * @return Physical coordinates of (0, 0, 0) grid point
 *
 * @code
 * auto world = world::create({100, 100, 100}, {-5, -5, 0}, {0.1, 0.1, 0.1});
 * Real3 lower = world::get_lower_bounds(world);  // Returns {-5, -5, 0}
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline Real3 get_lower_bounds(const World<T> &world) noexcept {
  return to_coords(world, {0, 0, 0});
}

/**
 * @brief Get physical upper bounds (far corner)
 *
 * Returns the physical coordinates of the grid point at the maximum indices
 * (nx-1, ny-1, nz-1). This is the upper corner of the domain.
 *
 * @param world World instance
 * @return Physical coordinates of (nx-1, ny-1, nz-1) grid point
 *
 * @code
 * auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});
 * Real3 upper = world::get_upper_bounds(world);  // Returns {9.9, 9.9, 9.9}
 * @endcode
 *
 * Time complexity: O(1)
 */
template <typename T> inline Real3 get_upper_bounds(const World<T> &world) noexcept {
  const auto size = get_size(world);
  return to_coords(world, {size[0] - 1, size[1] - 1, size[2] - 1});
}

// ============================================================================
// World Construction Helpers
// ============================================================================

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

// export World class to the pfc namespace, so we hopefully don't have to write
// `world::World world = world::create_world(...)` kind of things :D
using World = world::CartesianWorld;

} // namespace pfc
