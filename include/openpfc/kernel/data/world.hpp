// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file world.hpp
 * @brief World class definition and unified interface
 *
 * @details
 * The `World` class defines the **global simulation domain** in
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
 * - **Zero-overhead abstractions**: Inline functions, no runtime polymorphism
 * - **Explicit over implicit**: Clear, self-documenting APIs
 *
 * ## Status (0.2)
 *
 * `World` is a deprecated thin wrapper over `pfc::Domain` with a `Box3i`
 * subdomain member for Gen-1 compatibility. This is the M1 A0 adapter: World
 * provides deprecated member methods for Gen-1 source compatibility; framework
 * code should use `Domain` + `Box3i` directly. New code should not use World or
 * its deprecated member methods.
 *
 * @see world_factory.hpp for World creation functions
 * @see world_queries.hpp for queries and coordinate transforms
 * @see world_helpers.hpp for convenience constructors
 */

#pragma once
#include <array>
#include <ostream>
#include <stdexcept>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/types.hpp>

// Deprecation attribute guard
#if !defined(OPENPFC_SUPPRESS_LEGACY_WARNINGS)
#if defined(__GNUC__) || defined(__clang__)
#define OPENPFC_DEPRECATED_API [[deprecated("World is deprecated; use pfc::Domain + pfc::world free functions instead")]]
#elif defined(_MSC_VER)
#define OPENPFC_DEPRECATED_API __declspec(deprecated("World is deprecated; use pfc::Domain + pfc::world free functions instead"))
#else
#define OPENPFC_DEPRECATED_API
#endif
#else
#define OPENPFC_DEPRECATED_API
#endif

namespace pfc::world {

using pfc::Box3i;
using pfc::Domain;
using pfc::types::Int3;

/**
 * @brief Represents the global simulation domain (the "world").
 *
 * World is a deprecated thin wrapper over Domain with a Box3i subdomain member
 * for Gen-1 compatibility. The wrapper maintains exact ABI compatibility for
 * existing code while establishing Domain as the primary abstraction.
 *
 * As of the 0.2 M1 refactor this is the **A0 deprecated shim** over the
 * canonical `Domain` (see `domain.hpp`). The member methods below are provided
 * only for Gen-1 source compatibility and are deprecated; new code should
 * prefer `Domain` + `Box3i` with the free functions in `world_queries.hpp`.
 */
struct OPENPFC_DEPRECATED_API World final {
  Box3i subdomain_;           ///< Local subdomain box for subdomain role
  Domain domain_;             ///< Global Cartesian coordinate system (origin/spacing/periodic)

  // ========================================================================
  // Deprecated member methods (A0 shim for Gen-1 compatibility)
  // ========================================================================

  /**
   * @brief Get the global domain box.
   *
   * @deprecated Use `world::get_size()` and `world::get_lower/upper_bounds()` instead.
   */
  OPENPFC_DEPRECATED_API Box3i get_domain() const {
    return pfc::domain::index_box(domain_);
  }

  /**
   * @brief Get the subdomain box (local subdomain).
   *
   * @deprecated Use `world::get_lower()` and `world::get_upper()` instead.
   */
  OPENPFC_DEPRECATED_API Box3i get_subdomain() const {
    return subdomain_;
  }

  /**
   * @brief Total size of the domain.
   *
   * @deprecated Use `world::get_size()` instead.
   */
  OPENPFC_DEPRECATED_API int size() const {
    return static_cast<int>(pfc::domain::get_total_size(domain_));
  }

  /**
   * @brief Size in a specific dimension.
   *
   * @deprecated Use `world::get_size(world, dim)` instead.
   */
  OPENPFC_DEPRECATED_API int get_size(int dim) const {
    return pfc::domain::get_size(domain_, dim);
  }

  /**
   * @brief Lower bound (origin) in a specific dimension.
   *
   * @deprecated Use `world::get_lower(world, dim)` instead.
   */
  OPENPFC_DEPRECATED_API int origin(int dim) const {
    return subdomain_.low[dim];
  }

  /**
   * @brief Upper bound in a specific dimension.
   *
   * @deprecated Use `world::get_upper(world, dim)` instead.
   */
  OPENPFC_DEPRECATED_API int upper(int dim) const {
    return subdomain_.high[dim];
  }

  /**
   * @brief Set the subdomain box.
   *
   * @deprecated World is deprecated; use Domain + Box3i directly.
   */
  OPENPFC_DEPRECATED_API void set_subdomain(const Box3i &subdomain) {
    subdomain_ = subdomain;
  }

  // ========================================================================
  // Core constructors and operators (non-deprecated)
  // ========================================================================

  /**
   * @brief Constructs a World object.
   * @param lower Lower index bounds of the world.
   * @param upper Upper index bounds of the world.
   * @param domain Coordinate system (origin/spacing/periodicity) this box lives in.
   */
  explicit World(const Int3 &lower, const Int3 &upper, const Domain &domain);

  /**
   * @brief Get the periodicity flags for the world.
   * @return Bool3 array with periodicity flags [periodic_x, periodic_y, periodic_z]
   */
  Bool3 get_periodicity() const;

  /**
   * @brief Equality operator.
   * @param other Another World object.
   * @return True if equal, false otherwise.
   */
  bool operator==(const World &other) const noexcept {
    return subdomain_ == other.subdomain_ && domain_ == other.domain_;
  }

  /**
   * @brief Inequality operator.
   * @param other Another World object.
   * @return True if not equal, false otherwise.
   */
  bool operator!=(const World &other) const noexcept {
    return !(*this == other);
  }

  /**
   * @brief Stream output operator.
   * @param os Output stream.
   * @param w World object.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const World &w);
};

/// Deprecated alias retained for source compatibility (equals `World`).
using CartesianWorld = World;

} // namespace pfc::world

namespace pfc {
// Export World to pfc namespace for convenient usage
using World = world::CartesianWorld;
} // namespace pfc

// Include World functionality modules
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_helpers.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
