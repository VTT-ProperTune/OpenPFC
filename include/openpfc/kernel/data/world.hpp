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
 * `World` is a plain 3D Cartesian value type and a deprecated compatibility
 * shim over `pfc::Domain` (see `domain.hpp`). New code should prefer `Domain`
 * + `Box3i` directly; `World` is retained only so legacy call sites compile.
 *
 * @see world_factory.hpp for World creation functions
 * @see world_queries.hpp for queries and coordinate transforms
 * @see world_helpers.hpp for convenience constructors
 */

#pragma once

#include <array>
#include <ostream>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>

namespace pfc::world {

using pfc::Box3i;
using pfc::Domain;
using pfc::types::Int3;

/**
 * @brief Represents the global simulation domain (the "world").
 *
 * The World class defines the size of the global simulation domain and
 * coordinate system. It is a purely functional object with no mutable state,
 * constructed once and immutable thereafter. This design enhances correctness,
 * thread safety, testability, and reproducibility.
 *
 * As of the 0.2 M1 refactor this is a plain (non-template) 3D Cartesian type:
 * the coordinate-system tag parameter had exactly one instantiation and is being
 * removed. `World` is the deprecated **A0 shim** over the canonical `Domain`
 * (see `domain.hpp`); framework code migrates to `Domain` + `Box3i`.
 *
 * @see world_factory.hpp for construction
 * @see world_queries.hpp for accessing properties
 */
struct World final {
  const Box3i m_box; ///< Index range [low, high] + size (subdomain role)
  const Domain
      m_domain; ///< Global Cartesian coordinate system (origin/spacing/periodic)

  /**
   * @brief Constructs a World object.
   * @param lower Lower index bounds of the world.
   * @param upper Upper index bounds of the world.
   * @param domain Coordinate system (origin/spacing/periodicity) this box lives in.
   */
  explicit World(const Int3 &lower, const Int3 &upper, const Domain &domain);

  /**
   * @brief Equality operator.
   * @param other Another World object.
   * @return True if equal, false otherwise.
   */
  bool operator==(const World &other) const noexcept {
    return m_box == other.m_box && m_domain == other.m_domain;
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
