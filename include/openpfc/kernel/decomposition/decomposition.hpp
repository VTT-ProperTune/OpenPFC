// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file decomposition.hpp
 * @brief Domain decomposition for parallel MPI simulations
 *
 * @details
 * This file defines the Decomposition class and related utilities for distributing
 * a simulation domain across multiple MPI processes. Domain decomposition is
 * essential for parallel spectral method simulations, enabling efficient FFT
 * operations and field storage across distributed memory systems.
 *
 * The Decomposition class handles:
 * - Splitting the global World into local subdomains (one per MPI rank)
 * - Managing inbox/outbox regions for FFT pencil decomposition
 * - Coordinate transformations between global and local index spaces
 * - Integration with HeFFTe for distributed FFT operations
 *
 * Typical usage:
 * @code
 * #include <openpfc/kernel/data/domain.hpp>
 * #include <openpfc/kernel/decomposition/decomposition.hpp>
 * #include <openpfc/kernel/field/field_factory.hpp>
 *
 * auto domain = pfc::domain::create({128, 128, 128});
 * auto decomp = pfc::decomposition::create(domain, pfc::mpi::get_size());
 * int rank = pfc::mpi::get_rank();
 * auto field = pfc::data::field_from_subdomain<float>(decomp, rank, 1);
 * @endcode
 *
 * This file is part of the Core Infrastructure module, providing parallel
 * decomposition capabilities for distributed-memory HPC systems.
 *
 * @see kernel/data/world.hpp for global domain definition
 * @see fft.hpp for FFT operations using decomposition
 * @see factory/decomposition_factory.hpp for construction helpers
 */

#pragma once

#include <algorithm>
#include <array>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace pfc {

/**
 * @brief Namespace for decomposition-related classes and functions.
 */

namespace decomposition {

using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

using World = pfc::world::World;
using Int3 = pfc::types::Int3;

/**
 * @brief Describes a static, pure partitioning of the global simulation domain
 * into local subdomains.
 *
 * The Decomposition struct encapsulates how the global World domain is split
 * across compute units, such as MPI processes, OpenMP threads, or GPU tiles. It
 * represents the *ownership* layout, not how communication is performed.
 *
 * Each Decomposition instance defines the local subdomain assigned to the
 * current compute entity, including bounding box, size, and global offset. It
 * provides a consistent, backend-independent view of how the World is
 * subdivided.
 *
 *
 * ## Responsibilities
 *
 * - Partition the World into non-overlapping subdomains.
 * - Store the local bounding box for the current process/thread/tile.
 * - Provide basic global-to-local coordinate mappings.
 * - Support communication planning and field allocation.
 *
 *
 * ## Design Principles
 *
 * - **Immutable**: All members are set at construction; no mutation after
 *   creation.
 * - **Pure**: No behavior; only data. All logic is implemented via free
 *   functions in the `pfc::decomposition` namespace.
 * - **Backend-agnostic**: The decomposition itself contains no knowledge of
 *   MPI, GPU, or FFT specifics.
 * - **Strategy-based construction**: Backends (DifferentialOperators) define
 *   their requirements using a `DecompositionRequest`, and decomposition is
 *   created to satisfy that request.
 * - **Composable and inspectable**: Designed to be shared and reused across
 *   modules.
 *
 *
 * ## Integration with Backends
 *
 * Decomposition supports an inversion of control model where
 * *DifferentialOperator* (or any backend) declares its layout requirements via
 * a `DecompositionRequest`:
 *
 * ```cpp
 * struct FiniteDifferenceBackend {
 *   DecompositionRequest decomposition_request() const;
 * };
 * ```
 *
 * This request is passed to the decomposition builder:
 *
 * ```cpp
 * Decomposition decomp = pfc::decomposition::create(world, rank, size,
 * backend.decomposition_request());
 * ```
 *
 * This design allows:
 * - Clean separation of concerns between numerical kernels and layout logic.
 * - Support for specialized strategies (e.g., slab vs pencil, real vs complex).
 * - Reuse of decompositions across multiple algorithmic backends.
 *
 *
 * ## Usage Context
 *
 * - Used during startup to allocate Fields with correct local shape.
 * - Passed to communication planning logic to derive what needs to be
 *   exchanged.
 * - Required by FFT, finite difference, or hybrid backends to define their
 *   working layout.
 *
 *
 * ## Extensibility
 *
 * The decomposition system is extensible by:
 * - Adding new strategy types (`SplitStrategy`) or request properties.
 * - Supporting templated decomposition traits (e.g., GPU-aware or NUMA-aware
 *   partitions).
 * - Implementing high-level abstractions over common layouts (e.g., block,
 *   slab, tile).
 *
 *
 * ## Limitations
 *
 * - Assumes a structured, rectangular global World domain.
 * - Does not encode or perform communication (that's a separate layer).
 * - Does not (yet) support dynamic repartitioning or adaptive remeshing.
 */
struct Decomposition {

  /**
   * The global World this decomposition partitions.
   *
   * Stored **by value** so the `Decomposition` is self-contained: a
   * factory function may safely return a `Decomposition` whose source
   * `World` only existed in the factory's local scope. (Earlier this
   * member was `const World&`, which silently dangled in exactly that
   * pattern — see
   * `tests/unit/kernel/decomposition/test_decomposition_lifetime.cpp`.)
   */
  pfc::World m_global_world; ///< Backward compatibility: kept for migration.
  const std::array<int, 3> m_grid; ///< The number of parts in each dimension.
  std::vector<pfc::Box3i> m_local_boxes; ///< Local subdomain boxes (M1.3).
  pfc::Domain m_domain; ///< Global domain extracted from World (M1.3).

  Decomposition(const World &world, const Int3 &grid);

  friend std::ostream &operator<<(std::ostream &os, const Decomposition &d) {
    os << "Decomposition:\n";
    os << "  Global World: " << d.m_global_world << "\n";
    os << "  Grid: [" << d.m_grid.at(0) << ", " << d.m_grid.at(1) << ", "
       << d.m_grid.at(2) << "]\n";
    os << "  Local boxes: " << d.m_local_boxes.size() << " subdomains\n";
    return os;
  }
};

/**
 * @brief Get the global World that this decomposition partitions
 *
 * Returns a reference to the complete, unpartitioned computational domain.
 * This is the World that was split into subdomains.
 *
 * @param[in] decomposition The decomposition to query
 * @return Reference to the global World object
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({256, 256, 256});
 * auto decomp = decomposition::create(domain, {2, 2, 1});
 *
 * // Use the Domain directly (modern M2 approach)
 * auto coordinate_system = decomposition::domain(decomp);
 * std::cout << "Global size: "
 *           << pfc::domain::get_size(coordinate_system) << "\n";  // [256, 256, 256]
 * ```
 *
 * @see get_world() - alias for this function
 * @see get_subworld() - get a specific subdomain
 */
inline const auto &get_global_world(const Decomposition &decomposition) noexcept {
  return decomposition.m_global_world;
}

/**
 * @brief Alias for get_global_world()
 *
 * @param[in] decomposition The decomposition to query
 * @return Reference to the global World object
 *
 * @see get_global_world() - the function this aliases
 */
inline const auto &get_world(const Decomposition &decomposition) noexcept {
  return get_global_world(decomposition);
}

/**
 * @brief Get the decomposition grid pattern
 *
 * Returns the 3D grid layout showing how many subdomains exist in each
 * dimension. For example, [2, 2, 1] means a 2×2×1 grid = 4 total subdomains.
 *
 * @param[in] decomposition The decomposition to query
 * @return Array [nx, ny, nz] where total subdomains = nx * ny * nz
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({128, 128, 128});
 * auto decomp = decomposition::create(domain, {4, 2, 1});
 *
 * auto grid = decomposition::get_grid(decomp);
 * std::cout << "Grid: " << grid[0] << "×" << grid[1] << "×" << grid[2] << "\n";
 * std::cout << "Total domains: " << (grid[0] * grid[1] * grid[2]) << "\n";
 * ```
 *
 * @note The grid pattern affects communication overhead. Minimize surface area
 *       for better performance (use proc_setup_min_surface for automatic selection).
 *
 * @see get_num_domains() - total number of subdomains
 * @see create() - specify or automatically determine grid
 */
inline const auto &get_grid(const Decomposition &decomposition) noexcept {
  return decomposition.m_grid;
}

// Removed per M1.3: get_subworlds() and get_subworld() were the World accessor.
// Use the Box3i/Domain access instead: local_box() / global_box() / domain().

/**
 * @brief Create decomposition with explicit grid pattern
 *
 * Partitions the global World into subdomains according to the specified grid
 * layout [nx, ny, nz]. Total subdomains = nx * ny * nz.
 *
 * @param[in] world The global computational domain to partition
 * @param[in] grid Decomposition pattern [nx, ny, nz] in each dimension
 * @return Decomposition object containing all subdomains
 *
 * @example
 * **2×2×1 Decomposition (4 MPI ranks) - Domain-based (M2 modern usage)**
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({128, 128, 128});
 * auto decomp = decomposition::create(domain, {2, 2, 1});
 *
 * // Each rank gets 64×64×128 subdomain
 * auto grid = decomposition::get_grid(decomp);
 * std::cout << "Grid: [" << grid[0] << ", " << grid[1] << ", " << grid[2] << "]\n";
 *
 * // Allocate a field for the local rank's subdomain
 * int rank = pfc::mpi::get_rank();
 * auto field = pfc::data::field_from_subdomain<float>(decomp, rank, 1);
 * ```
 *
 * @example
 * **Slab Decomposition (1D splitting)**
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({256, 256, 256});
 * auto decomp = decomposition::create(domain, {1, 1, 8});  // Split only in Z
 *
 * // Each rank gets 256×256×32 slab
 * ```
 *
 * @deprecated Prefer `create(domain, grid)` with `pfc::domain::create()` for new
 *             code. The World-based overload is retained for backward compatibility
 *             with existing consumers.
 * @note Choose grid to minimize communication (minimize surface area between ranks).
 * @note For automatic grid selection, use create(domain, nparts) instead.
 * @note Grid dimensions must evenly divide World dimensions for optimal load
 * balance.
 *
 * @see create(world, nparts) - automatic grid selection
 * @see proc_setup_min_surface() - algorithm for optimal grid
 */
[[nodiscard]] Decomposition create(const World &world, const Int3 &grid);

/**
 * @brief Create decomposition with automatic grid selection
 *
 * Partitions the global World into the specified number of subdomains,
 * automatically choosing a grid pattern that minimizes communication surface
 * area (uses HeFFTe's proc_setup_min_surface algorithm).
 *
 * @param[in] world The global computational domain to partition
 * @param[in] nparts Number of subdomains (typically MPI size)
 * @return Decomposition with optimally chosen grid pattern
 *
 * @example
 * **Automatic Grid for 16 MPI Ranks - Domain-based (M2 modern usage)**
 * ```cpp
 * using namespace pfc;
 *
 * int size;
 * MPI_Comm_size(MPI_COMM_WORLD, &size);  // e.g., size = 16
 *
 * auto domain = pfc::domain::create({256, 256, 256});
 * auto decomp = decomposition::create(domain, size);
 *
 * auto grid = decomposition::get_grid(decomp);
 * // Likely chooses 4×4×1 or 4×2×2 (minimizes surface area)
 * std::cout << "Auto-selected grid: [" << grid[0] << ", "
 *           << grid[1] << ", " << grid[2] << "]\n";
 * ```
 *
 * @example
 * **Query Selected Grid**
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({200, 100, 50});
 * auto decomp = decomposition::create(domain, 8);
 *
 * auto grid = decomposition::get_grid(decomp);
 * std::cout << "For 8 ranks with domain [200, 100, 50]:\n";
 * std::cout << "  Chose grid [" << grid[0] << ", " << grid[1] << ", " << grid[2]
 *           << "]\n";
 * // Adapts to domain aspect ratio
 * ```
 *
 * @deprecated Prefer `create(domain, nparts)` with `pfc::domain::create()` for new
 *             code. The World-based overload is retained for backward compatibility
 *             with existing consumers.
 * @note This is the **recommended** method for most applications - let the
 *       algorithm choose the optimal grid.
 * @note The algorithm considers domain dimensions and communication patterns.
 * @note For manual control, use create(domain, grid) instead.
 *
 * @see create(world, grid) - manual grid specification
 * @see proc_setup_min_surface() - HeFFTe's grid selection algorithm
 */
[[nodiscard]] Decomposition create(const World &world, const int &nparts);

/**
 * @brief Domain-based decomposition creation with manual grid specification
 *
 * Partitions the global Domain into subdomains using a specified grid pattern.
 * This overload accepts Domain directly for M1 migration convenience.
 *
 * @param[in] domain The global computational domain to partition
 * @param[in] grid   Grid pattern {gx, gy, gz} (number of subdomains per axis)
 * @return Decomposition with the specified grid pattern
 *
 * @example
 * **Manual Grid with Domain**
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({256, 256, 256});
 * auto decomp = decomposition::create(domain, {2, 2, 1});
 *
 * auto grid = decomposition::get_grid(decomp);
 * // grid = [2, 2, 1]
 * ```
 *
 * @see create(World&, const Int3&) - World-based overload
 * @see create(domain, nparts) - automatic grid selection
 */
[[nodiscard]] Decomposition create(const Domain &domain, const Int3 &grid);

/**
 * @brief Domain-based decomposition creation with automatic grid selection
 *
 * Partitions the global Domain into the specified number of subdomains,
 * automatically choosing a grid pattern that minimizes communication surface
 * area. This overload accepts Domain directly for M1 migration convenience.
 *
 * @param[in] domain The global computational domain to partition
 * @param[in] nparts Number of subdomains (typically MPI size)
 * @return Decomposition with optimally chosen grid pattern
 *
 * @example
 * **Automatic Grid with Domain**
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({256, 256, 256});
 * auto decomp = decomposition::create(domain, 8);
 *
 * auto grid = decomposition::get_grid(decomp);
 * // Likely chooses 4×2×1 or 2×2×2 (minimizes surface area)
 * ```
 *
 * @see create(World&, const int&) - World-based overload
 * @see create(domain, grid) - manual grid specification
 */
[[nodiscard]] Decomposition create(const Domain &domain, const int &nparts);

/**
 * @brief Get the total number of subdomains
 *
 * Returns the count of subdomains in this decomposition. Equals the product
 * of grid dimensions: num_domains = grid[0] * grid[1] * grid[2].
 *
 * @param[in] decomposition The decomposition to query
 * @return Total number of subdomains (typically equals MPI size)
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({128, 128, 128});
 * auto decomp = decomposition::create(domain, {2, 2, 1});
 *
 * int num = decomposition::get_num_domains(decomp);
 * std::cout << "Total subdomains: " << num << "\n";  // 4
 *
 * auto grid = decomposition::get_grid(decomp);
 * assert(num == grid[0] * grid[1] * grid[2]);  // Always true
 * ```
 *
 * @example
 * **Validate MPI Size**
 * ```cpp
 * using namespace pfc;
 *
 * int mpi_size;
 * MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
 *
 * auto domain = pfc::domain::create({256, 256, 256});
 * auto decomp = decomposition::create(domain, mpi_size);
 *
 * int num_domains = decomposition::get_num_domains(decomp);
 * assert(num_domains == mpi_size);  // Should match
 * ```
 *
 * @see get_grid() - decomposition pattern
 */
inline int get_num_domains(const Decomposition &decomposition) noexcept {
  return static_cast<int>(decomposition.m_local_boxes.size());
}

// ---------------------------------------------------------------------------
// 0.2 canonical accessors (M1.3): expose the decomposition as `Domain` (global
// coordinate system) + `Box3i` (index ranges), the target types that replace
// the `World`-as-box conflation. Derived from the existing `World` members, so
// they are additive and numerically identical to the `get_subworld*` surface;
// consumers migrate onto them across M1.3–M1.4, after which the `World`
// accessors become the deprecated A0 shim.
// ---------------------------------------------------------------------------

/// The global coordinate system this decomposition partitions, as a `Domain`.
/// Derived from the stored Domain (M1.3 direct access).
[[nodiscard]] inline Domain domain(const Decomposition &decomposition) {
  return decomposition.m_domain;
}

/// The global index box `[lower, upper]` covered by the whole decomposition.
/// Derived from the local boxes (M1.3 direct access).
[[nodiscard]] inline Box3i global_box(const Decomposition &decomposition) {
  // Derive global extent from all local boxes (union of lower/upper bounds)
  if (decomposition.m_local_boxes.empty()) {
    return Box3i{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  }

  std::array<int, 3> global_low = decomposition.m_local_boxes[0].low;
  std::array<int, 3> global_high = decomposition.m_local_boxes[0].high;

  for (const auto &local_box : decomposition.m_local_boxes) {
    for (int d = 0; d < 3; ++d) {
      global_low[d] = std::min(global_low[d], local_box.low[d]);
      global_high[d] = std::max(global_high[d], local_box.high[d]);
    }
  }

  return Box3i::from_bounds(global_low, global_high);
}

/// The index box owned by subdomain `i` (typically the MPI rank).
/// Direct access to stored Box3i (M1.3).
/// @throws std::out_of_range if `i` is not a valid subdomain index.
[[nodiscard]] inline Box3i local_box(const Decomposition &decomposition, int i) {
  return decomposition.m_local_boxes.at(i);
}

// ---------------------------------------------------------------------------
// Backward compatibility accessor (M1.3b): get_subworld implemented using
// stored Box3i+Domain for legacy code that still expects World objects.
// Note: get_subworlds() (vector accessor) was removed per M1.3b - use
// local_box()/domain() accessors instead.
// ---------------------------------------------------------------------------

/**
 * @brief Get a specific subdomain as World (backward compatibility)
 *
 * Returns a World object for the specified rank, constructed from the stored
 * Box3i local box and Domain. This provides backward compatibility for code
 * that expects World objects from decomposition.
 *
 * @param[in] decomposition The decomposition to query
 * @param[in] rank The rank/subdomain index (0 to get_num_domains()-1)
 * @return World object representing the subdomain's index space and coordinate system
 * @throws std::out_of_range if rank is out of range
 *
 * @deprecated This compatibility function provides World objects for legacy code.
 *             New code should use Box3i/Domain accessors: local_box() / domain().
 *
 * @see local_box() - direct Box3i access (preferred for new code)
 * @see domain() - coordinate system access
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = pfc::domain::create({128, 128, 128});
 * auto decomp = decomposition::create(domain, {2, 2, 1});
 *
 * // Legacy usage (still supported for compatibility)
 * const World &subworld = decomposition::get_subworld(decomp, 0);
 *
 * // Preferred new usage (faster, uses modern M2 types)
 * Box3i local_box = decomposition::local_box(decomp, 0);
 * Domain coord_sys = decomposition::domain(decomp);
 * auto field = pfc::data::field_from_subdomain<float>(decomp, 0, 1);
 * ```
 */
[[nodiscard]] inline World
get_subworld(const Decomposition &decomp, int rank) {
  const Box3i &box = local_box(decomp, rank);
  const Domain &dom = domain(decomp);
  return World(box.low, box.high, dom);
}

} // namespace decomposition

using Decomposition = decomposition::Decomposition;

} // namespace pfc
