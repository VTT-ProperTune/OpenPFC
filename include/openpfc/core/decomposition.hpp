// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
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
 * pfc::World global_world = pfc::world::create({128, 128, 128});
 * pfc::decomposition::Decomposition decomp(global_world, MPI_COMM_WORLD);
 *
 * // Access local subdomain
 * pfc::World local_world = decomp.get_world();
 *
 * // Get FFT inbox/outbox for spectral operations
 * auto inbox = decomp.get_inbox();
 * auto outbox = decomp.get_outbox();
 * @endcode
 *
 * This file is part of the Core Infrastructure module, providing parallel
 * decomposition capabilities for distributed-memory HPC systems.
 *
 * @see core/world.hpp for global domain definition
 * @see fft.hpp for FFT operations using decomposition
 * @see factory/decomposition_factory.hpp for construction helpers
 */

#pragma once

#include "openpfc/core/world.hpp"
#include <array>
#include <heffte.h>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace pfc {

/**
 * @brief Namespace for decomposition-related classes and functions.
 */

namespace world {

/**
 * @brief Construct a new World object from an existing one and a box.
 */
template <typename T>
inline auto create(const World<T> &world, const heffte::box3d<int> &box) {
  return World(box.low, box.high, get_coordinate_system(world));
}

template <typename T> inline auto to_indices(const World<T> &world) {
  std::array<int, 3> lower = get_lower(world);
  std::array<int, 3> upper = get_upper(world);
  return heffte::box3d<int>(lower, upper);
}

template <typename T>
inline auto split_world(const World<T> &world, const Int3 &grid) {
  std::vector<World<T>> sub_worlds;
  for (const auto &box : heffte::split_world(to_indices(world), grid)) {
    sub_worlds.push_back(create(world, box));
  }
  return sub_worlds;
}

} // namespace world

namespace decomposition {

using pfc::csys::CartesianTag;
using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;

using World = pfc::world::World<CartesianTag>;
using Int3 = pfc::types::Int3;

using heffte::proc_setup_min_surface;

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

  const pfc::World &m_global_world; ///< The World object.
  const std::array<int, 3> m_grid;  ///< The number of parts in each dimension.
  const std::vector<pfc::World> m_subworlds; ///< The sub-worlds for each part.

  Decomposition(const World &world, const Int3 grid)
      : m_global_world(world), m_grid(grid), m_subworlds(split_world(world, grid)) {}

  friend std::ostream &operator<<(std::ostream &os, const Decomposition &d) {
    os << "Decomposition:\n";
    os << "  Global World: " << d.m_global_world << "\n";
    os << "  Grid: [" << d.m_grid.at(0) << ", " << d.m_grid.at(1) << ", "
       << d.m_grid.at(2) << "]\n";
    os << "  Sub-worlds:\n";
    for (size_t i = 0; i < d.m_subworlds.size(); ++i) {
      os << "    Sub-world " << i << ": " << d.m_subworlds[i] << "\n";
    }
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
 * auto world = world::create({256, 256, 256}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, {2, 2, 1});
 *
 * auto global = decomposition::get_global_world(decomp);
 * std::cout << "Global domain: " << world::get_size(global) << "\n";  // [256, 256,
 * 256]
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
 * auto world = world::create({128, 128, 128}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, {4, 2, 1});
 *
 * auto grid = decomposition::get_grid(decomp);
 * std::cout << "Grid: " << grid[0] << "×" << grid[1] << "×" << grid[2] << "\n";  //
 * 4×2×1 std::cout << "Total domains: " << (grid[0] * grid[1] * grid[2]) << "\n";  //
 * 8
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

/**
 * @brief Get all subdomains (local World instances)
 *
 * Returns a vector containing all partitioned subdomains. Each subdomain is a
 * World representing a rank-local portion of the global domain.
 *
 * @param[in] decomposition The decomposition to query
 * @return Vector of World objects, one per subdomain (size = nx*ny*nz)
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto world = world::create({100, 100, 100}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, {2, 2, 1});  // 4 subdomains
 *
 * auto subworlds = decomposition::get_subworlds(decomp);
 * for (int i = 0; i < subworlds.size(); ++i) {
 *     auto size = world::get_size(subworlds[i]);
 *     std::cout << "Rank " << i << ": " << size << "\n";  // Each is 50×50×100
 * }
 * ```
 *
 * @note Subdomains are ordered consistently with MPI rank assignment (in most
 * cases).
 * @note All subdomains are non-overlapping and collectively cover the global World.
 *
 * @see get_subworld() - get a single subdomain by index
 * @see get_num_domains() - number of subdomains
 */
inline const auto &get_subworlds(const Decomposition &decomposition) noexcept {
  return decomposition.m_subworlds;
}

/**
 * @brief Get a specific subdomain by index
 *
 * Returns the subdomain (local World) assigned to the specified index. In MPI
 * contexts, the index typically corresponds to the MPI rank.
 *
 * @param[in] decomposition The decomposition to query
 * @param[in] i Index of the subdomain to retrieve (0 to num_domains-1)
 * @return Reference to the World representing subdomain i
 *
 * @throws std::out_of_range If i >= num_domains
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto world = world::create({200, 200, 200}, {0.5, 0.5, 0.5});
 * auto decomp = decomposition::create(world, 4);  // 4 subdomains (auto grid)
 *
 * int rank;
 * MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 *
 * // Get this rank's local subdomain
 * auto local_world = decomposition::get_subworld(decomp, rank);
 * auto local_size = world::get_size(local_world);
 * std::cout << "Rank " << rank << " owns: " << local_size << "\n";
 * ```
 *
 * @note In MPI applications, typically each rank accesses get_subworld(decomp,
 * rank).
 * @note Bounds checking is performed; invalid indices throw std::out_of_range.
 *
 * @see get_subworlds() - get all subdomains
 * @see get_num_domains() - valid range for index i
 */
inline const auto &get_subworld(const Decomposition &decomposition, int i) {
  return get_subworlds(decomposition).at(i);
}

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
 * **2×2×1 Decomposition (4 MPI ranks)**
 * ```cpp
 * using namespace pfc;
 *
 * auto world = world::create({128, 128, 128}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, {2, 2, 1});
 *
 * // Each rank gets 64×64×128 subdomain
 * auto grid = decomposition::get_grid(decomp);
 * std::cout << "Grid: [" << grid[0] << ", " << grid[1] << ", " << grid[2] << "]\n";
 * ```
 *
 * @example
 * **Slab Decomposition (1D splitting)**
 * ```cpp
 * using namespace pfc;
 *
 * auto world = world::create({256, 256, 256}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, {1, 1, 8});  // Split only in Z
 *
 * // Each rank gets 256×256×32 slab
 * ```
 *
 * @note Choose grid to minimize communication (minimize surface area between ranks).
 * @note For automatic grid selection, use create(world, nparts) instead.
 * @note Grid dimensions must evenly divide World dimensions for optimal load
 * balance.
 *
 * @see create(world, nparts) - automatic grid selection
 * @see proc_setup_min_surface() - algorithm for optimal grid
 */
inline const auto create(const World &world, const Int3 &grid) noexcept {
  return Decomposition(world, grid);
};

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
 * **Automatic Grid for 16 MPI Ranks**
 * ```cpp
 * using namespace pfc;
 *
 * int size;
 * MPI_Comm_size(MPI_COMM_WORLD, &size);  // e.g., size = 16
 *
 * auto world = world::create({256, 256, 256}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, size);
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
 * auto world = world::create({200, 100, 50}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, 8);
 *
 * auto grid = decomposition::get_grid(decomp);
 * std::cout << "For 8 ranks with domain [200, 100, 50]:\n";
 * std::cout << "  Chose grid [" << grid[0] << ", " << grid[1] << ", " << grid[2] <<
 * "]\n";
 * // Adapts to domain aspect ratio
 * ```
 *
 * @note This is the **recommended** method for most applications - let the
 *       algorithm choose the optimal grid.
 * @note The algorithm considers domain dimensions and communication patterns.
 * @note For manual control, use create(world, grid) instead.
 *
 * @see create(world, grid) - manual grid specification
 * @see proc_setup_min_surface() - HeFFTe's grid selection algorithm
 */
inline const auto create(const World &world, const int &nparts) noexcept {
  auto indices = to_indices(world);
  auto grid = proc_setup_min_surface(indices, nparts);
  return create(world, grid);
}

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
 * auto world = world::create({128, 128, 128}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, {2, 2, 1});
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
 * auto world = world::create({256, 256, 256}, {1.0, 1.0, 1.0});
 * auto decomp = decomposition::create(world, mpi_size);
 *
 * int num_domains = decomposition::get_num_domains(decomp);
 * assert(num_domains == mpi_size);  // Should match
 * ```
 *
 * @see get_grid() - decomposition pattern
 * @see get_subworlds() - access all subdomains
 */
inline int get_num_domains(const Decomposition &decomposition) noexcept {
  return get_subworlds(decomposition).size();
}

} // namespace decomposition

using Decomposition = decomposition::Decomposition;

} // namespace pfc
