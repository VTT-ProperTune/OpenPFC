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
 * - Splitting the global Domain into local subdomains (one per MPI rank)
 * - Managing inbox/outbox regions for FFT pencil decomposition
 * - Coordinate transformations between global and local index spaces
 * - Integration with HeFFTe for distributed FFT operations
 *
 * Typical usage:
 * @code
 * pfc::Domain domain = pfc::domain::create({128, 128, 128});
 * auto decomp = pfc::decomposition::create(domain, pfc::mpi::get_size());
 * const pfc::Box3i local = pfc::decomposition::local_box(decomp,
 * pfc::mpi::get_rank());
 * @endcode
 *
 * This file is part of the Core Infrastructure module, providing parallel
 * decomposition capabilities for distributed-memory HPC systems.
 *
 * @see kernel/data/domain.hpp for global domain definition
 * @see fft.hpp for FFT operations using decomposition
 * @see factory/decomposition_factory.hpp for construction helpers
 */

#pragma once

#include <algorithm>
#include <array>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
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

using Int3 = pfc::types::Int3;

/**
 * @brief Global geometry and rank-owned boxes for a rectangular domain.
 *
 * Geometry is stored by value, independently of the source Domain's lifetime.
 * The process grid uses x-fastest rank order. No communication is performed
 * by this data type; MPI halo and FFT operations use its layout separately.
 *
 * The grid is const, but m_domain and m_local_boxes are public mutable data.
 * Treat a published layout as stable while fields, halo plans or FFT plans
 * refer to it; changing these members does not rebuild their dependent state.
 */
struct Decomposition {

  const std::array<int, 3> m_grid;       ///< The number of parts in each dimension.
  std::vector<pfc::Box3i> m_local_boxes; ///< Local subdomain boxes.
  pfc::Domain m_domain;                  ///< Global Domain this partitions.

  Decomposition(const Domain &domain, const Int3 &grid);

  friend std::ostream &operator<<(std::ostream &os, const Decomposition &d) {
    os << "Decomposition:\n";
    os << "  Domain: " << d.m_domain << "\n";
    os << "  Grid: [" << d.m_grid.at(0) << ", " << d.m_grid.at(1) << ", "
       << d.m_grid.at(2) << "]\n";
    os << "  Local boxes: " << d.m_local_boxes.size() << " subdomains\n";
    return os;
  }
};

/**
 * @brief Get the global Domain that this decomposition partitions
 *
 * Returns a reference to the complete, unpartitioned computational domain.
 * This is the Domain that was split into subdomains.
 *
 * @param[in] decomposition The decomposition to query
 * @return Reference to the global Domain object
 *
 * @example
 * ```cpp
 * using namespace pfc;
 *
 * auto domain = domain::create(GridSize({256, 256, 256}), PhysicalOrigin({0.0, 0.0,
 * 0.0}), GridSpacing({1.0, 1.0, 1.0})); auto decomp = decomposition::create(domain,
 * {2, 2, 1});
 *
 * auto global = decomposition::get_global_domain(decomp);
 * std::cout << "Global domain: " << domain::get_size(global) << "\n";  // [256, 256,
 * 256]
 * ```
 *
 * @see local_box() for a rank-owned index box
 */
inline const Domain &get_global_domain(const Decomposition &decomposition) noexcept {
  return decomposition.m_domain;
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
 * auto domain = domain::create(GridSize({128, 128, 128}), PhysicalOrigin({0.0, 0.0,
 * 0.0}), GridSpacing({1.0, 1.0, 1.0})); auto decomp = decomposition::create(domain,
 * {4, 2, 1});
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
 * auto domain = domain::create(GridSize({128, 128, 128}), PhysicalOrigin({0.0, 0.0,
 * 0.0}), GridSpacing({1.0, 1.0, 1.0})); auto decomp = decomposition::create(domain,
 * {2, 2, 1});
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
 * auto domain = domain::create(GridSize({256, 256, 256}), PhysicalOrigin({0.0, 0.0,
 * 0.0}), GridSpacing({1.0, 1.0, 1.0})); auto decomp = decomposition::create(domain,
 * mpi_size);
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
// Rank-local and global index boxes derived from the stored Domain.
// ---------------------------------------------------------------------------

/// The global coordinate system this decomposition partitions, as a `Domain`.
[[nodiscard]] inline Domain domain(const Decomposition &decomposition) {
  return decomposition.m_domain;
}

/// The global index box `[lower, upper]` covered by the whole decomposition.
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
/// @throws std::out_of_range if `i` is not a valid subdomain index.
[[nodiscard]] inline Box3i local_box(const Decomposition &decomposition, int i) {
  return decomposition.m_local_boxes.at(i);
}

} // namespace decomposition

using Decomposition = decomposition::Decomposition;

} // namespace pfc
