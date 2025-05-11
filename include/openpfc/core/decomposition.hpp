// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

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

inline const auto &get_global_world(const Decomposition &decomposition) noexcept {
  return decomposition.m_global_world;
}

inline const auto &get_world(const Decomposition &decomposition) noexcept {
  return get_global_world(decomposition);
}

inline const auto &get_grid(const Decomposition &decomposition) noexcept {
  return decomposition.m_grid;
}

inline const auto &get_subworlds(const Decomposition &decomposition) noexcept {
  return decomposition.m_subworlds;
}

inline const auto &get_subworld(const Decomposition &decomposition, int i) {
  return get_subworlds(decomposition).at(i);
}

inline const auto create(const World &world, const Int3 &grid) noexcept {
  return Decomposition(world, grid);
};

inline const auto create(const World &world, const int &nparts) noexcept {
  auto indices = to_indices(world);
  auto grid = heffte::proc_setup_min_surface(indices, nparts);
  return create(world, grid);
}

inline const auto get_num_domains(const Decomposition &decomposition) noexcept {
  return get_subworlds(decomposition).size();
}

} // namespace decomposition

using Decomposition = decomposition::Decomposition;

} // namespace pfc
