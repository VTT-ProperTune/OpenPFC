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
namespace decomposition {

using heffte::box3d;
using pfc::types::Bool3;
using pfc::types::Int3;
using pfc::types::Real3;
using Box3D = heffte::box3d<int>; ///< Type alias for 3D integer box.

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
template <typename T> struct Decomposition {

  const pfc::world::World<T> &m_world; ///< The World object.

  // const World<CoordTag>& m_global_world;
  // const std::vector<World<CoordTag>> m_sub_worlds;
  // const std::array<int, 3> m_parts;
  // const std::vector<std::array<int, 3>> m_rank_coords;

  // const std::vector<int> m_rank_to_index;
  // const std::vector<int> m_index_to_rank;

  const heffte::box3d<int> m_inbox, m_outbox; ///< Local communication boxes.
  const int m_r2c_direction = 0; ///< Real-to-complex symmetry direction.

  /**
   * @brief Construct a new Decomposition object.
   *
   * @param world Reference to the World object.
   * @param inbox The local inbox (real space) box.
   * @param outbox The local outbox (complex space) box.
   *
   * Numbering ranks starts from 0 (MPI convention). For example, if the domain
   * needs to be decomposed into four parts, those would be 0/4, 1/4, 2/4, 3/4
   * and NOT 1/4, 2/4, 3/4, 4/4.
   */
  Decomposition(const World &world, const Box3D &inbox, const Box3D &outbox)
      : m_world(world), m_inbox(inbox), m_outbox(outbox) {}

  // template <typename T>
  friend std::ostream &operator<<(std::ostream &os, const Decomposition<T> &d) {
    os << "Decomposition:\n";
    os << "  World: " << d.m_world << "\n";
    os << "  Inbox: " << d.m_inbox << "\n";
    os << "  Outbox: " << d.m_outbox << "\n";
    return os;
  }
};

template <typename T> const World &get_world(const Decomposition<T> &d) noexcept {
  return d.m_world;
}

template <typename T> const Box3D &get_inbox(const Decomposition<T> &d) noexcept {
  return d.m_inbox;
}

} // namespace decomposition

using Decomposition = decomposition::Decomposition<pfc::world::CartesianTag>;
using Box3D = decomposition::Box3D;

} // namespace pfc
