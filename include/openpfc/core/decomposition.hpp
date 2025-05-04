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
namespace decomposition {

// Type aliases for clarity
using Box3D = heffte::box3d<int>; ///< Type alias for 3D integer box.

/**
 * @brief Represents a partitioning of the global simulation domain into local
 * subdomains.
 *
 * The Decomposition class describes how the World is split among multiple
 * processes. Each instance provides access to the local subdomain owned by the
 * current process, including size, offset, and basic mappings between global
 * and local indices.
 *
 *
 * ## Responsibilities
 *
 * - Split the World into non-overlapping local boxes.
 * - Provide access to local box information.
 * - Support global-local coordinate mapping.
 *
 *
 * ## Design Justification
 *
 * - Keeps World (global domain) and local process view separate.
 * - Decouples decomposition logic from communication (MPI, FFT backends).
 * - Enables different decomposition strategies if needed (block, slab, pencil).
 *
 *
 * ## Relations to Other Components
 *
 * - Used by Fields and Arrays to allocate local data.
 * - Used by Communication components to know where data lives.
 * - Passed to FFT backends as a description of local domains.
 */

template <typename CoordinateSystemTag> struct Decomposition {

  const pfc::world::World<CoordinateSystemTag> &m_world; ///< The World object.
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
};

template <typename CoordinateSystemTag>
const Box3D &get_inbox(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_inbox;
}

template <typename CoordinateSystemTag>
const Box3D &get_outbox(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_outbox;
}

template <typename CoordinateSystemTag>
const Int3 &get_inbox_size(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_inbox.size;
}

template <typename CoordinateSystemTag>
const Int3 &get_inbox_offset(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_inbox.low;
}

template <typename CoordinateSystemTag>
const Int3 &get_outbox_size(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_outbox.size;
}

template <typename CoordinateSystemTag>
const Int3 &get_outbox_offset(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_outbox.low;
}

template <typename CoordinateSystemTag>
const World &get_world(const Decomposition<CoordinateSystemTag> &d) noexcept {
  return d.m_world;
}

template <typename CoordinateSystemTag>
std::ostream &operator<<(std::ostream &os,
                         const Decomposition<CoordinateSystemTag> &d) {
  os << "Decomposition: " << get_inbox(d) << " " << get_outbox(d);
  return os;
}

} // namespace decomposition

using Decomposition = decomposition::Decomposition<pfc::world::CartesianTag>;
using Box3D = decomposition::Box3D;

} // namespace pfc
