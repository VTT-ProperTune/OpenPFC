// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#ifndef PFC_DECOMPOSITION_HPP
#define PFC_DECOMPOSITION_HPP

#include "openpfc/core/world.hpp"
#include <array>
#include <heffte.h>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace pfc {

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
class Decomposition {
public:
  // Type aliases for clarity
  using Int3 = std::array<int, 3>;  ///< Type alias for 3D integer array.
  using Box3D = heffte::box3d<int>; ///< Type alias for 3D integer box.

private:
  const World &m_world;          ///< The World object.
  const Box3D m_inbox, m_outbox; ///< Local communication boxes.
  const int m_r2c_direction = 0; ///< Real-to-complex symmetry direction.

public:
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
  Decomposition(const World &world, const Box3D &inbox, const Box3D &outbox);

  /**
   * @brief Get the inbox box.
   *
   * @return const Box3D& The inbox box.
   */
  const Box3D &get_inbox() const noexcept;

  /**
   * @brief Get the outbox box.
   *
   * @return const Box3D& The outbox box.
   */
  const Box3D &get_outbox() const noexcept;

  /**
   * @brief Get the size of the inbox.
   *
   * @return Size of the inbox as a container (const std::array<int, 3>&).
   */
  const Int3 &get_inbox_size() const noexcept;

  /**
   * @brief Get the offset of the inbox (a.k.a lower limit of the box).
   *
   * @return Offset of the inbox as a container (const std::array<int, 3>&).
   */
  const Int3 &get_inbox_offset() const noexcept;

  /**
   * @brief Get the size of the outbox.
   *
   * @return Size of the outbox as a container (const std::array<int, 3>&).
   */
  const Int3 &get_outbox_size() const noexcept;

  /**
   * @brief Get the offset of the outbox (a.k.a lower limit of the box).
   *
   * @return Offset of the outbox as a container (const std::array<int, 3>&).
   */
  const Int3 &get_outbox_offset() const noexcept;

  /**
   * @brief Get the reference to the World object.
   *
   * @return Reference to the World object.
   */
  const World &get_world() const noexcept;

  /**
   * @brief Get the rank of the current process.
   *
   * @return The rank of the current process.
   */
  // int get_rank() const;

  /**
   * @brief Get the total number of sub-domains.
   *
   * @return int
   */
  // int get_num_domains() const;

  /**
   * @brief Output stream operator for Decomposition objects.
   *
   * Allows printing the state of a Decomposition object to an output stream.
   *
   * @param os The output stream to write to.
   * @param d The Decomposition object to be printed.
   * @return The updated output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const Decomposition &d);
};

} // namespace pfc

#endif // PFC_DECOMPOSITION_HPP
