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
private:
  const World &m_world;                                            ///< The World object.
  const int m_rank, m_num_domains;                                 ///< Processor ID and total number of processors.
  const int Lx_c, Ly_c, Lz_c;                                      ///< Dimensions of the complex domain.
  heffte::box3d<int> real_indexes;                                 ///< Index ranges for real domain.
  const heffte::box3d<int> complex_indexes;                        ///< Index ranges for complex domain.
  const std::array<int, 3> proc_grid;                              ///< Processor grid dimensions.
  const std::vector<heffte::box3d<int>> real_boxes, complex_boxes; ///< Local domain boxes.

public:
  const heffte::box3d<int> inbox, outbox; ///< Local communication boxes.
  const int r2c_direction = 0;            ///< Real-to-complex symmetry direction.

  /**
   * @brief Construct a new Decomposition object.
   *
   * @param world Reference to the World object.
   * @param id The id (rank) of the current process.
   * @param num_procs The total number of domains.
   *
   * Numbering ranks starts from 0 (MPI convention). For example, if the domain
   * needs to be decomposed into four parts, those would be 0/4, 1/4, 2/4, 3/4
   * and NOT 1/4, 2/4, 3/4, 4/4.
   */
  Decomposition(const World &world, int rank, int num_domains);

  /**
   * @brief Get the size of the inbox.
   *
   * @return Size of the inbox as a container (const std::array<int, 3>&).
   */
  const std::array<int, 3> &get_inbox_size() const;

  /**
   * @brief Get the offset of the inbox (a.k.a lower limit of the box).
   *
   * @return Offset of the inbox as a container (const std::array<int, 3>&).
   */
  const std::array<int, 3> &get_inbox_offset() const;

  /**
   * @brief Get the size of the outbox.
   *
   * @return Size of the outbox as a container (const std::array<int, 3>&).
   */
  const std::array<int, 3> &get_outbox_size() const;

  /**
   * @brief Get the offset of the outbox (a.k.a lower limit of the box).
   *
   * @return Offset of the outbox as a container (const std::array<int, 3>&).
   */
  const std::array<int, 3> &get_outbox_offset() const;

  /**
   * @brief Get the reference to the World object.
   *
   * @return Reference to the World object.
   */
  const World &get_world() const;

  /**
   * @brief Get the rank of the current process.
   *
   * @return The rank of the current process.
   */
  int get_rank() const;

  /**
   * @brief Get the total number of sub-domains.
   *
   * @return int
   */
  int get_num_domains() const;

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
