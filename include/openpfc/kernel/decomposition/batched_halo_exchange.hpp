// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file batched_halo_exchange.hpp
 * @brief Library-level multi-field halo exchange for CPU backends.
 *
 * @details
 * `pfc::communication::BatchedHaloExchange<T>` enables **batching multiple field
 * exchanges in a single MPI round**, reducing communication overhead for
 * multi-field physics simulations. This class extracts the proven patterns
 * from the production Kobayashi application and makes them available as a
 * general-purpose library component.
 *
 * **Performance Benefits:**
 * - Single MPI synchronization point instead of one per field
 * - Reduced `MPI_Waitall` overhead (1 call vs. N calls)
 * - Optimized for latency-bound, multi-field workloads
 *
 * **Tag Layout:**
 * Uses deterministic tag allocation: `base_tag + field_idx * 6 + face_slot`
 * where `face_slot` follows the canonical +X,-X,+Y,-Y,+Z,-Z convention.
 * This ensures no tag collisions across multi-field exchanges.
 *
 * **API Pattern:**
 * ```cpp
 * pfc::communication::BatchedHaloExchange<double> halo(
 *     domain, decomp, rank, halo_width, comm, n_fields, base_tag);
 *
 * std::vector<double*> fields = {field1.data(), field2.data(), field3.data()};
 * halo.exchange_halos(fields);
 * ```
 *
 * **Technical Details:**
 * - All fields must share the same decomposition, halo width, and communicator
 * - Fields are processed in a single non-blocking MPI round (Irecv → Isend →
 * Waitall)
 * - Supports direction sets (e.g., `Axes2D()` for 2D simulations)
 * - Compatible with existing `PaddedHaloExchanger` infrastructure
 *
 * @see kobayashi_batched_halo.hpp for inspiration and GPU-specific variant
 * @see padded_halo_exchange.hpp for single-field halo exchange
 * @see halo_geometry.hpp for direction set definitions
 */

#include <array>
#include <mpi.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc::communication {

/**
 * @brief Multi-field halo exchanger for CPU backends.
 *
 * Batches multiple field exchanges into a single MPI round for reduced
 * communication overhead. All fields must share the same geometric
 * properties (decomposition, halo width) and MPI communicator.
 *
 * @tparam T Element type for all fields (e.g., `double`).
 */
template <typename T = double> class BatchedHaloExchange {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct with explicit Box3i subdomain bounds and Domain reference.
   *        This is the preferred interface for M1+ code.
   *
   * @param subdomain_box Local subdomain box (bounds for this rank).
   * @param domain        Global domain (for periodicity/spacing metadata).
   * @param decomp        Decomposition for neighbor calculation.
   * @param rank          This MPI rank.
   * @param halo_width    Number of halo layers (e.g. 1 for 3-point stencil).
   * @param comm          MPI communicator.
   * @param n_fields      Number of fields to batch (must match exchange calls).
   * @param base_tag      Base tag for messages (field and face indexes added).
   * @param dirs          Direction set (defaults to `Axes3D()` for back-compat).
   * @param selector      Optional per-rank override of the direction set.
   *
   * @throws std::invalid_argument if `n_fields == 0`.
   */
  BatchedHaloExchange(const Box3i &subdomain_box, const Domain &domain,
                      const decomposition::Decomposition &decomp, int rank,
                      int halo_width, MPI_Comm comm, std::size_t n_fields,
                      int base_tag = 0,
                      halo::HaloDirectionSet dirs = halo::presets::Axes3D(),
                      halo::HaloDirectionSelector selector = {})
      : m_subdomain_box(subdomain_box), m_domain(domain), m_decomp(decomp),
        m_rank(rank), m_halo_width(halo_width), m_comm(comm), m_base_tag(base_tag),
        m_n_fields(n_fields),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {

    if (n_fields == 0) {
      throw std::invalid_argument("BatchedHaloExchange: n_fields must be > 0");
    }

    halo::validate_neighbour_direction_agreement(comm, decomp, rank, m_dirs);

    auto local_size = m_subdomain_box.size;
    int nx = local_size[0];
    int ny = local_size[1];
    int nz = local_size[2];

    // Create MPI derived types for six faces (shared across all fields)
    m_face_types = halo::create_padded_face_types_6(
        nx, ny, nz, m_halo_width, exchange::detail::get_mpi_type<T>());

    // Build neighbor table and active direction set
    const std::array<Int3, 6> direction_order = {{Int3{1, 0, 0}, Int3{-1, 0, 0},
                                                  Int3{0, 1, 0}, Int3{0, -1, 0},
                                                  Int3{0, 0, 1}, Int3{0, 0, -1}}};

    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_dirs.contains(direction_order[i])) {
        continue; // Excluded by direction set
      }

      const auto dir = direction_order[i];
      int neighbor = decomposition::get_neighbor_rank(m_decomp, m_rank, dir);
      if (neighbor >= 0) {
        m_active_faces.push_back(static_cast<int>(i));
        m_neighbors.push_back(neighbor);
      }
    }

    // Allocate request array: 2 requests per (field, face) for Irecv/Isend
    const std::size_t total_requests = 2 * m_active_faces.size() * n_fields;
    m_requests.resize(total_requests, MPI_REQUEST_NULL);
  }

  /**
   * @brief Construct using legacy Decomposition-based interface.
   *
   * @param decomp     Decomposition for neighbor calculation.
   * @param rank       This MPI rank.
   * @param halo_width Number of halo layers.
   * @param comm       MPI communicator.
   * @param n_fields   Number of fields to batch.
   * @param base_tag   Base tag for messages.
   * @param dirs       Direction set (defaults to `Axes3D()`).
   * @param selector   Optional per-rank override of the direction set.
   *
   * @deprecated Use explicit Box3i + Domain constructor.
   */
  [[deprecated("Use explicit Box3i + Domain constructor")]]
  BatchedHaloExchange(const decomposition::Decomposition &decomp, int rank,
                      int halo_width, MPI_Comm comm, std::size_t n_fields,
                      int base_tag = 0,
                      halo::HaloDirectionSet dirs = halo::presets::Axes3D(),
                      halo::HaloDirectionSelector selector = {})
      : BatchedHaloExchange(decomposition::local_box(decomp, rank),
                            decomposition::domain(decomp), decomp, rank, halo_width,
                            comm, n_fields, base_tag, dirs, selector) {}

  BatchedHaloExchange(const BatchedHaloExchange &) = delete;
  BatchedHaloExchange &operator=(const BatchedHaloExchange &) = delete;

  /**
   * @brief Exchange halos for multiple fields in a single MPI round.
   *
   * All fields are processed with one post-receive, post-send, and wait
   * sequence, significantly reducing MPI synchronization overhead.
   *
   * @param fields Vector of field pointers (must have exactly `n_fields()` entries).
   *               Each field should be in padded brick layout matching the
   *               subdomain box used during construction.
   *
   * @throws std::invalid_argument if `fields.size() != n_fields()`.
   *
   * @note Tag allocation: `base_tag + field_idx * 6 + face_slot`
   * @note MPI pattern: Post all Irecvs → Post all Isends → Waitall
   */
  void exchange_halos(std::vector<T *> fields) {
    const double t0 = MPI_Wtime();

    if (fields.size() != m_n_fields) {
      throw std::invalid_argument(
          "BatchedHaloExchange::exchange_halos: field count mismatch (" +
          std::to_string(fields.size()) + " vs. " + std::to_string(m_n_fields) +
          " expected)");
    }

    std::size_t req_count = 0;

    // Phase 1: Post all Irecvs for all (field, face) pairs
    for (std::size_t f = 0; f < m_n_fields; ++f) {
      void *buf = static_cast<void *>(fields[f]);
      const int field_tag_offset = m_base_tag + static_cast<int>(f) * 6;

      for (std::size_t i = 0; i < m_active_faces.size(); ++i) {
        const int face_slot = m_active_faces[i];
        const int neighbor = m_neighbors[i];
        const int opposite_face = opposite_slot(face_slot);
        const int tag = field_tag_offset + opposite_face;

        exchange::irecv_face(buf, m_face_types[face_slot].recv_type.get(), neighbor,
                             m_comm, &m_requests[req_count], tag);
        ++req_count;
      }
    }

    // Phase 2: Post all Isends for all (field, face) pairs
    for (std::size_t f = 0; f < m_n_fields; ++f) {
      void *buf = static_cast<void *>(fields[f]);
      const int field_tag_offset = m_base_tag + static_cast<int>(f) * 6;

      for (std::size_t i = 0; i < m_active_faces.size(); ++i) {
        const int face_slot = m_active_faces[i];
        const int neighbor = m_neighbors[i];
        const int tag = field_tag_offset + face_slot;

        exchange::isend_face(buf, m_face_types[face_slot].send_type.get(), neighbor,
                             m_comm, &m_requests[req_count], tag);
        ++req_count;
      }
    }

    // Phase 3: Wait for all communication to complete
    exchange::wait_all(m_requests.data(), static_cast<int>(req_count));

    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
  }

  /// Get number of fields configured for batching.
  [[nodiscard]] std::size_t n_fields() const noexcept { return m_n_fields; }

  /// Get the active direction set (after selector resolution).
  [[nodiscard]] const halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

  /// Get number of active face directions (0-6).
  [[nodiscard]] std::size_t num_active_faces() const noexcept {
    return m_active_faces.size();
  }

private:
  /// Compute opposite face slot for canonical tag allocation.
  static int opposite_slot(int slot) {
    switch (slot) {
    case 0: return 1; // +X ↔ -X
    case 1: return 0;
    case 2: return 3; // +Y ↔ -Y
    case 3: return 2;
    case 4: return 5; // +Z ↔ -Z
    case 5: return 4;
    default: return -1;
    }
  }

  // Box3i + Domain interface (preferred for M1+)
  Box3i m_subdomain_box;
  Domain m_domain;
  bool m_use_box3i_domain = true;

  const decomposition::Decomposition &m_decomp;
  int m_rank;
  int m_halo_width;
  MPI_Comm m_comm;
  int m_base_tag;
  std::size_t m_n_fields;
  halo::HaloDirectionSet m_dirs;

  // MPI derived types for six faces (shared across all fields)
  std::array<halo::FaceTypes, 6> m_face_types;

  // Active faces and corresponding neighbors
  std::vector<int> m_active_faces;
  std::vector<int> m_neighbors;

  // Request array: 2 * active_faces * n_fields
  std::vector<MPI_Request> m_requests;
};

} // namespace pfc::communication
