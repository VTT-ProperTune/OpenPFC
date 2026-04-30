// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_halo_exchange.hpp
 * @brief Non-blocking face halo exchange for the padded brick layout.
 *
 * @details
 * Sibling of `pfc::HaloExchanger` (no padding, overwrites the outermost
 * owned cells) and `pfc::SeparatedFaceHaloExchanger` (separate face
 * vectors). `pfc::PaddedHaloExchanger<T>` is the in-place exchanger
 * that targets a `(nx+2hw)*(ny+2hw)*(nz+2hw)` `pfc::field::PaddedBrick<T>`
 * buffer:
 *
 *   - It builds the **padded** face MPI subarrays from
 *     `pfc::halo::create_padded_face_types_6` so each direction's `recv`
 *     subarray writes into the dedicated halo ring rather than over the
 *     outermost owned cells.
 *   - It exposes the same non-blocking
 *     `start_halo_exchange(T*, std::size_t)` /
 *     `finish_halo_exchange()` pair as the existing exchangers, plus a
 *     blocking `exchange_halos(...)` convenience wrapper.
 *
 * Periodic boundaries are handled by the underlying decomposition: in
 * a single-rank run every direction wraps to self, and `MPI_Isend` /
 * `MPI_Irecv` to self complete locally.
 *
 * The exchange is **face-only** — corner halo cells (e.g. `u(-1, -1, k)`)
 * are not filled. That matches the needs of the 7-point Laplacian and
 * any face-only stencil; mixed derivatives that read corners need a
 * 26-neighbor exchanger that this class does not provide.
 *
 * @see pfc::field::PaddedBrick
 * @see pfc::halo::create_padded_face_types_6
 */

#include <array>
#include <cstddef>
#include <mpi.h>
#include <vector>

#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc {

/**
 * @brief In-place non-blocking face halo exchange for a padded brick.
 *
 * Drives a 6-message face exchange (one per direction) on the
 * `(nx+2hw)*(ny+2hw)*(nz+2hw)` buffer that backs a
 * `pfc::field::PaddedBrick<T>`. The recv subarrays write directly into
 * the brick's halo ring, so the user's per-cell stencil can index
 * `u(i +/- hw, j, k)` after `finish_halo_exchange()` returns.
 */
template <typename T = double> class PaddedHaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct the exchanger and pre-build the 6 face MPI types.
   *
   * @param decomp     Decomposition (must outlive this object).
   * @param rank       This MPI rank.
   * @param halo_width Ghost ring thickness `hw`. Must match the brick
   *                   that `start_halo_exchange` is called with.
   * @param comm       MPI communicator.
   * @param base_tag   Base tag for messages (direction index added).
   */
  PaddedHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                      int halo_width, MPI_Comm comm, int base_tag = 0)
      : m_decomp(decomp), m_rank(rank), m_halo_width(halo_width), m_comm(comm),
        m_base_tag(base_tag) {
    const auto &local_world = decomposition::get_subworld(m_decomp, m_rank);
    const auto local_size = world::get_size(local_world);
    const int nx = local_size[0];
    const int ny = local_size[1];
    const int nz = local_size[2];

    m_face_types = halo::create_padded_face_types_6(
        nx, ny, nz, m_halo_width, exchange::detail::get_mpi_type<T>());

    const std::array<Int3, 6> dirs = {
        {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    for (const auto &dir : dirs) {
      m_neighbors.push_back(decomposition::get_neighbor_rank(m_decomp, m_rank, dir));
    }
    m_requests.resize(2 * 6);
  }

  /**
   * @brief Run one halo exchange (post-recv, post-send, wait).
   * @param padded_buf Pointer to the start of the **padded** brick
   *                   (i.e. `brick.data()`). Layout: row-major
   *                   `(nx+2hw, ny+2hw, nz+2hw)`, x fastest.
   * @param padded_size Total elements (`brick.size()`); accepted for
   *                    API symmetry with `pfc::HaloExchanger` but not
   *                    used by the zero-copy face path.
   */
  void exchange_halos(T *padded_buf, std::size_t padded_size) {
    start_halo_exchange(padded_buf, padded_size);
    finish_halo_exchange();
  }

  /**
   * @brief Post `Irecv` then `Isend` for every face direction; return
   *        immediately so the caller can compute the inner region while
   *        the messages are in flight.
   *
   * Pair with `finish_halo_exchange` after the inner work.
   */
  void start_halo_exchange(T *padded_buf, std::size_t padded_size) {
    (void)padded_size;
    m_pending_field = padded_buf;
    void *buf = static_cast<void *>(padded_buf);
    std::size_t req_count = 0;
    for (std::size_t i = 0; i < 6; ++i) {
      const int tag = m_base_tag + opposite_slot(static_cast<int>(i));
      exchange::irecv_face(buf, m_face_types[i].recv_type.get(), m_neighbors[i],
                           m_comm, &m_requests[req_count], tag);
      ++req_count;
    }
    for (std::size_t i = 0; i < 6; ++i) {
      const int tag = m_base_tag + static_cast<int>(i);
      exchange::isend_face(buf, m_face_types[i].send_type.get(), m_neighbors[i],
                           m_comm, &m_requests[req_count], tag);
      ++req_count;
    }
    m_request_count = static_cast<int>(req_count);
  }

  /**
   * @brief Wait on every outstanding request from `start_halo_exchange`.
   *
   * Records the elapsed time into the
   * `kProfilingRegionCommunication` profiling slot, matching the
   * convention used by `pfc::HaloExchanger`.
   */
  void finish_halo_exchange() {
    const double t0 = MPI_Wtime();
    exchange::wait_all(m_requests.data(), m_request_count);
    m_pending_field = nullptr;
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
  }

  /// Number of face directions (always 6 with periodic decompositions).
  std::size_t num_directions() const { return 6; }

private:
  static int opposite_slot(int slot) {
    switch (slot) {
    case 0: return 1;
    case 1: return 0;
    case 2: return 3;
    case 3: return 2;
    case 4: return 5;
    case 5: return 4;
    default: return -1;
    }
  }

  const decomposition::Decomposition &m_decomp;
  int m_rank;
  int m_halo_width;
  MPI_Comm m_comm;
  int m_base_tag;

  std::array<halo::FaceTypes, 6> m_face_types;
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
  T *m_pending_field = nullptr;
};

} // namespace pfc
