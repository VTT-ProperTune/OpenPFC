// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_halo_exchange.hpp
 * @brief Internal Faces backend for `pfc::comm::HaloExchange` (host).
 *
 * @details
 * Not a public API. Callers bind a padded `pfc::data::Field` through
 * `pfc::comm::HaloExchange` with `HaloConnectivity::Faces`. This header
 * owns the 6-face MPI subarray path (`create_padded_face_types_6`).
 *
 * Face-only: corners/edges are not filled. Full 26-direction fill is
 * `HaloConnectivity::Full`.
 *
 * @see comm_halo_exchange.hpp
 * @see pfc::halo::create_padded_face_types_6
 */

#include <array>
#include <cstddef>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_direction_agreement.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc::comm::detail {

/**
 * @brief In-place non-blocking face halo exchange for a padded brick.
 *
 * Drives a 6-message face exchange (one per direction) on the
 * `(nx+2hw)*(ny+2hw)*(nz+2hw)` buffer that backs a padded
 * `pfc::data::Field<T, HostSpace>`. The recv subarrays write directly into
 * the field's halo ring, so the user's per-cell stencil can index
 * `u(i +/- hw, j, k)` after `finish_halo_exchange()` returns.
 */
template <typename T = double> class HostFacesHalo {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct the exchanger with explicit Box3i subdomain bounds and Domain
   * reference. This is the preferred interface for M1+ code.
   *
   * @param subdomain_box Local subdomain box (bounds for this rank).
   * @param domain        Global domain (for periodicity/spacing metadata).
   * @param decomp        Decomposition for neighbor calculation (must outlive this
   * object).
   * @param rank          This MPI rank.
   * @param halo_width    Ghost ring thickness `hw`. Must match the brick
   *                      that `start_halo_exchange` is called with.
   * @param comm          MPI communicator.
   * @param base_tag      Base tag for messages (direction index added).
   */
  HostFacesHalo(const Box3i &subdomain_box, const Domain &domain,
                const decomposition::Decomposition &decomp, int rank, int halo_width,
                MPI_Comm comm, int base_tag = 0)
      : HostFacesHalo(subdomain_box, domain, decomp, rank, halo_width, comm,
                      halo::presets::Axes3D(), base_tag,
                      halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct the exchanger and pre-build the 6 face MPI types
   *        (default: full `Axes3D()` set, identical to the historical 6-face
   *        exchange).
   *
   * @param decomp     Decomposition (must outlive this object).
   * @param rank       This MPI rank.
   * @param halo_width Ghost ring thickness `hw`. Must match the brick
   *                   that `start_halo_exchange` is called with.
   * @param comm       MPI communicator.
   * @param base_tag   Base tag for messages (direction index added).
   *
   * @deprecated Use explicit Box3i + Domain constructor instead.
   */
  [[deprecated("Use explicit Box3i + Domain constructor: HostFacesHalo(box, "
               "domain, decomp, rank, ...)")]]
  HostFacesHalo(const decomposition::Decomposition &decomp, int rank, int halo_width,
                MPI_Comm comm, int base_tag = 0)
      : HostFacesHalo(decomposition::local_box(decomp, rank),
                      decomposition::domain(decomp), decomp, rank, halo_width, comm,
                      halo::presets::Axes3D(), base_tag,
                      halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct with a user-selected halo direction set.
   *
   * Restricts the active face slots to those listed in `dirs`. Non-face
   * directions (edges, corners) are tolerated but ignored — this exchanger
   * is face-only. For full 26-direction fills use
   * `pfc::comm::HaloExchange` with `HaloConnectivity::Full`.
   *
   * If `selector` is provided the active set for this rank is
   * `selector(rank)`; otherwise the uniform `dirs` is used.
   *
   * @param decomp     Decomposition (must outlive this object).
   * @param rank       This MPI rank.
   * @param halo_width Ghost ring thickness `hw`.
   * @param comm       MPI communicator.
   * @param dirs       Direction set (defaults to `Axes3D()` for back-compat).
   * @param base_tag   Base tag for messages (direction index added).
   * @param selector   Optional per-rank override of the direction set.
   *
   * @deprecated Use explicit Box3i + Domain constructor instead.
   */
  [[deprecated("Use explicit Box3i + Domain constructor: HostFacesHalo(box, "
               "domain, decomp, rank, ...)")]]
  HostFacesHalo(const decomposition::Decomposition &decomp, int rank, int halo_width,
                MPI_Comm comm, halo::HaloDirectionSet dirs, int base_tag = 0,
                halo::HaloDirectionSelector selector = {})
      : HostFacesHalo(decomposition::local_box(decomp, rank),
                      decomposition::domain(decomp), decomp, rank, halo_width, comm,
                      dirs, base_tag, selector) {}

  // Box3i + Domain constructor implementation
  HostFacesHalo(const Box3i &subdomain_box, const Domain &domain,
                const decomposition::Decomposition &decomp, int rank, int halo_width,
                MPI_Comm comm, halo::HaloDirectionSet dirs, int base_tag = 0,
                halo::HaloDirectionSelector selector = {})
      : m_subdomain_box(subdomain_box), m_domain(domain), m_decomp(decomp),
        m_rank(rank), m_halo_width(halo_width), m_comm(comm), m_base_tag(base_tag),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)),
        m_use_decomp(false) {
    if (halo::neighbour_agreement_enabled()) {
      halo::validate_neighbour_direction_agreement(comm, decomp, rank, m_dirs);
    }

    // Extract local size from explicit Box3i bounds
    const int nx = m_subdomain_box.size[0];
    const int ny = m_subdomain_box.size[1];
    const int nz = m_subdomain_box.size[2];

    m_face_types = halo::create_padded_face_types_6(
        nx, ny, nz, m_halo_width, exchange::detail::get_mpi_type<T>());

    // Compute neighbors from decomposition
    const std::array<Int3, 6> dirs_canon = {
        {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}};
    for (std::size_t i = 0; i < 6; ++i) {
      m_active[i] = m_dirs.contains(dirs_canon[i]);
      m_neighbors.push_back(
          decomposition::get_neighbor_rank(m_decomp, m_rank, dirs_canon[i]));
    }
    m_requests.resize(2 * 6);
  }

  /**
   * @brief Bind a padded `pfc::data::Field<T, HostSpace>` (storage_halo > 0).
   *
   * Geometry comes from `u.box()` / `u.domain()`; neighbours still need the
   * live `decomp` (Field does not carry a Decomposition). Captures `u.data()`
   * for the no-arg `start()` / `finish()` / `exchange` helpers.
   *
   * @throws std::invalid_argument if `u.storage_halo() <= 0` (unpadded Fields
   *         use `SparseHaloExchanger` / face buffers, not this layout).
   */
  HostFacesHalo(data::Field<T, HostSpace> &u,
                const decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
                int base_tag = 0)
      : HostFacesHalo(u.box(), u.domain(), decomp, rank, u.storage_halo(), comm,
                      halo::presets::Axes3D(), base_tag,
                      halo::HaloDirectionSelector{}) {
    bind_field_(u);
  }

  /// Same as the Field-binding constructor, with a custom direction set.
  HostFacesHalo(data::Field<T, HostSpace> &u,
                const decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
                halo::HaloDirectionSet dirs, int base_tag = 0,
                halo::HaloDirectionSelector selector = {})
      : HostFacesHalo(u.box(), u.domain(), decomp, rank, u.storage_halo(), comm,
                      dirs, base_tag, selector) {
    bind_field_(u);
  }

  /**
   * @brief Run one halo exchange (post-recv, post-send, wait).
   * @param padded_buf Pointer to the start of the **padded** brick
   *                   (i.e. `brick.data()`). Layout: row-major
   *                   `(nx+2hw, ny+2hw, nz+2hw)`, x fastest.
   * @param padded_size Total elements (`brick.size()`); accepted for
   *                    API symmetry but not used by the zero-copy face path.
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
      if (!m_active[i]) {
        continue;
      }
      const int tag = m_base_tag + halo::opposite_slot(static_cast<int>(i));
      exchange::irecv_face(buf, m_face_types[i].recv_type.get(), m_neighbors[i],
                           m_comm, &m_requests[req_count], tag);
      ++req_count;
    }
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i]) {
        continue;
      }
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
   * `kProfilingRegionCommunication` profiling slot.
   */
  void finish_halo_exchange() {
    const double t0 = MPI_Wtime();
    exchange::wait_all(m_requests.data(), m_request_count);
    m_pending_field = nullptr;
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
  }

  // ---- Field-bound API ---------------------------------------------------

  /**
   * @brief Post the asynchronous exchange on the bound field buffer.
   *
   * Equivalent to `start_halo_exchange(field.data(), field.size())` but
   * with no chance of passing a mismatched buffer or stale halo width.
   * Requires that the exchanger was constructed from a padded
   * `pfc::data::Field`.
   */
  void start() {
    require_bound_("start()");
    start_halo_exchange(m_bound_buf, m_bound_size);
  }

  /// Wait on the in-flight exchange started by `start()`.
  void finish() {
    require_bound_("finish()");
    finish_halo_exchange();
  }

  /// `true` once the exchanger has captured a Field buffer.
  [[nodiscard]] bool is_bound() const noexcept { return m_bound_buf != nullptr; }

  /// Number of active face directions (`0..6` depending on the direction set).
  std::size_t num_directions() const {
    std::size_t n = 0;
    for (bool a : m_active) {
      if (a) ++n;
    }
    return n;
  }

  /// Read-only access to the active direction set.
  [[nodiscard]] const halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

private:
  void bind_field_(data::Field<T, HostSpace> &u) {
    if (u.storage_halo() <= 0) {
      throw std::invalid_argument(
          "pfc::comm::detail::HostFacesHalo: Field binding requires "
          "storage_halo > 0 (padded layout). Unpadded Fields use "
          "SparseExchange / face buffers.");
    }
    m_bound_buf = u.data();
    m_bound_size = u.size();
  }

  void require_bound_(const char *what) const {
    if (m_bound_buf == nullptr) {
      throw std::logic_error(std::string("pfc::comm::detail::HostFacesHalo::") +
                             what +
                             ": exchanger is not bound to a padded Field. "
                             "Use a Field-binding constructor or call "
                             "start_halo_exchange(buf, size) directly.");
    }
  }

  // Box3i + Domain interface (preferred for M1+)
  Box3i m_subdomain_box;
  Domain m_domain;

  // Decomposition-based interface (deprecated)
  const decomposition::Decomposition &m_decomp;
  bool m_use_decomp =
      false; // True when constructed via deprecated Decomposition path

  int m_rank;
  int m_halo_width;
  MPI_Comm m_comm;
  int m_base_tag;
  halo::HaloDirectionSet m_dirs;

  std::array<halo::FaceTypes, 6> m_face_types;
  std::array<bool, 6> m_active{};
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
  T *m_pending_field = nullptr;

  // Optional Field binding (set by the Field-binding constructors).
  T *m_bound_buf = nullptr;
  std::size_t m_bound_size = 0;
};

/**
 * @name Free helpers for `HostFacesHalo`
 *
 * `exchange(halo)` runs a full non-blocking exchange (start then finish)
 * with no overlap — the usual choice in compact drivers.
 *
 * `start_exchange` / `finish_exchange` split the pair so inner work can
 * run while messages are in flight (same shape as `start_halo_exchange` /
 * `finish_halo_exchange` on the raw buffer API).
 *
 * The exchanger must be bound to a padded
 * `pfc::data::Field<T, HostSpace>` (see the Field-binding constructors).
 * @{
 */
template <typename T> inline void start_exchange(HostFacesHalo<T> &h) { h.start(); }
template <typename T> inline void finish_exchange(HostFacesHalo<T> &h) {
  h.finish();
}
template <typename T> inline void exchange(HostFacesHalo<T> &h) {
  start_exchange(h);
  finish_exchange(h);
}
/// @}

} // namespace pfc::comm::detail
