// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_halo_exchange.hpp
 * @brief Non-blocking face halo exchange for the padded brick layout.
 *
 * @details
 * Sibling of `pfc::HaloExchanger` (no padding, overwrites the outermost
 * owned cells) and `pfc::SparseHaloExchanger` (sparse, separate face
 * vectors). `pfc::communication::PaddedHaloExchanger<T>` is the in-place
 * exchanger
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
 *   - The **preferred** entry point is the brick-binding constructor
 *     `communication::PaddedHaloExchanger(field::PaddedBrick<T>& u, MPI_Comm comm)`:
 *     it pulls the decomposition / rank / halo width from `u` and
 *     captures the buffer pointer once, so the time loop can read
 *
 *       pfc::communication::PaddedHaloExchanger<double> halo(u, MPI_COMM_WORLD);
 *       pfc::communication::exchange(halo);   // blocking; one call
 *       // Pro overlap: pfc::communication::start_exchange(halo); …
 *       //               pfc::communication::finish_exchange(halo);
 *
 *     with no chance of drift between the brick layout and the
 *     exchanger arguments.
 *
 * Periodic boundaries are handled by the underlying decomposition: in
 * a single-rank run every direction wraps to self, and `MPI_Isend` /
 * `MPI_Irecv` to self complete locally.
 *
 * The exchange is **face-only** — corner halo cells (e.g. `u(-1, -1, k)`)
 * are not filled. That matches the needs of the 7-point Laplacian and
 * any face-only stencil; mixed derivatives that read corners need a
 * 26-neighbor exchanger — use `pfc::communication::FullPaddedHaloExchanger`
 * on the host, or `pfc::cuda::FullPaddedDeviceHalo` on device.
 *
 * @see pfc::field::PaddedBrick
 * @see pfc::halo::create_padded_face_types_6
 * @see full_padded_halo_exchange.hpp — host 26-direction twin
 */

#include <array>
#include <cstddef>
#include <mpi.h>
#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_direction_agreement.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

// `pfc::exchange` is already a namespace (`exchange.hpp`). The blocking
// one-shot helper therefore lives only as `pfc::communication::exchange`
// and is not re-exported into `pfc::`.

namespace pfc::communication {

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
   */
  PaddedHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                      int halo_width, MPI_Comm comm, int base_tag = 0)
      : PaddedHaloExchanger(decomp, rank, halo_width, comm, halo::presets::Axes3D(),
                            base_tag, halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct with a user-selected halo direction set.
   *
   * Restricts the active face slots to those listed in `dirs`. Non-face
   * directions (edges, corners) are tolerated but ignored — this exchanger
   * is face-only. For full 26-direction fills use
   * `pfc::communication::FullPaddedHaloExchanger` (host) or
   * `pfc::cuda::FullPaddedDeviceHalo` (CUDA).
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
   */
  PaddedHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                      int halo_width, MPI_Comm comm, halo::HaloDirectionSet dirs,
                      int base_tag = 0, halo::HaloDirectionSelector selector = {})
      : m_decomp(decomp), m_rank(rank), m_halo_width(halo_width), m_comm(comm),
        m_base_tag(base_tag),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {
    halo::validate_neighbour_direction_agreement(comm, decomp, rank, m_dirs);

    const auto local_size = decomposition::local_box(m_decomp, m_rank).size;
    const int nx = local_size[0];
    const int ny = local_size[1];
    const int nz = local_size[2];

    m_face_types = halo::create_padded_face_types_6(
        nx, ny, nz, m_halo_width, exchange::detail::get_mpi_type<T>());

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
   * @brief Construct directly from a `pfc::field::PaddedBrick<T>`.
   *
   * Pulls decomposition, rank, halo width, **and the buffer pointer**
   * from `u`. After construction the exchanger is "bound" to that brick
   * and you can drive it with the no-arg `start()` / `finish()` member
   * functions or the free `pfc::communication::start_exchange(exchanger)` /
   * `pfc::communication::finish_exchange(exchanger)` wrappers — no need to re-pass
   * the buffer or the halo width and risk drift.
   *
   * The brick must outlive the exchanger; its buffer pointer is captured
   * once at construction and `PaddedBrick` does not reallocate, so this
   * is safe for the typical "construct once, exchange many times" loop.
   *
   * @param u        Padded brick to bind to (decomp/rank/hw read from `u`).
   * @param comm     MPI communicator.
   * @param base_tag Base tag for messages (direction index added).
   */
  PaddedHaloExchanger(field::PaddedBrick<T> &u, MPI_Comm comm, int base_tag = 0)
      : PaddedHaloExchanger(u.decomposition(), u.rank(), u.halo_width(), comm,
                            halo::presets::Axes3D(), base_tag,
                            halo::HaloDirectionSelector{}) {
    bind_(u);
  }

  /// Same as the brick-binding constructor, with a custom direction set.
  PaddedHaloExchanger(field::PaddedBrick<T> &u, MPI_Comm comm,
                      halo::HaloDirectionSet dirs, int base_tag = 0,
                      halo::HaloDirectionSelector selector = {})
      : PaddedHaloExchanger(u.decomposition(), u.rank(), u.halo_width(), comm, dirs,
                            base_tag, selector) {
    bind_(u);
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
      if (!m_active[i]) {
        continue;
      }
      const int tag = m_base_tag + opposite_slot(static_cast<int>(i));
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

  // ---- Brick-bound API ---------------------------------------------------

  /**
   * @brief Post the asynchronous exchange on the bound brick buffer.
   *
   * Equivalent to `start_halo_exchange(brick.data(), brick.size())` but
   * with no chance of passing a mismatched buffer or stale halo width.
   * Requires that the exchanger was constructed from a `PaddedBrick`.
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

  /// `true` once the exchanger has captured a brick buffer.
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
  void bind_(field::PaddedBrick<T> &u) noexcept {
    m_bound_buf = u.data();
    m_bound_size = u.size();
  }

  void require_bound_(const char *what) const {
    if (m_bound_buf == nullptr) {
      throw std::logic_error(
          std::string("pfc::communication::PaddedHaloExchanger::") + what +
          ": exchanger is not bound to a PaddedBrick. "
          "Use the (PaddedBrick&, MPI_Comm) constructor or call "
          "start_halo_exchange(buf, size) directly.");
    }
  }

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
  halo::HaloDirectionSet m_dirs;

  std::array<halo::FaceTypes, 6> m_face_types;
  std::array<bool, 6> m_active{};
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;
  int m_request_count = 0;
  T *m_pending_field = nullptr;

  // Optional brick binding (set by the (PaddedBrick&, MPI_Comm, ...) ctors).
  T *m_bound_buf = nullptr;
  std::size_t m_bound_size = 0;
};

/**
 * @name Free helpers for `PaddedHaloExchanger`
 *
 * `exchange(halo)` runs a full non-blocking exchange (start then finish)
 * with no overlap — the usual choice in compact drivers.
 *
 * `start_exchange` / `finish_exchange` split the pair so inner work can
 * run while messages are in flight (same shape as `start_halo_exchange` /
 * `finish_halo_exchange` on the raw buffer API).
 *
 * The exchanger must be bound to a `pfc::field::PaddedBrick<T>` (see the
 * brick-binding constructors).
 * @{
 */
template <typename T> inline void start_exchange(PaddedHaloExchanger<T> &h) {
  h.start();
}
template <typename T> inline void finish_exchange(PaddedHaloExchanger<T> &h) {
  h.finish();
}
template <typename T> inline void exchange(PaddedHaloExchanger<T> &h) {
  start_exchange(h);
  finish_exchange(h);
}
/// @}

} // namespace pfc::communication

namespace pfc {
using communication::finish_exchange;
using communication::PaddedHaloExchanger;
using communication::start_exchange;
// Note: `communication::exchange` is not brought into `pfc::` — it would
// collide with the existing `pfc::exchange` namespace from `exchange.hpp`.
} // namespace pfc
