// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file sparse_halo_exchange.hpp
 * @brief Fully sparse, grid-agnostic halo exchanger.
 *
 * @details
 * `pfc::SparseHaloExchanger<T>` is the **flexible** half of OpenPFC's halo
 * story. It complements `pfc::PaddedHaloExchanger<T>` (the **classical**
 * padded-brick exchanger that bakes the halo width into the field layout):
 *
 *   - Use `pfc::field::PaddedBrick<T>` + `pfc::PaddedHaloExchanger<T>` (or
 *     the CUDA `pfc::cuda::PaddedDeviceHaloExchanger`) for **pure FD** with
 *     a single contiguous `(nx+2hw)*(ny+2hw)*(nz+2hw)` array and negative
 *     halo indexing. This is the **default** for FD-only apps.
 *   - Use `pfc::field::LocalField<T>` (unpadded `nx*ny*nz`) +
 *     `pfc::SparseHaloExchanger<T>` for **mixed FD + spectral** (the FFT
 *     does not tolerate halo regions in the data block), **non-axis halos**,
 *     **arbitrary peer/index patterns**, and as the foundation for future
 *     **unstructured / FEM** halo communication.
 *
 * The exchanger has **no notion** of a grid, of axis-aligned faces, or of
 * a `HaloDirectionSet`. It accepts a `std::vector<halo::RemoteHalo<T>>`
 * where each entry describes **one** logical exchange with **one** peer:
 *
 *   - `peer_rank` — the MPI rank to talk to.
 *   - `send_tag` / `recv_tag` — symmetric tag scheme (this rank's `send_tag`
 *     must match the peer's `recv_tag` for the corresponding entry on the
 *     peer side, and vice versa).
 *   - `send_values` — a `core::SparseVector<CpuTag, T>` whose **indices**
 *     are positions in the **local field** to gather from, and whose
 *     **data** buffer is a scratch send buffer of the same length.
 *   - `recv_values` — a `core::SparseVector<CpuTag, T>` whose **data**
 *     buffer is the receive buffer; if `scatter_after_recv` is `true`,
 *     the exchanger calls `core::scatter` to write the received values
 *     back into the local field at `recv_values.indices()` after the
 *     wait. If `false`, the user reads `recv_values.data()` directly.
 *
 * For the standard structured-grid face exchange (today's six-face
 * scheme, or any subset of the 26 axis/edge/corner directions) use
 * `pfc::halo::make_structured_halos<T>(...)` to build the `RemoteHalo`
 * list from a `Decomposition + HaloDirectionSet`. That helper preserves
 * the canonical opposite-slot tag convention so a structured exchange
 * over `Axes3D()` is wire-compatible with the legacy face exchanger.
 *
 * @see kernel/decomposition/halo_face_layout.hpp for `make_structured_halos`,
 *      `copy_to_face_layout`, `allocate_face_halos`.
 * @see kernel/field/padded_brick.hpp + kernel/decomposition/padded_halo_exchange.hpp
 *      for the padded-brick (classical FD) alternative.
 * @see docs/concepts/halo_exchange.md for "which exchanger when".
 */

#include <cstddef>
#include <mpi.h>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/sparse_vector.hpp>
#include <openpfc/kernel/decomposition/sparse_vector_ops.hpp>
#include <openpfc/kernel/execution/backend_tags.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc::halo {

/**
 * @brief One logical halo exchange with one peer rank.
 *
 * @details
 * Carries everything `pfc::SparseHaloExchanger<T>` needs to drive a
 * non-blocking `MPI_Isend` / `MPI_Irecv` pair against `peer_rank`:
 *
 *   - `send_values.indices()` are positions in the local field that this
 *     rank will gather from (**send side**).
 *   - `send_values.data()` is the scratch send buffer (allocated by the
 *     SparseVector to match `send_values.size()`).
 *   - `recv_values.data()` is the receive buffer (sized at construction
 *     to match the number of incoming elements).
 *   - If `scatter_after_recv` is `true`, `recv_values.indices()` are the
 *     positions in the local field where the received values are written
 *     after the wait completes. If `false` (default), the caller reads
 *     `recv_values.data()` directly and `recv_values.indices()` is
 *     unused (it can be sized to match the data buffer or left empty;
 *     the exchanger ignores it).
 *
 * Tags are user-controlled so the symmetric matching can be enforced by
 * whoever builds the list (e.g. `make_structured_halos`). For face halos
 * the canonical pattern is `send_tag = base + slot`, `recv_tag = base +
 * opposite_slot(slot)`.
 *
 * @tparam T Element type of the exchanged field (e.g. `double`).
 */
template <typename T> struct RemoteHalo {
  /// Peer rank to exchange with. Self-rank entries are allowed (periodic
  /// wrap on a single-rank decomposition); MPI handles them locally.
  int peer_rank{-1};

  /// Tag used by this rank when posting `MPI_Isend` for this entry.
  int send_tag{0};

  /// Tag used by this rank when posting `MPI_Irecv` for this entry.
  int recv_tag{0};

  /// Indices into the local field to gather from + scratch send buffer.
  /// Default-initialised to a zero-size SparseVector so `RemoteHalo<T>`
  /// is default-constructible (helpers like `make_structured_halos` rely
  /// on this); users assign a sized SparseVector before posting an
  /// exchange.
  core::SparseVector<backend::CpuTag, T> send_values{static_cast<std::size_t>(0)};

  /// Receive buffer (`data()`) and optional scatter destinations
  /// (`indices()`) into the local field.
  core::SparseVector<backend::CpuTag, T> recv_values{static_cast<std::size_t>(0)};

  /// If `true`, the exchanger calls `core::scatter(recv_values, field, …)`
  /// after `wait_all`. If `false`, the caller reads `recv_values.data()`
  /// directly (typical for `LocalField` + face-buffer Laplacian patterns).
  bool scatter_after_recv{false};

  /// **Optional, purely informational** direction hint, set by
  /// `pfc::halo::make_structured_halos` so adapters such as
  /// `pfc::halo::copy_to_face_layout` can recover which face / edge / corner
  /// this entry corresponds to. The exchanger itself **ignores** this
  /// field — it has no grid semantics. Default `{0,0,0}` means "no
  /// associated direction" (typical for fully-custom user-built lists,
  /// e.g. FEM neighbours).
  pfc::types::Int3 direction{0, 0, 0};
};

} // namespace pfc::halo

namespace pfc {

/**
 * @brief Fully sparse, grid-agnostic non-blocking halo exchanger.
 *
 * @details
 * Drives one round of `MPI_Isend` / `MPI_Irecv` per `halo::RemoteHalo<T>`
 * entry; no knowledge of faces, axes, halo widths, or any grid concept.
 * The non-blocking pattern (post all `Irecv`, then all `Isend`, then
 * `Waitall`) follows OpenPFC's existing face exchangers.
 *
 * Single-rank periodic wrap (every entry's `peer_rank == m_rank`) works
 * because `MPI_Isend` / `MPI_Irecv` to self are valid; the wait
 * resolves locally.
 *
 * @tparam T Field element type. CPU backend only in this revision; CUDA /
 *           HIP backends can be added later by templating on a
 *           `BackendTag` once the matching `exchange::*` device overloads
 *           land.
 */
template <typename T = double> class SparseHaloExchanger {
public:
  using halo_type = halo::RemoteHalo<T>;

  /**
   * @brief Construct from a list of `RemoteHalo` entries.
   *
   * The exchanger takes ownership of the entries (move) and pre-sizes
   * the request array to `2 * halos.size()`. No MPI traffic happens
   * during construction.
   *
   * @param comm  MPI communicator used for every send/recv.
   * @param rank  This MPI rank (must equal `MPI_Comm_rank(comm, …)`).
   * @param halos User-supplied list of `RemoteHalo` entries; may be
   *              empty (no-op exchange) and may include self-rank
   *              entries (periodic wrap on a single-rank decomposition).
   */
  SparseHaloExchanger(MPI_Comm comm, int rank, std::vector<halo_type> halos)
      : m_comm(comm), m_rank(rank), m_halos(std::move(halos)),
        m_requests(2 * m_halos.size(), MPI_REQUEST_NULL) {}

  /**
   * @brief Run one halo exchange (gather → MPI → optional scatter).
   *
   * Equivalent to `start_halo_exchange(...)` followed immediately by
   * `finish_halo_exchange()`; convenience for callers that don't
   * overlap halo wait time with computation.
   *
   * @param field_ptr  Local field pointer (row-major; layout opaque to
   *                   the exchanger — only the indices matter).
   * @param field_size Total number of accessible elements at
   *                   `field_ptr`; used for index bounds checks in
   *                   `core::gather` / `core::scatter`.
   */
  void exchange_halos(T *field_ptr, std::size_t field_size) {
    start_halo_exchange(field_ptr, field_size);
    finish_halo_exchange();
  }

  /**
   * @brief Post all `Irecv`, then all `Isend`; return immediately for
   *        compute/communication overlap.
   *
   * Pair with `finish_halo_exchange()` after the inner work. The data
   * pointer must remain valid until the matching `finish_halo_exchange`
   * returns; the exchanger does not copy the field.
   */
  void start_halo_exchange(T *field_ptr, std::size_t field_size) {
    m_pending_field = field_ptr;
    m_pending_size = field_size;
    const std::size_t n = m_halos.size();
    if (n == 0) {
      m_request_count = 0;
      return;
    }

    // 1. Gather every send index list into the per-entry scratch send
    //    buffer. After this the SparseVector send_values.data() holds
    //    the contiguous bytes that MPI_Isend will ship.
    for (std::size_t i = 0; i < n; ++i) {
      core::gather(m_halos[i].send_values, field_ptr, field_size);
    }

    // 2. Post all Irecvs first (deadlock-free pattern).
    std::size_t req_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
      exchange::irecv_data(m_halos[i].recv_values, m_halos[i].peer_rank, m_rank,
                           m_comm, &m_requests[req_count], m_halos[i].recv_tag);
      ++req_count;
    }

    // 3. Then post all Isends.
    for (std::size_t i = 0; i < n; ++i) {
      exchange::isend_data(m_halos[i].send_values, m_rank, m_halos[i].peer_rank,
                           m_comm, &m_requests[req_count], m_halos[i].send_tag);
      ++req_count;
    }
    m_request_count = static_cast<int>(req_count);
  }

  /**
   * @brief Wait for all outstanding requests; optionally scatter.
   *
   * After the wait, every entry whose `scatter_after_recv` is `true`
   * has its `recv_values` written back into the local field at
   * `recv_values.indices()`. Entries with `scatter_after_recv == false`
   * leave their received values in `recv_values.data()` for the caller
   * to consume directly.
   *
   * Records elapsed wait+scatter time in
   * `profiling::kProfilingRegionCommunication` (matching the convention
   * of the padded and persistent face exchangers).
   */
  void finish_halo_exchange() {
    const double t0 = MPI_Wtime();
    exchange::wait_all(m_requests.data(), m_request_count);
    if (m_pending_field != nullptr) {
      for (auto &h : m_halos) {
        if (h.scatter_after_recv && !h.recv_values.empty()) {
          core::scatter(h.recv_values, m_pending_field, m_pending_size);
        }
      }
    }
    m_pending_field = nullptr;
    m_pending_size = 0;
    m_request_count = 0;
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
  }

  /// Number of `RemoteHalo` entries (one per logical peer exchange).
  [[nodiscard]] std::size_t num_halos() const noexcept { return m_halos.size(); }

  /// Read-only access to the underlying `RemoteHalo` entries (e.g. so a
  /// kernel can read `recv_values.data()` directly without a scatter).
  [[nodiscard]] const std::vector<halo_type> &halos() const noexcept {
    return m_halos;
  }

  /// MPI rank this exchanger was constructed for.
  [[nodiscard]] int rank() const noexcept { return m_rank; }

private:
  MPI_Comm m_comm;
  int m_rank;
  std::vector<halo_type> m_halos;
  std::vector<MPI_Request> m_requests;
  int m_request_count{0};
  T *m_pending_field{nullptr};
  std::size_t m_pending_size{0};
};

} // namespace pfc
