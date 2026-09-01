// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file comm_halo_exchange.hpp
 * @brief Unified host `pfc::comm::HaloExchange` (M4).
 *
 * @details
 * M4 folds the structured halo zoo onto one class: 6-face or 26-direction
 * connectivity, blocking `exchange()`, split `start()`/`finish()`, optional
 * persistent requests, and multi-field tag blocks from `halo_geometry.hpp`.
 *
 * Host Faces MPI (`HostFacesHalo`) lives in this header. Full and
 * persistent still compose `HostFullHalo` / `HostPersistentFaces` until
 * those backends are inlined. Device `HaloExchange<CUDASpace/HIPSpace>`
 * lives in `runtime/gpu/comm_halo_exchange_gpu.hpp`.
 *
 * Tag layout: field `f` uses `halo::field_tag_base(exchange_base, f)` so two
 * exchangers (or six fields) with distinct bases cannot collide.
 *
 * @see halo_geometry.hpp
 * @see full_padded_halo_exchange.hpp
 */

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/full_padded_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/halo_direction_agreement.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/decomposition/halo_persistent.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/kernel/execution/memory_space.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc::comm {

/// Structured halo connectivity: 6 faces, or faces+edges+corners (26).
enum class HaloConnectivity { Faces, Full };

/// Construction knobs for `HaloExchange`.
struct HaloExchangeOptions {
  HaloConnectivity connectivity = HaloConnectivity::Faces;
  bool persistent = false;
  int exchange_base = 0;
  /// Empty: Faces → `Axes3D()`, Full → `Full3D()`. Set `Axes2D()` for 2D slabs.
  halo::HaloDirectionSet directions{};
  /// If set, `selector(rank)` replaces `directions` / the Faces/Full default.
  halo::HaloDirectionSelector selector{};
};

/// Resolve the direction set actually used by a `HaloExchange` construction.
[[nodiscard]] inline halo::HaloDirectionSet
resolved_halo_directions(const HaloExchangeOptions &opt) {
  if (!opt.directions.empty()) {
    return opt.directions;
  }
  return opt.connectivity == HaloConnectivity::Full ? halo::presets::Full3D()
                                                    : halo::presets::Axes3D();
}

namespace detail {

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
        m_use_decomp(false), m_rank(rank), m_halo_width(halo_width), m_comm(comm),
        m_base_tag(base_tag),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {
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
   *         use `SparseExchange` / face buffers, not this layout).
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

  /// Outstanding MPI requests from the last `start()` (for a combined Waitall).
  [[nodiscard]] int outstanding_count() const noexcept { return m_request_count; }
  [[nodiscard]] MPI_Request *outstanding() noexcept { return m_requests.data(); }

  /// Copy back the Waitall result and clear the in-flight buffer pointer.
  void take_waitall_result(const MPI_Request *completed, int n) {
    if (completed != nullptr && n > 0) {
      for (int i = 0; i < n; ++i) {
        m_requests[static_cast<std::size_t>(i)] = completed[i];
      }
    }
    m_pending_field = nullptr;
    m_request_count = 0;
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

/// Wait on every outstanding Faces request in one `MPI_Waitall`.
template <typename Ex>
void wait_concatenated(std::vector<std::unique_ptr<Ex>> &exs) {
  std::vector<MPI_Request> all;
  for (auto &e : exs) {
    const int n = e->outstanding_count();
    if (n > 0) {
      MPI_Request *r = e->outstanding();
      all.insert(all.end(), r, r + n);
    }
  }
  const double t0 = MPI_Wtime();
  exchange::wait_all(all.empty() ? nullptr : all.data(),
                     static_cast<int>(all.size()));
  profiling::record_time(profiling::kProfilingRegionCommunication, MPI_Wtime() - t0);
  std::size_t off = 0;
  for (auto &e : exs) {
    const int n = e->outstanding_count();
    e->take_waitall_result(n > 0 ? all.data() + off : nullptr, n);
    off += static_cast<std::size_t>(n);
  }
}

} // namespace detail

/**
 * @brief Primary template: `HostSpace` is specialized below.
 *
 * Device spaces are specialized in `runtime/gpu/comm_halo_exchange_gpu.hpp`.
 */
template <typename MemorySpace, typename T = double> class HaloExchange {
  static_assert(std::is_same_v<MemorySpace, HostSpace>,
                "pfc::comm::HaloExchange device specializations live in "
                "runtime/gpu/comm_halo_exchange_gpu.hpp");
};

/**
 * @brief Host structured halo exchange over one or more padded Fields.
 */
template <typename T> class HaloExchange<HostSpace, T> {
public:
  using FieldT = data::Field<T, HostSpace>;

  /**
   * @brief Bind a single padded field.
   *
   * @throws std::invalid_argument if the field is unpadded, or if
   *         `persistent` is requested with `HaloConnectivity::Full`.
   */
  HaloExchange(FieldT &field, const decomposition::Decomposition &decomp, int rank,
               MPI_Comm comm, HaloExchangeOptions opt = {})
      : HaloExchange(std::vector<FieldT *>{&field}, decomp, rank, comm, opt) {}

  /**
   * @brief Bind several padded fields; each gets its own tag block.
   *
   * Field `i` uses MPI tags
   * `[field_tag_base(exchange_base, i), field_tag_base(...) + 33)`.
   */
  HaloExchange(std::vector<FieldT *> fields,
               const decomposition::Decomposition &decomp, int rank, MPI_Comm comm,
               HaloExchangeOptions opt = {})
      : m_opt(opt), m_fields(std::move(fields)) {
    if (m_fields.empty()) {
      throw std::invalid_argument(
          "pfc::comm::HaloExchange: at least one field is required");
    }
    if (m_opt.persistent && m_opt.connectivity == HaloConnectivity::Full) {
      throw std::invalid_argument(
          "pfc::comm::HaloExchange: persistent requests are Faces-only "
          "(Full connectivity has no persistent path)");
    }
    m_faces.reserve(m_fields.size());
    m_full.reserve(m_fields.size());
    m_persist.reserve(m_fields.size());
    for (std::size_t i = 0; i < m_fields.size(); ++i) {
      FieldT *f = m_fields[i];
      if (f == nullptr) {
        throw std::invalid_argument(
            "pfc::comm::HaloExchange: field pointer must not be null");
      }
      if (f->storage_halo() <= 0) {
        throw std::invalid_argument(
            "pfc::comm::HaloExchange: Field binding requires storage_halo > 0");
      }
      const int tag0 =
          halo::field_tag_base(m_opt.exchange_base, static_cast<int>(i));
      const auto dirs = halo::resolve_direction_set(resolved_halo_directions(m_opt),
                                                    m_opt.selector, rank);
      if (m_opt.persistent) {
        m_persist.push_back(std::make_unique<detail::HostPersistentFaces<T>>(
            f->box(), f->domain(), decomp, rank, f->storage_halo(), comm, f->data(),
            dirs, tag0));
      } else if (m_opt.connectivity == HaloConnectivity::Full) {
        m_full.push_back(std::make_unique<detail::HostFullHalo<T>>(
            f->box(), f->domain(), decomp, rank, f->storage_halo(), comm, dirs,
            tag0));
      } else {
        m_faces.push_back(std::make_unique<detail::HostFacesHalo<T>>(
            *f, decomp, rank, comm, dirs, tag0));
      }
    }
  }

  /// Blocking exchange of every bound field.
  ///
  /// Faces posts every field first, then one `MPI_Waitall` (Kobayashi
  /// multi-field batching). Full stays sequential because each axis pass
  /// must complete before the next. Persistent stays per-field
  /// `exchange_halos()` (self-wrap persistent is MPI-implementation-sensitive).
  void exchange() {
    if (!m_persist.empty()) {
      for (auto &p : m_persist) {
        p->exchange_halos();
      }
      return;
    }
    if (!m_full.empty()) {
      for (std::size_t i = 0; i < m_full.size(); ++i) {
        m_full[i]->exchange_halos(m_fields[i]->data(), m_fields[i]->size());
      }
      return;
    }
    for (auto &h : m_faces) {
      h->start();
    }
    detail::wait_concatenated(m_faces);
  }

  /**
   * @brief Post Irecv/Isend (or MPI_Startall) and return.
   *
   * @throws std::logic_error if connectivity is Full (no split-phase API).
   */
  void start() {
    if (!m_full.empty()) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::start: Full connectivity has no "
          "split-phase API; use exchange()");
    }
    if (!m_persist.empty()) {
      for (auto &p : m_persist) {
        p->start_exchange();
      }
      return;
    }
    for (auto &h : m_faces) {
      h->start();
    }
  }

  /// Wait for `start()`.
  void finish() {
    if (!m_full.empty()) {
      throw std::logic_error(
          "pfc::comm::HaloExchange::finish: Full connectivity has no "
          "split-phase API; use exchange()");
    }
    if (!m_persist.empty()) {
      for (auto &p : m_persist) {
        p->wait_exchange();
      }
      return;
    }
    detail::wait_concatenated(m_faces);
  }

  [[nodiscard]] HaloConnectivity connectivity() const noexcept {
    return m_opt.connectivity;
  }
  [[nodiscard]] bool persistent() const noexcept { return m_opt.persistent; }
  [[nodiscard]] std::size_t num_fields() const noexcept { return m_fields.size(); }

private:
  HaloExchangeOptions m_opt{};
  std::vector<FieldT *> m_fields;
  std::vector<std::unique_ptr<detail::HostFacesHalo<T>>> m_faces;
  std::vector<std::unique_ptr<detail::HostFullHalo<T>>> m_full;
  std::vector<std::unique_ptr<detail::HostPersistentFaces<T>>> m_persist;
};

} // namespace pfc::comm
