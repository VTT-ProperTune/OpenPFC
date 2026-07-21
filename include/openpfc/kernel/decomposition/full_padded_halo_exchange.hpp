// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file full_padded_halo_exchange.hpp
 * @brief Full **26-direction** host halo exchange (faces + edges + corners)
 *        for a `PaddedBrick` buffer.
 *
 * @details
 * `pfc::communication::PaddedHaloExchanger` performs a **single-pass 6-face**
 * exchange. After it returns, only the **6 axis-aligned face halos** are
 * populated — corners and edges are left untouched. That is sufficient for
 * 7-point Laplacians and other axis-aligned stencils.
 *
 * It is **not** sufficient for stencils that read diagonal neighbours
 * `(i±1, j±1, k)` or `(i±1, j±1, k±1)` — i.e. anything that needs the
 * **mixed second derivatives** `u_xy`, `u_xz`, `u_yz`.
 *
 * `FullPaddedHaloExchanger` populates **all 26 neighbour halo cells** in 3
 * **widening passes** along the canonical X → Y → Z axis order. After it
 * returns, every cell of the halo ring `[-hw, 0)` and `[n, n+hw)` on every
 * axis (faces, edges, corners) carries the periodic-equivalent value of the
 * corresponding interior cell on the appropriate neighbour rank.
 *
 * This is the host twin of `pfc::cuda::FullPaddedDeviceHalo` (same slab
 * geometry and tag/slot conventions). Kernel headers must not include the
 * CUDA runtime header; the algorithm is duplicated here for G3 layering.
 *
 * **Per-axis self-handling:** if the process grid has extent **1** along
 * axis `a`, the ±a neighbours are the **same MPI rank**. Those passes use
 * a **host pack/unpack** of the widened slabs instead of MPI send/recv to
 * self. The periodic direction is correct: the source slab for the +a halo
 * is the rank's **first** `hw` owned cells along `a`.
 *
 * @see padded_halo_exchange.hpp — single-pass 6-face host exchanger
 * @see runtime/cuda/full_padded_device_halo.hpp — device twin (CUDA)
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc::communication {

namespace detail {

/// Private slab POD (mirrors CUDA `FaceSlabSpec`; not vendored from runtime).
struct FullPaddedSlabSpec {
  int ox = 0;
  int oy = 0;
  int oz = 0;
  int sx = 0;
  int sy = 0;
  int sz = 0;
};

} // namespace detail

/**
 * @brief 26-direction host halo exchanger for a `PaddedBrick<T>` buffer.
 *
 * Holds 3 widened slab specs, 3 sets of MPI face derived types (one per
 * axis pass), a host scratch buffer for self-axis pack/unpack, and an MPI
 * request vector sized for one axis pass. Passes run sequentially.
 *
 * Non-copyable; tie lifetime to the owning padded brick (or buffer).
 */
template <typename T = double> class FullPaddedHaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct with the historical 26-direction default (`Full3D()`).
   *
   * @param decomp      Decomposition (must outlive this object).
   * @param rank        MPI rank of the caller.
   * @param halo_width  Halo ring thickness `hw` on every side; must be `>=1`.
   * @param comm        MPI communicator for the exchange.
   * @param base_tag    Starting MPI tag (uses `[base, base + 6)`).
   */
  FullPaddedHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm, int base_tag = 0)
      : FullPaddedHaloExchanger(decomp, rank, halo_width, comm,
                                halo::presets::Full3D(), base_tag,
                                halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct with a user-selected halo direction set.
   *
   * **Default:** `Full3D()` — the historical 26-direction widening exchange.
   *
   * **Direction-set semantics for the 3 widening passes** (same as the
   * CUDA twin):
   *   - Each axis pass `a ∈ {0=X, 1=Y, 2=Z}` is **enabled** iff at least one
   *     of `±a` is in the active set.
   *   - A pass `a` runs with **widened** slabs iff some direction `d` in the
   *     active set has `d[a] != 0` and `d[b] != 0` for some `b < a`.
   *
   * If `selector` is provided the active set for this rank is
   * `selector(rank)`; otherwise the uniform `dirs` is used.
   */
  FullPaddedHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm,
                          halo::HaloDirectionSet dirs, int base_tag = 0,
                          halo::HaloDirectionSelector selector = {})
      : m_rank(rank), m_halo_width(halo_width), m_comm(comm),
        m_base_tag(base_tag),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {
    if (halo_width < 1) {
      throw std::invalid_argument(
          "FullPaddedHaloExchanger: halo_width must be >= 1");
    }

    const auto &local_world = decomposition::get_subworld(decomp, m_rank);
    const auto local_size = world::get_size(local_world);
    m_nx = local_size[0];
    m_ny = local_size[1];
    m_nz = local_size[2];
    const int hw = m_halo_width;
    m_nxp = m_nx + 2 * hw;
    m_nyp = m_ny + 2 * hw;
    m_nzp = m_nz + 2 * hw;

    static constexpr std::array<std::array<Int3, 2>, 3> kAxisDirs = {{
        {{Int3{1, 0, 0}, Int3{-1, 0, 0}}}, // X
        {{Int3{0, 1, 0}, Int3{0, -1, 0}}}, // Y
        {{Int3{0, 0, 1}, Int3{0, 0, -1}}}, // Z
    }};
    for (int a = 0; a < 3; ++a) {
      m_axis_active[a] =
          m_dirs.contains(kAxisDirs[a][0]) || m_dirs.contains(kAxisDirs[a][1]);
      m_axis_widen[a] = false;
      if (a > 0) {
        for (const auto &d : m_dirs.dirs) {
          if (d[a] == 0) {
            continue;
          }
          for (int b = 0; b < a; ++b) {
            if (d[b] != 0) {
              m_axis_widen[a] = true;
              break;
            }
          }
          if (m_axis_widen[a]) {
            break;
          }
        }
      }
      for (int f = 0; f < 2; ++f) {
        m_neighbors[a][f] =
            decomposition::get_neighbor_rank(decomp, m_rank, kAxisDirs[a][f]);
      }
      m_axis_is_self[a] = (m_neighbors[a][0] == m_rank);
    }

    build_slabs_(m_nx, m_ny, m_nz, hw);
    build_types_();

    m_scratch_elems = 0;
    for (int a = 0; a < 3; ++a) {
      if (!m_axis_active[a]) {
        continue;
      }
      for (int f = 0; f < 2; ++f) {
        const auto &send = m_slabs[a][f].first;
        const std::size_t c = static_cast<std::size_t>(send.sx) *
                              static_cast<std::size_t>(send.sy) *
                              static_cast<std::size_t>(send.sz);
        m_scratch_elems = std::max(m_scratch_elems, c);
      }
    }
    const bool any_self_axis = (m_axis_is_self[0] && m_axis_active[0]) ||
                               (m_axis_is_self[1] && m_axis_active[1]) ||
                               (m_axis_is_self[2] && m_axis_active[2]);
    if (any_self_axis && m_scratch_elems > 0) {
      m_scratch.assign(m_scratch_elems, T{});
    }

    // One axis pass posts at most 2 Irecvs + 2 Isends.
    m_requests.assign(4, MPI_REQUEST_NULL);
  }

  /**
   * @brief Preferred: bind layout + buffer from a `PaddedBrick<T>`.
   *
   * Pulls decomposition, rank, and halo width from `u`. Default direction
   * set is `Full3D()`.
   */
  FullPaddedHaloExchanger(field::PaddedBrick<T> &u, MPI_Comm comm,
                          int base_tag = 0)
      : FullPaddedHaloExchanger(u.decomposition(), u.rank(), u.halo_width(),
                                comm, halo::presets::Full3D(), base_tag,
                                halo::HaloDirectionSelector{}) {
    bind_(u);
  }

  /// Same as the brick-binding constructor, with a custom direction set.
  FullPaddedHaloExchanger(field::PaddedBrick<T> &u, MPI_Comm comm,
                          halo::HaloDirectionSet dirs, int base_tag = 0,
                          halo::HaloDirectionSelector selector = {})
      : FullPaddedHaloExchanger(u.decomposition(), u.rank(), u.halo_width(),
                                comm, dirs, base_tag, selector) {
    bind_(u);
  }

  FullPaddedHaloExchanger(const FullPaddedHaloExchanger &) = delete;
  FullPaddedHaloExchanger &operator=(const FullPaddedHaloExchanger &) = delete;

  /**
   * @brief Blocking 3-pass exchange on an explicit padded buffer.
   *
   * @param padded_buf  Pointer to the start of the padded brick
   *                    (`brick.data()`). Layout: row-major
   *                    `(nx+2hw, ny+2hw, nz+2hw)`, x fastest.
   * @param padded_size Total elements (`brick.size()`); accepted for API
   *                    symmetry, unused by the exchange path.
   */
  void exchange_halos(T *padded_buf, std::size_t padded_size) {
    (void)padded_size;
    const double t0 = MPI_Wtime();
    for (int a = 0; a < 3; ++a) {
      if (!m_axis_active[a]) {
        continue;
      }
      if (m_axis_is_self[a]) {
        run_self_pass_(a, padded_buf);
      } else {
        run_mpi_pass_(a, padded_buf);
      }
    }
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
  }

  /// Blocking exchange on the bound brick (requires brick-binding ctor).
  void exchange() {
    require_bound_("exchange()");
    exchange_halos(m_bound_buf, m_bound_size);
  }

  [[nodiscard]] bool is_bound() const noexcept { return m_bound_buf != nullptr; }

  [[nodiscard]] const halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

private:
  using SlabSpec = detail::FullPaddedSlabSpec;
  using FaceTypes = halo::FaceTypes;

  static int opposite_face_slot_(int slot) noexcept { return slot ^ 1; }

  static std::size_t lin_(int i, int j, int k, int nxp, int nyp) noexcept {
    return static_cast<std::size_t>(i) +
           static_cast<std::size_t>(j) * static_cast<std::size_t>(nxp) +
           static_cast<std::size_t>(k) * static_cast<std::size_t>(nxp) *
               static_cast<std::size_t>(nyp);
  }

  void bind_(field::PaddedBrick<T> &u) noexcept {
    m_bound_buf = u.data();
    m_bound_size = u.size();
  }

  void require_bound_(const char *what) const {
    if (m_bound_buf == nullptr) {
      throw std::logic_error(
          std::string("pfc::communication::FullPaddedHaloExchanger::") + what +
          ": exchanger is not bound to a PaddedBrick. "
          "Use the (PaddedBrick&, MPI_Comm) constructor or call "
          "exchange_halos(buf, size) directly.");
    }
  }

  /// Build per-axis (send, recv) slab pairs — offsets copied from CUDA twin.
  void build_slabs_(int nx, int ny, int nz, int hw) {
    const int X = nx + 2 * hw;
    const int Y = ny + 2 * hw;

    // Pass 0 — X axis: always narrow.
    m_slabs[0][0] = {SlabSpec{nx, hw, hw, hw, ny, nz},
                     SlabSpec{nx + hw, hw, hw, hw, ny, nz}}; // +X
    m_slabs[0][1] = {SlabSpec{hw, hw, hw, hw, ny, nz},
                     SlabSpec{0, hw, hw, hw, ny, nz}}; // -X

    // Pass 1 — Y axis.
    if (m_axis_widen[1]) {
      m_slabs[1][0] = {SlabSpec{0, ny, hw, X, hw, nz},
                       SlabSpec{0, ny + hw, hw, X, hw, nz}}; // +Y widened
      m_slabs[1][1] = {SlabSpec{0, hw, hw, X, hw, nz},
                       SlabSpec{0, 0, hw, X, hw, nz}}; // -Y widened
    } else {
      m_slabs[1][0] = {SlabSpec{hw, ny, hw, nx, hw, nz},
                       SlabSpec{hw, ny + hw, hw, nx, hw, nz}}; // +Y narrow
      m_slabs[1][1] = {SlabSpec{hw, hw, hw, nx, hw, nz},
                       SlabSpec{hw, 0, hw, nx, hw, nz}}; // -Y narrow
    }

    // Pass 2 — Z axis.
    if (m_axis_widen[2]) {
      m_slabs[2][0] = {SlabSpec{0, 0, nz, X, Y, hw},
                       SlabSpec{0, 0, nz + hw, X, Y, hw}}; // +Z widened
      m_slabs[2][1] = {SlabSpec{0, 0, hw, X, Y, hw},
                       SlabSpec{0, 0, 0, X, Y, hw}}; // -Z widened
    } else {
      m_slabs[2][0] = {SlabSpec{hw, hw, nz, nx, ny, hw},
                       SlabSpec{hw, hw, nz + hw, nx, ny, hw}}; // +Z narrow
      m_slabs[2][1] = {SlabSpec{hw, hw, hw, nx, ny, hw},
                       SlabSpec{hw, hw, 0, nx, ny, hw}}; // -Z narrow
    }
  }

  void build_types_() {
    const MPI_Datatype elem = exchange::detail::get_mpi_type<T>();
    for (int a = 0; a < 3; ++a) {
      for (int f = 0; f < 2; ++f) {
        const auto &s = m_slabs[a][f].first;
        const auto &r = m_slabs[a][f].second;
        m_face_types[a][f].send_type = halo::create_face_type(
            m_nxp, m_nyp, m_nzp, s.ox, s.oy, s.oz, s.sx, s.sy, s.sz, elem);
        m_face_types[a][f].recv_type = halo::create_face_type(
            m_nxp, m_nyp, m_nzp, r.ox, r.oy, r.oz, r.sx, r.sy, r.sz, elem);
      }
    }
  }

  void pack_slab_(const T *buf, const SlabSpec &s) {
    std::size_t idx = 0;
    for (int k = 0; k < s.sz; ++k) {
      for (int j = 0; j < s.sy; ++j) {
        for (int i = 0; i < s.sx; ++i) {
          m_scratch[idx++] =
              buf[lin_(s.ox + i, s.oy + j, s.oz + k, m_nxp, m_nyp)];
        }
      }
    }
  }

  void unpack_slab_(T *buf, const SlabSpec &r) {
    std::size_t idx = 0;
    for (int k = 0; k < r.sz; ++k) {
      for (int j = 0; j < r.sy; ++j) {
        for (int i = 0; i < r.sx; ++i) {
          buf[lin_(r.ox + i, r.oy + j, r.oz + k, m_nxp, m_nyp)] =
              m_scratch[idx++];
        }
      }
    }
  }

  /// Periodic self-loop: pack send slab of face `f` into opposite recv.
  void run_self_pass_(int axis, T *padded_buf) {
    if (m_scratch.empty()) {
      throw std::runtime_error(
          "FullPaddedHaloExchanger: self-axis pass needs host scratch");
    }
    for (int f = 0; f < 2; ++f) {
      const auto &send = m_slabs[axis][f].first;
      const auto &recv_opp = m_slabs[axis][f ^ 1].second;
      pack_slab_(padded_buf, send);
      unpack_slab_(padded_buf, recv_opp);
    }
  }

  /// Real-MPI exchange along `axis`: post Irecvs, then Isends, wait.
  void run_mpi_pass_(int axis, T *padded_buf) {
    void *buf = static_cast<void *>(padded_buf);
    std::size_t req_count = 0;
    for (int f = 0; f < 2; ++f) {
      const int slot = axis * 2 + f;
      const int tag = m_base_tag + opposite_face_slot_(slot);
      exchange::irecv_face(buf, m_face_types[axis][f].recv_type.get(),
                           m_neighbors[axis][f], m_comm, &m_requests[req_count],
                           tag);
      ++req_count;
    }
    for (int f = 0; f < 2; ++f) {
      const int slot = axis * 2 + f;
      const int tag = m_base_tag + slot;
      exchange::isend_face(buf, m_face_types[axis][f].send_type.get(),
                           m_neighbors[axis][f], m_comm, &m_requests[req_count],
                           tag);
      ++req_count;
    }
    exchange::wait_all(m_requests.data(), static_cast<int>(req_count));
  }

  int m_rank = 0;
  int m_halo_width = 1;
  MPI_Comm m_comm = MPI_COMM_NULL;
  int m_base_tag = 0;
  halo::HaloDirectionSet m_dirs;

  int m_nx = 0, m_ny = 0, m_nz = 0;
  int m_nxp = 0, m_nyp = 0, m_nzp = 0;

  std::array<std::array<std::pair<SlabSpec, SlabSpec>, 2>, 3> m_slabs{};
  std::array<std::array<FaceTypes, 2>, 3> m_face_types{};

  std::array<std::array<int, 2>, 3> m_neighbors{};
  std::array<bool, 3> m_axis_is_self{};
  std::array<bool, 3> m_axis_active{};
  std::array<bool, 3> m_axis_widen{};

  std::vector<MPI_Request> m_requests;
  std::size_t m_scratch_elems = 0;
  std::vector<T> m_scratch;

  T *m_bound_buf = nullptr;
  std::size_t m_bound_size = 0;
};

} // namespace pfc::communication

namespace pfc {
using communication::FullPaddedHaloExchanger;
} // namespace pfc
