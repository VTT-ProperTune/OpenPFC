// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file full_padded_device_halo_gpu.hpp
 * @brief Internal Full (26-direction) backend for device `pfc::comm::HaloExchange`.
 *
 * @details
 * Not a public API. Callers bind a padded `pfc::data::Field` through
 * `pfc::comm::HaloExchange<CUDASpace>` / `HaloExchange<HIPSpace>` with
 * `HaloConnectivity::Full`. This header owns `pfc::gpu::DeviceFullHalo`
 * (templated on CUDA/HIP ops). Vendor headers are thin includes.
 *
 * Full 26-fill uses three widening axis passes unless `*_FORCE_PACKED_HALO=1`
 * (that env falls back to 6-face-only DeviceFacesHalo). Without GPU-aware
 * MPI, real-neighbor axes host-stage: pack on device, D2H, MPI on host, H2D,
 * unpack. Self-only axes still pack on device with no MPI.
 *
 * Env names stay vendor-specific (`OPENPFC_CUDA_FORCE_PACKED_HALO` /
 * `OPENPFC_HIP_FORCE_PACKED_HALO`; `*_USE_SUBARRAY_HALO=1` restores
 * derived-type GPU-aware MPI). Default GPU-aware transport is
 * pack-to-contiguous + device-pointer MPI.
 *
 * @see comm_halo_exchange_gpu.hpp
 * @see padded_device_halo_exchange_gpu.hpp — device Faces
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/runtime/gpu/padded_device_halo_exchange_gpu.hpp>

namespace pfc::gpu {

/**
 * @brief 26-direction halo exchanger for a padded Field device buffer.
 *
 * Holds **3** widened slab specs, **3** sets of widened MPI face derived types
 * (one per axis pass), one device scratch buffer for self-axis pack/unpack,
 * and a flat MPI request vector sized for `2 * n_fields * (real-faces of one
 * axis pass)` — passes are run sequentially so requests are reused.
 *
 * Non-copyable; tie lifetime to the owning padded device allocations.
 */
template <typename Ops> class DeviceFullHalo {
public:
  using Int3 = pfc::types::Int3;
  using stream_t = typename Ops::stream_t;

  /**
   * @brief Construct with the historical 26-direction default (`Full3D()`).
   *
   * @param decomp      Decomposition shared by every field.
   * @param rank        MPI rank of the caller.
   * @param halo_width  Halo ring thickness `hw` on every side; must be `>=1`.
   * @param comm        MPI communicator for the exchange.
   * @param n_fields    Number of fields exchanged together; `>=1`.
   * @param base_tag    Starting MPI tag (uses `[base, base + n_fields*6)`).
   */
  DeviceFullHalo(const decomposition::Decomposition &decomp, int rank,
                 int halo_width, MPI_Comm comm, std::size_t n_fields,
                 int base_tag = 0)
      : DeviceFullHalo(decomp, rank, halo_width, comm, n_fields,
                       halo::presets::Full3D(), base_tag,
                       halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct with a user-selected halo direction set.
   *
   * **Default:** `Full3D()` — the historical 26-direction widening exchange.
   *
   * **Direction-set semantics for the 3 widening passes:**
   *   - Each axis pass `a ∈ {0=X, 1=Y, 2=Z}` is **enabled** iff at least one
   *     of `±a` is in the active set. A pass that is not enabled is skipped
   *     entirely (no MPI, no self-pack along that axis).
   *   - A pass `a` runs with **widened** slabs (covering halos filled by
   *     passes `< a`) iff some direction `d` in the active set has
   *     `d[a] != 0` **and** `d[b] != 0` for some `b < a` (i.e. corner/edge
   *     fill is requested along that axis pair). Otherwise the pass uses
   *     **narrow** slabs that match `DeviceFacesHalo`'s 6-face
   *     specs (no corner fill in that pass).
   *
   * **Examples:**
   *   - `Full3D()`   → 3 widening passes (default — produces full 26-fill).
   *   - `Full2D()`   → X (narrow) + Y (widened over X) + Z skipped (8-fill).
   *   - `Axes3D()`   → 3 narrow passes (produces 6-face-only fill).
   *   - `Axes2D()`   → 2 narrow passes (X and Y) + Z skipped (4-face-only).
   *   - **Custom**   → narrow per active axis. The 26-fill semantics no
   *     longer hold (e.g. requesting `±X` and `(0, 1, 1)` widens Z but not
   *     necessarily covers `(0, 1, 1)` correctly because Y pass did not run
   *     widened); document non-preset use as an advanced API.
   *
   * Non-face entries in custom sets influence *which* passes widen; they
   * are not exchanged as separate diagonal subarray messages.
   *
   * If `selector` is provided the active set for this rank is
   * `selector(rank)`; otherwise the uniform `dirs` is used.
   *
   * @param dirs     Direction set (defaults to `Full3D()` for back-compat).
   * @param selector Optional per-rank override of the direction set.
   */
  DeviceFullHalo(const decomposition::Decomposition &decomp, int rank,
                 int halo_width, MPI_Comm comm, std::size_t n_fields,
                 halo::HaloDirectionSet dirs, int base_tag = 0,
                 halo::HaloDirectionSelector selector = {})
      : m_rank(rank), m_halo_width(halo_width), m_comm(comm), m_base_tag(base_tag),
        m_n_fields(n_fields),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {
    if (halo_width < 1) {
      throw std::invalid_argument("DeviceFullHalo: halo_width must be >= 1");
    }
    if (n_fields == 0) {
      throw std::invalid_argument("DeviceFullHalo: n_fields must be > 0");
    }

    const auto &local_world = pfc::decomposition::get_subworld(decomp, m_rank);
    const auto local_size = pfc::world::get_size(local_world);
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
            pfc::decomposition::get_neighbor_rank(decomp, m_rank, kAxisDirs[a][f]);
      }
      m_axis_is_self[a] = (m_neighbors[a][0] == m_rank);
    }

    build_slabs_(m_nx, m_ny, m_nz, hw);
    build_types_(m_nx, m_ny, m_nz, hw);

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

    m_requests.assign(static_cast<std::size_t>(4) * m_n_fields, MPI_REQUEST_NULL);

    const bool force_packed = pfc::gpu::detail::getenv_truthy(Ops::force_packed_env);
    const bool use_subarray = pfc::gpu::detail::getenv_truthy(Ops::use_subarray_env);
    m_use_gpu_aware = !force_packed && Ops::mpi_aware();
    m_use_contiguous = m_use_gpu_aware && !use_subarray;
    m_use_full_widening = !force_packed;

    bool any_real_neighbor_axis = false;
    for (int a = 0; a < 3; ++a) {
      if (m_axis_active[a] && !m_axis_is_self[a]) {
        any_real_neighbor_axis = true;
        break;
      }
    }

    if (m_use_full_widening) {
      const bool any_self_axis = (m_axis_is_self[0] && m_axis_active[0]) ||
                                 (m_axis_is_self[1] && m_axis_active[1]) ||
                                 (m_axis_is_self[2] && m_axis_active[2]);
      if (any_self_axis && m_scratch_elems > 0) {
        Ops::malloc_dev(reinterpret_cast<void **>(&m_d_scratch),
                        m_scratch_elems * sizeof(double), Ops::malloc_full_scratch);
      }
      const bool need_device_pools = any_real_neighbor_axis && m_scratch_elems > 0 &&
                                     (m_use_contiguous || !m_use_gpu_aware);
      if (need_device_pools) {
        const std::size_t nslot = 2 * m_n_fields;
        const std::size_t bytes = m_scratch_elems * sizeof(double);
        m_d_send_pool.assign(nslot, nullptr);
        m_d_recv_pool.assign(nslot, nullptr);
        for (std::size_t i = 0; i < nslot; ++i) {
          Ops::malloc_dev(reinterpret_cast<void **>(&m_d_send_pool[i]), bytes,
                          Ops::malloc_contig_send);
          Ops::malloc_dev(reinterpret_cast<void **>(&m_d_recv_pool[i]), bytes,
                          Ops::malloc_contig_recv);
        }
        if (!m_use_gpu_aware) {
          m_h_send_pool.assign(nslot, nullptr);
          m_h_recv_pool.assign(nslot, nullptr);
          for (std::size_t i = 0; i < nslot; ++i) {
            Ops::malloc_host(reinterpret_cast<void **>(&m_h_send_pool[i]), bytes,
                             Ops::malloc_host_send);
            Ops::malloc_host(reinterpret_cast<void **>(&m_h_recv_pool[i]), bytes,
                             Ops::malloc_host_recv);
          }
        }
      }
    } else {
      m_per_field_packed.reserve(m_n_fields);
      for (std::size_t f = 0; f < m_n_fields; ++f) {
        const int per_field_tag = m_base_tag + static_cast<int>(f) * 6;
        m_per_field_packed.push_back(std::make_unique<DeviceFacesHalo<Ops>>(
            decomp, m_rank, m_halo_width, m_comm, m_dirs, per_field_tag));
      }
    }
  }

  DeviceFullHalo(const DeviceFullHalo &) = delete;
  DeviceFullHalo &operator=(const DeviceFullHalo &) = delete;

  ~DeviceFullHalo() {
    if (m_d_scratch != nullptr) {
      Ops::free_dev(m_d_scratch);
      m_d_scratch = nullptr;
    }
    for (double *p : m_d_send_pool) {
      if (p != nullptr) {
        Ops::free_dev(p);
      }
    }
    for (double *p : m_d_recv_pool) {
      if (p != nullptr) {
        Ops::free_dev(p);
      }
    }
    for (double *p : m_h_send_pool) {
      if (p != nullptr) {
        Ops::free_host(p);
      }
    }
    for (double *p : m_h_recv_pool) {
      if (p != nullptr) {
        Ops::free_host(p);
      }
    }
  }

  [[nodiscard]] bool uses_gpu_aware_mpi() const noexcept { return m_use_gpu_aware; }
  [[nodiscard]] bool uses_contiguous_device_mpi() const noexcept {
    return m_use_contiguous;
  }
  [[nodiscard]] std::size_t n_fields() const noexcept { return m_n_fields; }

  [[nodiscard]] const halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

  /**
   * @brief Fill the full 26-direction halo for `n_fields` device buffers.
   *
   * @param fields Pointer to an array of `n_fields()` device buffers in
   *               padded Field layout (outer extents `(nx+2hw, ny+2hw, nz+2hw)`).
   * @param stream Stream the caller used to populate `fields`. Fully
   *               synchronised before MPI starts.
   *
   * @note `*_FORCE_PACKED_HALO=1` falls back to a per-field 6-face exchange
   *       that **does not** fill corners or edges. Without GPU-aware MPI the
   *       26-fill still runs; real-neighbor axes host-stage packed slabs.
   */
  void exchange(double *const *fields, stream_t stream) {
    const bool perf = Ops::perf_enabled();
    auto &H = Ops::timers();

    double t_mark = MPI_Wtime();
    Ops::stream_sync(stream, Ops::sync_pre_full);
    if (perf) {
      H.pre_stream_sync += MPI_Wtime() - t_mark;
      ++H.n_calls;
    }

    const double t0 = MPI_Wtime();
    if (m_use_full_widening) {
      t_mark = MPI_Wtime();
      for (int a = 0; a < 3; ++a) {
        if (!m_axis_active[a]) {
          continue;
        }
        run_pass_(a, fields, stream);
      }
      if (perf) {
        H.gpu_aware_mpi += MPI_Wtime() - t_mark;
      }
    } else {
      for (std::size_t f = 0; f < m_n_fields; ++f) {
        m_per_field_packed[f]->exchange_halos_device(fields[f], 0, stream);
      }
    }
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionCommunication,
                                MPI_Wtime() - t0);
  }

  void exchange(std::initializer_list<double *> fields, stream_t stream = nullptr) {
    if (fields.size() != m_n_fields) {
      throw std::invalid_argument("DeviceFullHalo::exchange: field count mismatch");
    }
    exchange(fields.begin(), stream);
  }

private:
  using SlabSpec = pfc::gpu::detail::FaceSlabSpec;
  using FaceTypes = pfc::halo::FaceTypes;

  static int opposite_face_slot_(int slot) noexcept { return slot ^ 1; }

  void build_slabs_(int nx, int ny, int nz, int hw) {
    const int X = nx + 2 * hw;
    const int Y = ny + 2 * hw;

    m_slabs[0][0] = {SlabSpec{nx, hw, hw, hw, ny, nz},
                     SlabSpec{nx + hw, hw, hw, hw, ny, nz}}; // +X
    m_slabs[0][1] = {SlabSpec{hw, hw, hw, hw, ny, nz},
                     SlabSpec{0, hw, hw, hw, ny, nz}}; // -X

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

  void build_types_(int nx, int ny, int nz, int hw) {
    (void)nx;
    (void)ny;
    (void)nz;
    (void)hw;
    const MPI_Datatype elem = pfc::exchange::detail::get_mpi_type<double>();
    for (int a = 0; a < 3; ++a) {
      for (int f = 0; f < 2; ++f) {
        const auto &s = m_slabs[a][f].first;
        const auto &r = m_slabs[a][f].second;
        m_face_types[a][f].send_type = pfc::halo::create_face_type(
            m_nxp, m_nyp, m_nzp, s.ox, s.oy, s.oz, s.sx, s.sy, s.sz, elem);
        m_face_types[a][f].recv_type = pfc::halo::create_face_type(
            m_nxp, m_nyp, m_nzp, r.ox, r.oy, r.oz, r.sx, r.sy, r.sz, elem);
      }
    }
  }

  void run_pass_(int axis, double *const *fields, stream_t stream) {
    if (m_axis_is_self[axis]) {
      run_self_pass_(axis, fields, stream);
    } else {
      run_mpi_pass_(axis, fields, stream);
    }
    Ops::stream_sync(stream, Ops::sync_after_full_pass);
  }

  void run_self_pass_(int axis, double *const *fields, stream_t stream) {
    if (m_d_scratch == nullptr) {
      throw std::runtime_error(
          "DeviceFullHalo: self-axis pass needs device scratch");
    }
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      double *d_pad = fields[fld];
      for (int f = 0; f < 2; ++f) {
        const auto &send = m_slabs[axis][f].first;
        const auto &recv_opp = m_slabs[axis][f ^ 1].second;
        Ops::pack_face(m_d_scratch, d_pad, send.ox, send.oy, send.oz, send.sx,
                       send.sy, send.sz, m_nxp, m_nyp, m_nzp, stream);
        Ops::unpack_face(d_pad, m_d_scratch, recv_opp.ox, recv_opp.oy, recv_opp.oz,
                         recv_opp.sx, recv_opp.sy, recv_opp.sz, m_nxp, m_nyp, m_nzp,
                         stream);
      }
    }
    Ops::stream_sync(stream, Ops::sync_after_full_self);
  }

  void run_mpi_pass_(int axis, double *const *fields, stream_t stream) {
    if (m_use_contiguous) {
      run_mpi_pass_contiguous_(axis, fields, stream);
      return;
    }
    if (!m_use_gpu_aware) {
      run_mpi_pass_host_staged_(axis, fields, stream);
      return;
    }
    std::size_t req_count = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      void *buf = static_cast<void *>(fields[fld]);
      const int field_tag_off = m_base_tag + static_cast<int>(fld) * 6;
      for (int f = 0; f < 2; ++f) {
        const int slot = axis * 2 + f;
        const int tag = field_tag_off + opposite_face_slot_(slot);
        pfc::exchange::irecv_face(buf, m_face_types[axis][f].recv_type.get(),
                                  m_neighbors[axis][f], m_comm,
                                  &m_requests[req_count], tag);
        ++req_count;
      }
    }
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      void *buf = static_cast<void *>(fields[fld]);
      const int field_tag_off = m_base_tag + static_cast<int>(fld) * 6;
      for (int f = 0; f < 2; ++f) {
        const int slot = axis * 2 + f;
        const int tag = field_tag_off + slot;
        pfc::exchange::isend_face(buf, m_face_types[axis][f].send_type.get(),
                                  m_neighbors[axis][f], m_comm,
                                  &m_requests[req_count], tag);
        ++req_count;
      }
    }
    pfc::exchange::wait_all(m_requests.data(), static_cast<int>(req_count));
  }

  void run_mpi_pass_host_staged_(int axis, double *const *fields, stream_t stream) {
    std::size_t req_count = 0;
    std::size_t idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      const int field_tag_off = m_base_tag + static_cast<int>(fld) * 6;
      for (int f = 0; f < 2; ++f) {
        const auto &recv = m_slabs[axis][f].second;
        const std::size_t n = static_cast<std::size_t>(recv.sx) *
                              static_cast<std::size_t>(recv.sy) *
                              static_cast<std::size_t>(recv.sz);
        const int slot = axis * 2 + f;
        const int tag = field_tag_off + opposite_face_slot_(slot);
        const int face_count =
            pfc::mpi::ensure_mpi_int_count(n, "DeviceFullHalo host-staged face");
        pfc::mpi::throw_on_mpi_error(MPI_Irecv(m_h_recv_pool[idx], face_count,
                                               MPI_DOUBLE, m_neighbors[axis][f], tag,
                                               m_comm, &m_requests[req_count]),
                                     "DeviceFullHalo host-staged MPI_Irecv");
        ++req_count;
        ++idx;
      }
    }

    idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      for (int f = 0; f < 2; ++f) {
        const auto &send = m_slabs[axis][f].first;
        const std::size_t n = static_cast<std::size_t>(send.sx) *
                              static_cast<std::size_t>(send.sy) *
                              static_cast<std::size_t>(send.sz);
        Ops::pack_face(m_d_send_pool[idx], fields[fld], send.ox, send.oy, send.oz,
                       send.sx, send.sy, send.sz, m_nxp, m_nyp, m_nzp, stream);
        Ops::memcpy_async_d2h(m_h_send_pool[idx], m_d_send_pool[idx],
                              n * sizeof(double), stream, Ops::memcpy_d2h);
        ++idx;
      }
    }
    Ops::stream_sync(stream, Ops::sync_pack);

    idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      const int field_tag_off = m_base_tag + static_cast<int>(fld) * 6;
      for (int f = 0; f < 2; ++f) {
        const auto &send = m_slabs[axis][f].first;
        const std::size_t n = static_cast<std::size_t>(send.sx) *
                              static_cast<std::size_t>(send.sy) *
                              static_cast<std::size_t>(send.sz);
        const int slot = axis * 2 + f;
        const int tag = field_tag_off + slot;
        const int face_count =
            pfc::mpi::ensure_mpi_int_count(n, "DeviceFullHalo host-staged face");
        pfc::mpi::throw_on_mpi_error(MPI_Isend(m_h_send_pool[idx], face_count,
                                               MPI_DOUBLE, m_neighbors[axis][f], tag,
                                               m_comm, &m_requests[req_count]),
                                     "DeviceFullHalo host-staged MPI_Isend");
        ++req_count;
        ++idx;
      }
    }
    pfc::exchange::wait_all(m_requests.data(), static_cast<int>(req_count));

    idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      for (int f = 0; f < 2; ++f) {
        const auto &recv = m_slabs[axis][f].second;
        const std::size_t n = static_cast<std::size_t>(recv.sx) *
                              static_cast<std::size_t>(recv.sy) *
                              static_cast<std::size_t>(recv.sz);
        Ops::memcpy_async_h2d(m_d_recv_pool[idx], m_h_recv_pool[idx],
                              n * sizeof(double), stream, Ops::memcpy_h2d);
        Ops::unpack_face(fields[fld], m_d_recv_pool[idx], recv.ox, recv.oy, recv.oz,
                         recv.sx, recv.sy, recv.sz, m_nxp, m_nyp, m_nzp, stream);
        ++idx;
      }
    }
  }

  void run_mpi_pass_contiguous_(int axis, double *const *fields, stream_t stream) {
    std::size_t req_count = 0;
    std::size_t idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      const int field_tag_off = m_base_tag + static_cast<int>(fld) * 6;
      for (int f = 0; f < 2; ++f) {
        const auto &recv = m_slabs[axis][f].second;
        const std::size_t n = static_cast<std::size_t>(recv.sx) *
                              static_cast<std::size_t>(recv.sy) *
                              static_cast<std::size_t>(recv.sz);
        const int slot = axis * 2 + f;
        const int tag = field_tag_off + opposite_face_slot_(slot);
        const int face_count =
            pfc::mpi::ensure_mpi_int_count(n, "DeviceFullHalo contig face");
        pfc::mpi::throw_on_mpi_error(MPI_Irecv(m_d_recv_pool[idx], face_count,
                                               MPI_DOUBLE, m_neighbors[axis][f], tag,
                                               m_comm, &m_requests[req_count]),
                                     "DeviceFullHalo contig MPI_Irecv");
        ++req_count;
        ++idx;
      }
    }

    idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      for (int f = 0; f < 2; ++f) {
        const auto &send = m_slabs[axis][f].first;
        Ops::pack_face(m_d_send_pool[idx], fields[fld], send.ox, send.oy, send.oz,
                       send.sx, send.sy, send.sz, m_nxp, m_nyp, m_nzp, stream);
        ++idx;
      }
    }
    Ops::stream_sync(stream, Ops::sync_pack);

    idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      const int field_tag_off = m_base_tag + static_cast<int>(fld) * 6;
      for (int f = 0; f < 2; ++f) {
        const auto &send = m_slabs[axis][f].first;
        const std::size_t n = static_cast<std::size_t>(send.sx) *
                              static_cast<std::size_t>(send.sy) *
                              static_cast<std::size_t>(send.sz);
        const int slot = axis * 2 + f;
        const int tag = field_tag_off + slot;
        const int face_count =
            pfc::mpi::ensure_mpi_int_count(n, "DeviceFullHalo contig face");
        pfc::mpi::throw_on_mpi_error(MPI_Isend(m_d_send_pool[idx], face_count,
                                               MPI_DOUBLE, m_neighbors[axis][f], tag,
                                               m_comm, &m_requests[req_count]),
                                     "DeviceFullHalo contig MPI_Isend");
        ++req_count;
        ++idx;
      }
    }
    pfc::exchange::wait_all(m_requests.data(), static_cast<int>(req_count));

    idx = 0;
    for (std::size_t fld = 0; fld < m_n_fields; ++fld) {
      for (int f = 0; f < 2; ++f) {
        const auto &recv = m_slabs[axis][f].second;
        Ops::unpack_face(fields[fld], m_d_recv_pool[idx], recv.ox, recv.oy, recv.oz,
                         recv.sx, recv.sy, recv.sz, m_nxp, m_nyp, m_nzp, stream);
        ++idx;
      }
    }
  }

  int m_rank = 0;
  int m_halo_width = 1;
  MPI_Comm m_comm = MPI_COMM_NULL;
  int m_base_tag = 0;
  std::size_t m_n_fields = 0;
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

  bool m_use_gpu_aware = false;
  bool m_use_contiguous = false;
  bool m_use_full_widening = false;

  double *m_d_scratch = nullptr;
  std::vector<double *> m_d_send_pool;
  std::vector<double *> m_d_recv_pool;
  std::vector<double *> m_h_send_pool;
  std::vector<double *> m_h_recv_pool;

  std::vector<std::unique_ptr<DeviceFacesHalo<Ops>>> m_per_field_packed;
};

} // namespace pfc::gpu

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
