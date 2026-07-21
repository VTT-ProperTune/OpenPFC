// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file kobayashi_batched_halo.hpp
 * @brief Per-app **multi-field** halo exchange for `kobayashi_fd_cuda`.
 *
 * @details
 * Phase 1 workaround that posts **all fields' halos in a single MPI round** with
 * **one** pre-stream sync, **one** `MPI_Waitall` and **one** post-device sync per
 * timestep, instead of one of each per field as
 * `pfc::cuda::PaddedDeviceHaloExchanger` does today. This cuts MPI/CUDA wait
 * overhead in the **small / latency-bound** regime where Kobayashi sits with 2
 * GPUs and 512^2.
 *
 * Reuses the GPU-aware MPI / packed fallback selection logic and the
 * `OPENPFC_CUDA_PROFILE_HALO` timers from
 * `openpfc/runtime/cuda/padded_device_halo_exchange.hpp` so the existing
 * `OPENPFC_CUDA_PROFILE_HALO_SUMMARY` line keeps working. In batched mode
 * `n_exchange_calls_max` reports **steps**, not fields x steps.
 *
 * GPU-aware path (the case we want to optimise):
 *   1. cudaStreamSynchronize(stream)              (one for all fields)
 *   2. for each field f, for each self-neighbour face:  pack -> unpack on stream
 *   3. cudaStreamSynchronize(stream)              (only if any self face was packed)
 *   4. for each field f, post Irecv on every non-self face
 *   5. for each field f, post Isend on every non-self face
 *   6. MPI_Waitall                                 (one for all fields)
 *   7. cudaDeviceSynchronize                      (one for all fields)
 *
 * Tag layout: `base_tag + field_idx * 6 + face_slot`. Face slots match
 * `pfc::halo::create_padded_face_types_6` (+X,-X,+Y,-Y,+Z,-Z); paired ranks use
 * the opposite slot for the recv tag (same convention as
 * `PaddedDeviceHaloExchanger::opposite_slot`).
 *
 * **Corner-fill mode (`corner_fill=true`):** the standard 6-face exchange does
 * **not** fill corner halo cells (e.g. `phi(-1, -1)` on a `2x1x1` proc grid).
 * For Kobayashi's "extended-halo" path, where stage_a is launched on
 * `interior + 1 ring` and a 5-point stencil at `(-1, 0)` reads `phi(-1, -1)`,
 * we need corners filled. With corner_fill the order is reversed:
 *   1. Post Irecv/Isend for **all real-MPI** faces (narrow slabs, ±X only in
 *      the supported nproc=2 case).
 *   2. MPI_Waitall (X halos now contain neighbour interior data).
 *   3. Self-pack/unpack for ±Y using a **widened X slab** (full padded X
 *      including the just-arrived X-halos). This propagates X-halo data into
 *      the X-Y corners.
 *   4. cudaStreamSynchronize.
 * The widened ±Z self-pack is unused by 2D Kobayashi (kernels read only iz=0)
 * but the implementation widens it for completeness. Currently this single-pass
 * corner-fill is correct only when at most **one axis has real MPI neighbours**
 * (i.e. proc grid has extent 1 along Y and Z); we assert that in the
 * constructor.
 *
 * Packed fallback path: not optimised here -- falls back to one
 * `pfc::cuda::PaddedDeviceHaloExchanger` per field so that
 * `OPENPFC_CUDA_FORCE_PACKED_HALO=1` still works for correctness checks. The
 * batched gain is in the GPU-aware path only.
 */

#if !defined(OpenPFC_ENABLE_CUDA)
#error "kobayashi_batched_halo.hpp requires CUDA"
#endif

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/runtime/cuda/padded_device_halo_exchange.hpp>

namespace kobayashi::cuda {

/**
 * @brief Multi-field padded device halo exchanger for the Kobayashi driver.
 *
 * Holds **one** set of 6 derived MPI face types and **one** scratch device slab,
 * and accepts an array of `n_fields` device pointers per `exchange` call. All
 * fields share the same `Decomposition`, halo width, and neighbour table.
 */
class BatchedPaddedDeviceHalo {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct with the historical 6-face axis-aligned set (`Axes3D()`).
   *
   * @param corner_fill See class docs; orthogonal to the direction set.
   */
  BatchedPaddedDeviceHalo(const pfc::decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm, std::size_t n_fields,
                          int base_tag = 0, bool corner_fill = false)
      : BatchedPaddedDeviceHalo(decomp, rank, halo_width, comm, n_fields,
                                pfc::halo::presets::Axes3D(), base_tag, corner_fill,
                                pfc::halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct with a user-selected halo direction set.
   *
   * Both the GPU-aware and packed branches skip excluded slots; same-rank
   * periodic faces inside the active set continue to use device pack/unpack
   * (no MPI-to-self).
   *
   * `corner_fill` is **orthogonal** to the direction set: it only changes
   * **how** corners are populated when the active set already includes
   * those directions (currently the implementation runs with face-only
   * specs but uses widened slabs for self ±Y/±Z when `corner_fill=true`).
   * For most callers using `Axes2D()` / `Axes3D()` it should stay `false`.
   *
   * @param dirs        Direction set (defaults to `Axes3D()` for back-compat).
   * @param corner_fill See class docs; orthogonal to the direction set.
   * @param selector    Optional per-rank override of the direction set.
   */
  BatchedPaddedDeviceHalo(const pfc::decomposition::Decomposition &decomp, int rank,
                          int halo_width, MPI_Comm comm, std::size_t n_fields,
                          pfc::halo::HaloDirectionSet dirs, int base_tag = 0,
                          bool corner_fill = false,
                          pfc::halo::HaloDirectionSelector selector = {})
      : m_rank(rank), m_halo_width(halo_width), m_comm(comm), m_base_tag(base_tag),
        m_n_fields(n_fields), m_corner_fill(corner_fill),
        m_dirs(pfc::halo::resolve_direction_set(dirs, selector, rank)) {
    if (n_fields == 0) {
      throw std::invalid_argument("BatchedPaddedDeviceHalo: n_fields must be > 0");
    }

    const auto &local_world = pfc::decomposition::get_subworld(decomp, m_rank);
    const auto local_size = pfc::world::get_size(local_world);
    const int nx = local_size[0];
    const int ny = local_size[1];
    const int nz = local_size[2];
    const int hw = m_halo_width;

    m_nxp = nx + 2 * hw;
    m_nyp = ny + 2 * hw;
    m_nzp = nz + 2 * hw;

    m_face_specs = pfc::cuda::detail::make_padded_face_slabs(nx, ny, nz, hw);
    m_face_specs_widened = make_widened_self_face_slabs_(nx, ny, nz, hw);

    m_face_types = pfc::halo::create_padded_face_types_6(
        nx, ny, nz, m_halo_width, pfc::exchange::detail::get_mpi_type<double>());

    const std::array<Int3, 6> dirs_canon{Int3{1, 0, 0}, Int3{-1, 0, 0},
                                         Int3{0, 1, 0}, Int3{0, -1, 0},
                                         Int3{0, 0, 1}, Int3{0, 0, -1}};
    m_neighbors.reserve(6);
    for (std::size_t i = 0; i < 6; ++i) {
      m_active[i] = m_dirs.contains(dirs_canon[i]);
      m_neighbors.push_back(
          pfc::decomposition::get_neighbor_rank(decomp, m_rank, dirs_canon[i]));
    }

    m_any_self_neighbor = false;
    int n_real_faces = 0;
    bool real_x = false, real_y = false, real_z = false;
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i]) {
        continue;
      }
      const int n = m_neighbors[i];
      if (n == m_rank) {
        m_any_self_neighbor = true;
      } else {
        ++n_real_faces;
        if (i < 2)
          real_x = true;
        else if (i < 4)
          real_y = true;
        else
          real_z = true;
      }
    }
    if (m_corner_fill) {
      const int real_axes = static_cast<int>(real_x) + static_cast<int>(real_y) +
                            static_cast<int>(real_z);
      if (real_axes > 1) {
        throw std::runtime_error(
            "BatchedPaddedDeviceHalo(corner_fill=true): single-pass corner fill "
            "only "
            "supports proc grids with at most ONE axis of real MPI neighbours "
            "(e.g. nproc=2 with 2x1x1). Use the multi-pass scheme for general "
            "grids.");
      }
    }
    m_requests.assign(static_cast<std::size_t>(2 * n_real_faces) * m_n_fields,
                      MPI_REQUEST_NULL);

    m_scratch_elems = 0;
    for (std::size_t i = 0; i < 6; ++i) {
      const auto &send_n = m_face_specs[i].first;
      const auto &send_w = m_face_specs_widened[i].first;
      const std::size_t cn = static_cast<std::size_t>(send_n.sx) *
                             static_cast<std::size_t>(send_n.sy) *
                             static_cast<std::size_t>(send_n.sz);
      const std::size_t cw = static_cast<std::size_t>(send_w.sx) *
                             static_cast<std::size_t>(send_w.sy) *
                             static_cast<std::size_t>(send_w.sz);
      m_face_elems[i] = cn;
      // Excluded slots never pack/unpack — don't count them when sizing scratch.
      if (m_active[i]) {
        m_scratch_elems = std::max(m_scratch_elems, std::max(cn, cw));
      }
    }

    const bool force_packed =
        pfc::cuda::detail::getenv_truthy("OPENPFC_CUDA_FORCE_PACKED_HALO");
    m_use_gpu_aware = !force_packed && pfc::cuda::detail::runtime_mpi_cuda_aware();

    if (m_use_gpu_aware) {
      if (m_any_self_neighbor && m_scratch_elems > 0) {
        pfc::cuda::detail::cuda_check(
            cudaMalloc(reinterpret_cast<void **>(&m_d_scratch),
                       m_scratch_elems * sizeof(double)),
            "cudaMalloc batched halo device scratch (self-face pack/unpack)");
      }
    } else {
      // Packed fallback: keep one PaddedDeviceHaloExchanger per field. We do not
      // try to micro-optimise this path -- the perf goal is the gpu_aware branch.
      // The packed exchanger inherits the active direction set so e.g. an
      // `Axes2D()` request still skips ±Z on the packed path.
      m_per_field_packed.reserve(m_n_fields);
      for (std::size_t f = 0; f < m_n_fields; ++f) {
        const int per_field_tag = m_base_tag + static_cast<int>(f) * 6;
        m_per_field_packed.push_back(
            std::make_unique<pfc::cuda::PaddedDeviceHaloExchanger>(
                decomp, m_rank, m_halo_width, m_comm, m_dirs, per_field_tag));
      }
    }
  }

  BatchedPaddedDeviceHalo(const BatchedPaddedDeviceHalo &) = delete;
  BatchedPaddedDeviceHalo &operator=(const BatchedPaddedDeviceHalo &) = delete;

  ~BatchedPaddedDeviceHalo() {
    if (m_d_scratch != nullptr) {
      (void)cudaFree(m_d_scratch);
      m_d_scratch = nullptr;
    }
  }

  [[nodiscard]] bool uses_gpu_aware_mpi() const noexcept { return m_use_gpu_aware; }
  [[nodiscard]] std::size_t n_fields() const noexcept { return m_n_fields; }

  /** Read-only access to the active direction set (after `selector` resolution). */
  [[nodiscard]] const pfc::halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

  /**
   * @brief Exchange halos for `n_fields` device buffers in **one** MPI round.
   *
   * @param fields Pointer to an array of `n_fields()` device buffers (padded brick
   *               layout, all sharing the same decomposition and halo width).
   * @param stream CUDA stream used by the caller to populate `fields`. The
   *               stream is fully synchronised before MPI starts.
   */
  void exchange(double *const *fields, cudaStream_t stream) {
    const bool perf = pfc::cuda::cuda_halo_exchange_perf_enabled();
    auto &H = pfc::cuda::cuda_halo_exchange_cpu_timers();

    double t_mark = MPI_Wtime();
    pfc::cuda::detail::cuda_check(cudaStreamSynchronize(stream),
                                  "cudaStreamSynchronize pre batched halo");
    if (perf) {
      H.pre_stream_sync += MPI_Wtime() - t_mark;
      ++H.n_calls;
    }

    const double t0 = MPI_Wtime();
    if (m_use_gpu_aware) {
      t_mark = MPI_Wtime();
      exchange_gpu_aware_(fields, stream);
      if (perf) {
        H.gpu_aware_mpi += MPI_Wtime() - t_mark;
      }
      t_mark = MPI_Wtime();
      pfc::cuda::detail::cuda_check(
          cudaDeviceSynchronize(),
          "cudaDeviceSynchronize post batched GPU-aware MPI");
      if (perf) {
        H.post_exchange_cuda_sync += MPI_Wtime() - t_mark;
      }
    } else {
      // Packed fallback: per-field, no batching. Each call internally adds one
      // entry to `H.n_calls` (so packed n_exchange_calls_max == steps * n_fields).
      for (std::size_t f = 0; f < m_n_fields; ++f) {
        m_per_field_packed[f]->exchange_halos_device(fields[f], 0, stream);
      }
    }
    pfc::profiling::record_time(pfc::profiling::kProfilingRegionCommunication,
                                MPI_Wtime() - t0);
  }

  /// Convenience overload that accepts an `std::initializer_list` of `n_fields()`
  /// pointers.
  void exchange(std::initializer_list<double *> fields,
                cudaStream_t stream = nullptr) {
    if (fields.size() != m_n_fields) {
      throw std::invalid_argument(
          "BatchedPaddedDeviceHalo::exchange: field count mismatch");
    }
    exchange(fields.begin(), stream);
  }

private:
  static int opposite_slot_(int slot) noexcept {
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

  void exchange_gpu_aware_(double *const *fields, cudaStream_t stream) {
    if (m_corner_fill) {
      exchange_gpu_aware_corner_fill_(fields, stream);
      return;
    }

    // Step 1: device pack/unpack for all (field, self-face) pairs on `stream`.
    if (m_any_self_neighbor) {
      if (m_d_scratch == nullptr) {
        throw std::runtime_error(
            "BatchedPaddedDeviceHalo: self-neighbor halo needs device scratch");
      }
      for (std::size_t f = 0; f < m_n_fields; ++f) {
        double *d_pad = fields[f];
        // Opposite-face unpack matches PaddedDeviceHaloExchanger / FullPaddedDeviceHalo.
        for (std::size_t i = 0; i < 6; ++i) {
          if (!m_active[i] || m_neighbors[i] != m_rank) {
            continue;
          }
          const auto &send = m_face_specs[i].first;
          const auto &recv =
              m_face_specs[static_cast<std::size_t>(
                               opposite_slot_(static_cast<int>(i)))]
                  .second;
          pfc::cuda::detail::launch_padded_pack_face(
              m_d_scratch, d_pad, send.ox, send.oy, send.oz, send.sx, send.sy,
              send.sz, m_nxp, m_nyp, m_nzp, stream);
          pfc::cuda::detail::launch_padded_unpack_face(
              d_pad, m_d_scratch, recv.ox, recv.oy, recv.oz, recv.sx, recv.sy,
              recv.sz, m_nxp, m_nyp, m_nzp, stream);
        }
      }
      pfc::cuda::detail::cuda_check(
          cudaStreamSynchronize(stream),
          "cudaStreamSynchronize after batched self-neighbor halo copies");
    }

    // Step 2: post all Irecvs, then all Isends (one per (field, real face)).
    std::size_t req_count = 0;
    for (std::size_t f = 0; f < m_n_fields; ++f) {
      void *buf = static_cast<void *>(fields[f]);
      const int field_tag_off = m_base_tag + static_cast<int>(f) * 6;
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_neighbors[i] == m_rank) {
          continue;
        }
        const int tag = field_tag_off + opposite_slot_(static_cast<int>(i));
        pfc::exchange::irecv_face(buf, m_face_types[i].recv_type.get(),
                                  m_neighbors[i], m_comm, &m_requests[req_count],
                                  tag);
        ++req_count;
      }
    }
    for (std::size_t f = 0; f < m_n_fields; ++f) {
      void *buf = static_cast<void *>(fields[f]);
      const int field_tag_off = m_base_tag + static_cast<int>(f) * 6;
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_neighbors[i] == m_rank) {
          continue;
        }
        const int tag = field_tag_off + static_cast<int>(i);
        pfc::exchange::isend_face(buf, m_face_types[i].send_type.get(),
                                  m_neighbors[i], m_comm, &m_requests[req_count],
                                  tag);
        ++req_count;
      }
    }
    pfc::exchange::wait_all(m_requests.data(), static_cast<int>(req_count));
  }

  void exchange_gpu_aware_corner_fill_(double *const *fields, cudaStream_t stream) {
    // Order: real-MPI faces first (narrow slabs); then self-faces with widened
    // slabs that pull from already-filled MPI halos to populate corners.

    // Step 1: post all Irecvs and Isends for real (non-self) faces.
    std::size_t req_count = 0;
    for (std::size_t f = 0; f < m_n_fields; ++f) {
      void *buf = static_cast<void *>(fields[f]);
      const int field_tag_off = m_base_tag + static_cast<int>(f) * 6;
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_neighbors[i] == m_rank) {
          continue;
        }
        const int tag = field_tag_off + opposite_slot_(static_cast<int>(i));
        pfc::exchange::irecv_face(buf, m_face_types[i].recv_type.get(),
                                  m_neighbors[i], m_comm, &m_requests[req_count],
                                  tag);
        ++req_count;
      }
    }
    for (std::size_t f = 0; f < m_n_fields; ++f) {
      void *buf = static_cast<void *>(fields[f]);
      const int field_tag_off = m_base_tag + static_cast<int>(f) * 6;
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_neighbors[i] == m_rank) {
          continue;
        }
        const int tag = field_tag_off + static_cast<int>(i);
        pfc::exchange::isend_face(buf, m_face_types[i].send_type.get(),
                                  m_neighbors[i], m_comm, &m_requests[req_count],
                                  tag);
        ++req_count;
      }
    }
    pfc::exchange::wait_all(m_requests.data(), static_cast<int>(req_count));

    // Step 2: MPI halos are now populated. Self-pack/unpack with the widened
    // slabs so the X halos (just arrived) propagate into ±Y / ±Z halo corners.
    if (m_any_self_neighbor) {
      if (m_d_scratch == nullptr) {
        throw std::runtime_error("BatchedPaddedDeviceHalo(corner_fill): "
                                 "self-neighbor halo needs scratch");
      }
      for (std::size_t f = 0; f < m_n_fields; ++f) {
        double *d_pad = fields[f];
        // Opposite-face unpack on widened slabs (same semantics as narrow path).
        for (std::size_t i = 0; i < 6; ++i) {
          if (!m_active[i] || m_neighbors[i] != m_rank) {
            continue;
          }
          const auto &send = m_face_specs_widened[i].first;
          const auto &recv =
              m_face_specs_widened[static_cast<std::size_t>(
                                      opposite_slot_(static_cast<int>(i)))]
                  .second;
          pfc::cuda::detail::launch_padded_pack_face(
              m_d_scratch, d_pad, send.ox, send.oy, send.oz, send.sx, send.sy,
              send.sz, m_nxp, m_nyp, m_nzp, stream);
          pfc::cuda::detail::launch_padded_unpack_face(
              d_pad, m_d_scratch, recv.ox, recv.oy, recv.oz, recv.sx, recv.sy,
              recv.sz, m_nxp, m_nyp, m_nzp, stream);
        }
      }
      pfc::cuda::detail::cuda_check(
          cudaStreamSynchronize(stream),
          "cudaStreamSynchronize after corner_fill self-neighbor halo copies");
    }
  }

  /// Self-face slab specs widened so packs along Y read full padded X (and Z slabs
  /// read full padded X and Y), letting MPI-filled X halos propagate into corners.
  /// Real-MPI faces (±X) keep the standard narrow specs; self-pack only uses the
  /// widened ones in `corner_fill` mode.
  static std::array<
      std::pair<pfc::cuda::detail::FaceSlabSpec, pfc::cuda::detail::FaceSlabSpec>, 6>
  make_widened_self_face_slabs_(int nx, int ny, int nz, int hw) {
    using S = pfc::cuda::detail::FaceSlabSpec;
    using P = std::pair<S, S>;
    const int X = nx + 2 * hw;
    const int Y = ny + 2 * hw;
    return {{
        // ±X stays narrow (used only for real-MPI here -- self ±X is unusual)
        P{S{nx, hw, hw, hw, ny, nz}, S{nx + hw, hw, hw, hw, ny, nz}}, // +X
        P{S{hw, hw, hw, hw, ny, nz}, S{0, hw, hw, hw, ny, nz}},       // -X
        // ±Y widened along X
        P{S{0, ny, hw, X, hw, nz}, S{0, ny + hw, hw, X, hw, nz}}, // +Y
        P{S{0, hw, hw, X, hw, nz}, S{0, 0, hw, X, hw, nz}},       // -Y
        // ±Z widened along X and Y
        P{S{0, 0, nz, X, Y, hw}, S{0, 0, nz + hw, X, Y, hw}}, // +Z
        P{S{0, 0, hw, X, Y, hw}, S{0, 0, 0, X, Y, hw}},       // -Z
    }};
  }

  int m_rank = 0;
  int m_halo_width = 1;
  MPI_Comm m_comm = MPI_COMM_NULL;
  int m_base_tag = 0;
  std::size_t m_n_fields = 0;

  int m_nxp = 0;
  int m_nyp = 0;
  int m_nzp = 0;

  std::array<
      std::pair<pfc::cuda::detail::FaceSlabSpec, pfc::cuda::detail::FaceSlabSpec>, 6>
      m_face_specs{};
  std::array<
      std::pair<pfc::cuda::detail::FaceSlabSpec, pfc::cuda::detail::FaceSlabSpec>, 6>
      m_face_specs_widened{};
  std::array<pfc::halo::FaceTypes, 6> m_face_types{};
  /** Per-slot membership in the active direction set
   *  (slot order +X,-X,+Y,-Y,+Z,-Z). Skipped slots never pack/Irecv/Isend. */
  std::array<bool, 6> m_active{};
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;

  std::array<std::size_t, 6> m_face_elems{};
  std::size_t m_scratch_elems = 0;

  bool m_use_gpu_aware = false;
  bool m_any_self_neighbor = false;
  bool m_corner_fill = false;
  pfc::halo::HaloDirectionSet m_dirs;

  double *m_d_scratch = nullptr;

  // Used only on the packed-fallback path -- one exchanger per field.
  std::vector<std::unique_ptr<pfc::cuda::PaddedDeviceHaloExchanger>>
      m_per_field_packed;
};

} // namespace kobayashi::cuda
