// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_device_halo_exchange.hpp
 * @brief Six-face padded-brick halo exchange with MPI into **device** buffers.
 *
 * @details
 * Uses the same MPI derived types as `pfc::PaddedHaloExchanger<double>` (see
 * `padded_halo_mpi_types.hpp`). Two modes:
 *
 * - **GPU-aware MPI** (`OpenPFC_MPI_CUDA_AWARE` at compile time and
 *   `MPIX_Query_cuda_support() == 1` at runtime on Open MPI): `MPI_Irecv` /
 *   `MPI_Isend` use the device pointer directly (after CUDA stream sync).
 * - **Packed fallback**: pack each send face into a contiguous device buffer,
 *   copy to pinned host, MPI on host, copy recv back and unpack into halos.
 *   Same-rank periodic faces use **device pack/unpack only** (no MPI-to-self),
 *   matching the GPU-aware path — important when **nz == 1** (±Z faces are full
 *   **nx×ny** slabs, not thin halos).
 *
 * Set **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`** to force the fallback path.
 *
 * **Self neighbors:** when the process grid has extent **1** along an axis (e.g.
 * **`nz = 1`** in the global world), periodic **±Z** neighbors are **the same MPI
 * rank**. GPU-aware **`MPI_Isend` / `MPI_Irecv` on device pointers to self** is
 * unreliable or extremely slow on some Open MPI + UCX stacks; this class applies a
 * **local device pack/unpack** for those faces and uses MPI only for **true
 * inter-rank** faces.
 *
 * **Profiling:** set **`OPENPFC_CUDA_PROFILE_HALO=1`** to accumulate CPU-side
 * `MPI_Wtime` buckets inside `exchange_halos_device` (pre-stream sync, MPI
 * window, post CUDA sync, and packed-path D2H / `MPI_Waitall` / H2D segments).
 * Call `pfc::cuda::print_cuda_halo_exchange_cpu_timers(comm)` on rank 0 after
 * the loop to print **`OPENPFC_CUDA_PROFILE_HALO_SUMMARY`** (MPI_MAX per bucket
 * across ranks).
 *
 * @see `kernel/decomposition/padded_halo_exchange.hpp` (CPU pointer path).
 */

#if defined(OpenPFC_ENABLE_CUDA)

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(OpenPFC_MPI_CUDA_AWARE) && defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#define OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT 1
#endif

#include <cuda_runtime.h>

#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/data/world_types.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>

namespace pfc::cuda {

namespace detail {

void launch_padded_pack_face(double *d_dst_contig, const double *d_pad, int ox,
                             int oy, int oz, int sx, int sy, int sz, int nxp,
                             int nyp, int nzp, cudaStream_t stream);

void launch_padded_unpack_face(double *d_pad, const double *d_src_contig, int ox,
                               int oy, int oz, int sx, int sy, int sz, int nxp,
                               int nyp, int nzp, cudaStream_t stream);

inline bool getenv_truthy(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] == '1';
}

inline bool runtime_mpi_cuda_aware() {
#if defined(OpenPFC_MPI_CUDA_AWARE) && defined(OPEN_MPI) &&                         \
    defined(OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT)
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
}

struct FaceSlabSpec {
  int ox = 0;
  int oy = 0;
  int oz = 0;
  int sx = 0;
  int sy = 0;
  int sz = 0;
};

/** Matches `create_padded_face_types_6` in `padded_halo_mpi_types.hpp`
 * (+X,-X,+Y,-Y,+Z,-Z). */
inline std::array<std::pair<FaceSlabSpec, FaceSlabSpec>, 6>
make_padded_face_slabs(int nx, int ny, int nz, int hw) {
  using P = std::pair<FaceSlabSpec, FaceSlabSpec>;
  return {{
      P{FaceSlabSpec{nx, hw, hw, hw, ny, nz},
        FaceSlabSpec{nx + hw, hw, hw, hw, ny, nz}}, // +X
      P{FaceSlabSpec{hw, hw, hw, hw, ny, nz},
        FaceSlabSpec{0, hw, hw, hw, ny, nz}}, // -X
      P{FaceSlabSpec{hw, ny, hw, nx, hw, nz},
        FaceSlabSpec{hw, ny + hw, hw, nx, hw, nz}}, // +Y
      P{FaceSlabSpec{hw, hw, hw, nx, hw, nz},
        FaceSlabSpec{hw, 0, hw, nx, hw, nz}}, // -Y
      P{FaceSlabSpec{hw, hw, nz, nx, ny, hw},
        FaceSlabSpec{hw, hw, nz + hw, nx, ny, hw}}, // +Z
      P{FaceSlabSpec{hw, hw, hw, nx, ny, hw},
        FaceSlabSpec{hw, hw, 0, nx, ny, hw}}, // -Z
  }};
}

inline void cuda_check(cudaError_t e, const char *what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

} // namespace detail

/** Wall-time buckets for `OPENPFC_CUDA_PROFILE_HALO=1` (CPU-side `MPI_Wtime`). */
struct CudaHaloExchangeCpuTimers {
  std::uint64_t n_calls = 0;
  double pre_stream_sync = 0;
  // GPU-aware path
  double gpu_aware_mpi = 0;
  /** `cudaDeviceSynchronize` (GPU-aware) or final `cudaStreamSynchronize` (packed).
   */
  double post_exchange_cuda_sync = 0;
  // Packed fallback: device pack + D2H + stream sync per face, then one MPI wait,
  // then H2D + sync + unpack per face
  double packed_face_pack_d2h_sync = 0;
  double packed_mpi_waitall = 0;
  double packed_face_h2d_unpack_sync = 0;
};

inline CudaHaloExchangeCpuTimers &cuda_halo_exchange_cpu_timers() {
  static CudaHaloExchangeCpuTimers t;
  return t;
}

inline bool cuda_halo_exchange_perf_enabled() {
  const char *v = std::getenv("OPENPFC_CUDA_PROFILE_HALO");
  return v != nullptr && v[0] == '1';
}

/** `MPI_MAX` across ranks of each bucket; rank 0 prints
 * `OPENPFC_CUDA_PROFILE_HALO_SUMMARY`. */
inline void print_cuda_halo_exchange_cpu_timers(MPI_Comm comm) {
  if (!cuda_halo_exchange_perf_enabled()) {
    return;
  }
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  auto &T = cuda_halo_exchange_cpu_timers();
  if (T.n_calls == 0) {
    return;
  }
  auto reduce_max = [&](double local) {
    double mx = 0;
    MPI_Reduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    return (rank == 0) ? mx : 0.0;
  };
  std::uint64_t nloc = T.n_calls;
  std::uint64_t nmax = 0;
  MPI_Reduce(&nloc, &nmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm);

  const double mx_pre = reduce_max(T.pre_stream_sync);
  const double mx_gw = reduce_max(T.gpu_aware_mpi);
  const double mx_post = reduce_max(T.post_exchange_cuda_sync);
  const double mx_pk = reduce_max(T.packed_face_pack_d2h_sync);
  const double mx_mpi = reduce_max(T.packed_mpi_waitall);
  const double mx_uu = reduce_max(T.packed_face_h2d_unpack_sync);
  const double mx_total = mx_pre + mx_gw + mx_post + mx_pk + mx_mpi + mx_uu;

  if (rank == 0) {
    std::cout << std::setprecision(17);
    std::cout << "OPENPFC_CUDA_PROFILE_HALO_SUMMARY"
              << " n_exchange_calls_max=" << nmax << " wall_s_max_per_rank{"
              << "pre_stream_sync=" << mx_pre << " gpu_aware_mpi=" << mx_gw
              << " post_exchange_cuda_sync=" << mx_post
              << " packed_pack_d2h_sync=" << mx_pk
              << " packed_mpi_waitall=" << mx_mpi
              << " packed_h2d_unpack_sync=" << mx_uu
              << " total_halo_cpu_wall_s=" << mx_total << "}\n";
  }
}

/**
 * @brief MPI halo exchange for a `PaddedBrick`-layout buffer on the CUDA device.
 *
 * Non-copyable; tie lifetime to the owning rank's padded device allocations.
 */
class PaddedDeviceHaloExchanger {
public:
  using Int3 = pfc::types::Int3;

  /**
   * @brief Construct with the historical 6-face axis-aligned set (`Axes3D()`).
   */
  PaddedDeviceHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                            int halo_width, MPI_Comm comm, int base_tag = 0)
      : PaddedDeviceHaloExchanger(decomp, rank, halo_width, comm,
                                  halo::presets::Axes3D(), base_tag,
                                  halo::HaloDirectionSelector{}) {}

  /**
   * @brief Construct with a user-selected halo direction set.
   *
   * Both the GPU-aware and packed branches skip excluded slots; same-rank
   * periodic faces inside the active set continue to use device pack/unpack
   * (no MPI-to-self).
   *
   * Non-face directions are tolerated but ignored — this is a face-only
   * exchanger. Use `FullPaddedDeviceHalo` for 26-direction fills.
   *
   * @param dirs     Direction set (defaults to `Axes3D()` for back-compat).
   * @param selector Optional per-rank override of the direction set.
   */
  PaddedDeviceHaloExchanger(const decomposition::Decomposition &decomp, int rank,
                            int halo_width, MPI_Comm comm,
                            halo::HaloDirectionSet dirs, int base_tag = 0,
                            halo::HaloDirectionSelector selector = {})
      : m_decomp(decomp), m_rank(rank), m_halo_width(halo_width), m_comm(comm),
        m_base_tag(base_tag),
        m_dirs(halo::resolve_direction_set(dirs, selector, rank)) {
    const auto &local_world = decomposition::get_subworld(m_decomp, m_rank);
    const auto local_size = pfc::world::get_size(local_world);
    const int nx = local_size[0];
    const int ny = local_size[1];
    const int nz = local_size[2];
    const int hw = m_halo_width;

    m_nxp = nx + 2 * hw;
    m_nyp = ny + 2 * hw;
    m_nzp = nz + 2 * hw;

    m_face_specs = detail::make_padded_face_slabs(nx, ny, nz, hw);

    m_face_types = halo::create_padded_face_types_6(
        nx, ny, nz, m_halo_width, exchange::detail::get_mpi_type<double>());

    const std::array<Int3, 6> dirs_canon{Int3{1, 0, 0}, Int3{-1, 0, 0},
                                         Int3{0, 1, 0}, Int3{0, -1, 0},
                                         Int3{0, 0, 1}, Int3{0, 0, -1}};
    m_neighbors.clear();
    for (std::size_t i = 0; i < 6; ++i) {
      m_active[i] = m_dirs.contains(dirs_canon[i]);
      m_neighbors.push_back(
          decomposition::get_neighbor_rank(m_decomp, m_rank, dirs_canon[i]));
    }
    m_requests.resize(2 * 6);

    m_scratch_elems = 0;
    for (std::size_t i = 0; i < 6; ++i) {
      const auto &send = m_face_specs[i].first;
      const std::size_t c = static_cast<std::size_t>(send.sx) *
                            static_cast<std::size_t>(send.sy) *
                            static_cast<std::size_t>(send.sz);
      m_face_elems[i] = c;
      // Only consider active slots when sizing scratch — excluded slots will
      // never pack/unpack so their footprint is irrelevant.
      if (m_active[i]) {
        m_scratch_elems = std::max(m_scratch_elems, c);
      }
    }

    const bool force_packed =
        detail::getenv_truthy("OPENPFC_CUDA_FORCE_PACKED_HALO");
    m_use_gpu_aware = !force_packed && detail::runtime_mpi_cuda_aware();

    m_any_self_neighbor = false;
    for (std::size_t i = 0; i < 6; ++i) {
      if (m_active[i] && m_neighbors[i] == m_rank) {
        m_any_self_neighbor = true;
        break;
      }
    }

    if (!m_use_gpu_aware) {
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_face_elems[i] == 0) {
          continue;
        }
        const std::size_t bytes = m_face_elems[i] * sizeof(double);
        detail::cuda_check(
            cudaMallocHost(reinterpret_cast<void **>(&m_h_send[i]), bytes),
            "cudaMallocHost halo send");
        detail::cuda_check(
            cudaMallocHost(reinterpret_cast<void **>(&m_h_recv[i]), bytes),
            "cudaMallocHost halo recv");
      }
    }
    if (m_scratch_elems > 0) {
      detail::cuda_check(cudaMalloc(reinterpret_cast<void **>(&m_d_scratch),
                                    m_scratch_elems * sizeof(double)),
                         "cudaMalloc halo device scratch (pack/unpack)");
    }
  }

  PaddedDeviceHaloExchanger(const PaddedDeviceHaloExchanger &) = delete;
  PaddedDeviceHaloExchanger &operator=(const PaddedDeviceHaloExchanger &) = delete;

  ~PaddedDeviceHaloExchanger() { cleanup(); }

  [[nodiscard]] bool uses_gpu_aware_mpi() const { return m_use_gpu_aware; }

  /** Read-only access to the active direction set (after `selector` resolution). */
  [[nodiscard]] const halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

  void exchange_halos_device(double *d_padded, std::size_t padded_size,
                             cudaStream_t stream = nullptr) {
    (void)padded_size;
    const bool perf = cuda_halo_exchange_perf_enabled();
    auto &H = cuda_halo_exchange_cpu_timers();

    double t_mark = MPI_Wtime();
    detail::cuda_check(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize pre halo");
    if (perf) {
      H.pre_stream_sync += MPI_Wtime() - t_mark;
      ++H.n_calls;
    }

    const double t0 = MPI_Wtime();
    if (m_use_gpu_aware) {
      t_mark = MPI_Wtime();
      exchange_gpu_aware_(d_padded, stream);
      if (perf) {
        H.gpu_aware_mpi += MPI_Wtime() - t_mark;
      }
      t_mark = MPI_Wtime();
      detail::cuda_check(cudaDeviceSynchronize(),
                         "cudaDeviceSynchronize post GPU-aware MPI");
      if (perf) {
        H.post_exchange_cuda_sync += MPI_Wtime() - t_mark;
      }
    } else {
      exchange_packed_fallback_(d_padded, stream);
      t_mark = MPI_Wtime();
      detail::cuda_check(cudaStreamSynchronize(stream),
                         "cudaStreamSynchronize post packed halo");
      if (perf) {
        H.post_exchange_cuda_sync += MPI_Wtime() - t_mark;
      }
    }
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
  }

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

  /**
   * GPU-aware halo: `MPI_Irecv` / `MPI_Isend` with **count = 1** and face types from
   * `MPI_Type_create_subarray` (`halo_mpi_types.hpp`). That datatype describes a
   * **non-contiguous** slab inside the padded brick. Application code does **not**
   * call `cudaMemcpy` here; transfers are meant to be performed by the MPI/CUDA
   * transport. In practice **Open MPI + UCX** often realize non-contiguous
   * **device** datatypes as **many small driver copies** (`cuMemcpyAsync`) and
   * matching syncs, which Nsight reports as huge **Num Calls** even when total moved
   * bytes are
   * \(O(\text{face})\). The **packed** path (`exchange_packed_fallback_`) gathers
   * each face to a **contiguous** buffer first, then uses a **small number** of
   * `cudaMemcpyAsync` calls per face — set **`OPENPFC_CUDA_FORCE_PACKED_HALO=1`**
   * to force that path for performance experiments.
   */
  void exchange_gpu_aware_(double *d_padded, cudaStream_t stream) {
    void *buf = static_cast<void *>(d_padded);
    // Same-rank periodic faces (e.g. ±Z when the MPI grid is one cell thick in Z):
    // avoid GPU-aware MPI to self — use device pack/unpack instead.
    if (m_any_self_neighbor) {
      if (m_d_scratch == nullptr || m_scratch_elems == 0) {
        throw std::runtime_error("PaddedDeviceHaloExchanger: self-neighbor halo "
                                 "needs non-zero device scratch");
      }
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_neighbors[i] != m_rank) {
          continue;
        }
        const auto &send = m_face_specs[i].first;
        const auto &recv = m_face_specs[i].second;
        detail::launch_padded_pack_face(m_d_scratch, d_padded, send.ox, send.oy,
                                        send.oz, send.sx, send.sy, send.sz, m_nxp,
                                        m_nyp, m_nzp, stream);
        detail::launch_padded_unpack_face(d_padded, m_d_scratch, recv.ox, recv.oy,
                                          recv.oz, recv.sx, recv.sy, recv.sz, m_nxp,
                                          m_nyp, m_nzp, stream);
      }
      detail::cuda_check(
          cudaStreamSynchronize(stream),
          "cudaStreamSynchronize after local self-neighbor halo copies");
    }

    std::size_t req_count = 0;
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const int tag = m_base_tag + opposite_slot(static_cast<int>(i));
      exchange::irecv_face(buf, m_face_types[i].recv_type.get(), m_neighbors[i],
                           m_comm, &m_requests[req_count], tag);
      ++req_count;
    }
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const int tag = m_base_tag + static_cast<int>(i);
      exchange::isend_face(buf, m_face_types[i].send_type.get(), m_neighbors[i],
                           m_comm, &m_requests[req_count], tag);
      ++req_count;
    }
    exchange::wait_all(m_requests.data(), static_cast<int>(req_count));
  }

  void exchange_packed_fallback_(double *d_padded, cudaStream_t stream) {
    const bool perf = cuda_halo_exchange_perf_enabled();
    auto &H = cuda_halo_exchange_cpu_timers();

    // Mirror **exchange_gpu_aware_**: periodic faces whose neighbor is **this rank**
    // must not use MPI (especially ±Z when **nz == 1**, where each face is **nx×ny**
    // doubles — MPI-to-self would move ~128 MiB per face per exchange). Use device
    // pack/unpack into **m_d_scratch** instead.
    if (m_any_self_neighbor) {
      if (m_d_scratch == nullptr || m_scratch_elems == 0) {
        throw std::runtime_error("PaddedDeviceHaloExchanger: packed self-neighbor "
                                 "halo needs non-zero device scratch");
      }
      for (std::size_t i = 0; i < 6; ++i) {
        if (!m_active[i] || m_neighbors[i] != m_rank) {
          continue;
        }
        const auto &send = m_face_specs[i].first;
        const auto &recv = m_face_specs[i].second;
        detail::launch_padded_pack_face(m_d_scratch, d_padded, send.ox, send.oy,
                                        send.oz, send.sx, send.sy, send.sz, m_nxp,
                                        m_nyp, m_nzp, stream);
        detail::launch_padded_unpack_face(d_padded, m_d_scratch, recv.ox, recv.oy,
                                          recv.oz, recv.sx, recv.sy, recv.sz, m_nxp,
                                          m_nyp, m_nzp, stream);
      }
      detail::cuda_check(
          cudaStreamSynchronize(stream),
          "cudaStreamSynchronize after packed self-neighbor halo copies");
    }

    std::size_t req_count = 0;
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const int tag = m_base_tag + opposite_slot(static_cast<int>(i));
      MPI_Irecv(m_h_recv[i], static_cast<int>(m_face_elems[i]), MPI_DOUBLE,
                m_neighbors[i], tag, m_comm, &m_requests[req_count]);
      ++req_count;
    }

    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const double t_face = perf ? MPI_Wtime() : 0.0;
      const auto &send = m_face_specs[i].first;
      detail::launch_padded_pack_face(m_d_scratch, d_padded, send.ox, send.oy,
                                      send.oz, send.sx, send.sy, send.sz, m_nxp,
                                      m_nyp, m_nzp, stream);
      detail::cuda_check(cudaMemcpyAsync(m_h_send[i], m_d_scratch,
                                         m_face_elems[i] * sizeof(double),
                                         cudaMemcpyDeviceToHost, stream),
                         "cudaMemcpyAsync pack face D2H");
      detail::cuda_check(cudaStreamSynchronize(stream),
                         "cudaStreamSynchronize pack face");
      if (perf) {
        H.packed_face_pack_d2h_sync += MPI_Wtime() - t_face;
      }

      const int tag = m_base_tag + static_cast<int>(i);
      MPI_Isend(m_h_send[i], static_cast<int>(m_face_elems[i]), MPI_DOUBLE,
                m_neighbors[i], tag, m_comm, &m_requests[req_count]);
      ++req_count;
    }

    const double t_mpi = perf ? MPI_Wtime() : 0.0;
    exchange::wait_all(m_requests.data(), static_cast<int>(req_count));
    if (perf) {
      H.packed_mpi_waitall += MPI_Wtime() - t_mpi;
    }

    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const double t_face = perf ? MPI_Wtime() : 0.0;
      detail::cuda_check(cudaMemcpyAsync(m_d_scratch, m_h_recv[i],
                                         m_face_elems[i] * sizeof(double),
                                         cudaMemcpyHostToDevice, stream),
                         "cudaMemcpyAsync unpack face H2D");
      detail::cuda_check(cudaStreamSynchronize(stream),
                         "cudaStreamSynchronize unpack H2D");

      const auto &recv = m_face_specs[i].second;
      detail::launch_padded_unpack_face(d_padded, m_d_scratch, recv.ox, recv.oy,
                                        recv.oz, recv.sx, recv.sy, recv.sz, m_nxp,
                                        m_nyp, m_nzp, stream);
      if (perf) {
        H.packed_face_h2d_unpack_sync += MPI_Wtime() - t_face;
      }
    }
  }

  void cleanup() {
    if (!m_use_gpu_aware) {
      for (std::size_t i = 0; i < 6; ++i) {
        if (m_h_send[i] != nullptr) {
          (void)cudaFreeHost(m_h_send[i]);
          m_h_send[i] = nullptr;
        }
        if (m_h_recv[i] != nullptr) {
          (void)cudaFreeHost(m_h_recv[i]);
          m_h_recv[i] = nullptr;
        }
      }
    }
    if (m_d_scratch != nullptr) {
      (void)cudaFree(m_d_scratch);
      m_d_scratch = nullptr;
    }
  }

  const decomposition::Decomposition &m_decomp;
  int m_rank = 0;
  int m_halo_width = 1;
  MPI_Comm m_comm = MPI_COMM_NULL;
  int m_base_tag = 0;
  halo::HaloDirectionSet m_dirs;

  int m_nxp = 0;
  int m_nyp = 0;
  int m_nzp = 0;

  std::array<std::pair<detail::FaceSlabSpec, detail::FaceSlabSpec>, 6>
      m_face_specs{};
  std::array<halo::FaceTypes, 6> m_face_types{};
  /** Per-slot membership in the active direction set (slot order
   *  +X,-X,+Y,-Y,+Z,-Z). Inactive slots are skipped in both the GPU-aware and
   *  packed branches. */
  std::array<bool, 6> m_active{};
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;

  std::array<std::size_t, 6> m_face_elems{};
  std::size_t m_scratch_elems = 0;

  bool m_use_gpu_aware = false;
  /** True if any of the six face neighbors equals **m_rank** (periodic self-link).
   */
  bool m_any_self_neighbor = false;
  std::array<double *, 6> m_h_send{};
  std::array<double *, 6> m_h_recv{};
  double *m_d_scratch = nullptr;
};

} // namespace pfc::cuda

#endif // OpenPFC_ENABLE_CUDA
