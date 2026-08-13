// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file padded_device_halo_exchange_gpu.hpp
 * @brief Single-source six-face padded halo exchange into device buffers (M3).
 *
 * Stamps `pfc::cuda::PaddedDeviceHaloExchanger` and/or
 * `pfc::hip::PaddedDeviceHaloExchanger`. Vendor headers are thin includes of
 * this file. Per-tag runtime calls use the native CUDA/HIP API (not
 * `gpu_api.hpp`) so a co-enabled translation unit can own both classes.
 *
 * HIP's Field-based constructor / `exchange_halos_device` overloads are
 * stamped for CUDA as well (`pfc::data::Field<T, CudaSpace>`).
 *
 * Env and timer names stay vendor-specific (`OPENPFC_CUDA_*` /
 * `OPENPFC_HIP_*`, `print_cuda_halo_exchange_cpu_timers` /
 * `print_hip_halo_exchange_cpu_timers`).
 *
 * @see kernel/decomposition/padded_halo_exchange.hpp (CPU pointer path)
 */

#if defined(OpenPFC_ENABLE_CUDA) || defined(OpenPFC_ENABLE_HIP)

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/padded_halo_mpi_types.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/profiling/context.hpp>
#include <openpfc/kernel/profiling/names.hpp>
#include <openpfc/runtime/gpu/gpu_check.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

#if defined(OpenPFC_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif
#if defined(OpenPFC_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#if (defined(OpenPFC_MPI_CUDA_AWARE) || defined(OpenPFC_MPI_HIP_AWARE)) &&          \
    defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#if defined(OpenPFC_MPI_CUDA_AWARE)
#ifndef OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT 1
#endif
#endif
#if defined(OpenPFC_MPI_HIP_AWARE)
#ifndef OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT
#define OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT 1
#endif
#endif
#endif

namespace pfc::gpu::detail {

inline bool getenv_truthy(const char *name) {
  const char *v = std::getenv(name);
  return v != nullptr && v[0] == '1';
}

inline int checked_padded_extent(const char *kind, int n, int hw) {
  if (hw < 0) {
    throw std::invalid_argument(
        std::string(kind) +
        " PaddedDeviceHaloExchanger: halo width must be non-negative (got " +
        std::to_string(hw) + ")");
  }
  const long long result =
      static_cast<long long>(n) + 2LL * static_cast<long long>(hw);
  if (result > static_cast<long long>(std::numeric_limits<int>::max()) ||
      result < static_cast<long long>(std::numeric_limits<int>::min())) {
    throw std::overflow_error(
        std::string(kind) + " PaddedDeviceHaloExchanger: padded extent overflow " +
        std::to_string(n) + " + 2*" + std::to_string(hw) + " exceeds int range");
  }
  return static_cast<int>(result);
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

} // namespace pfc::gpu::detail

#if defined(OpenPFC_ENABLE_CUDA)
namespace pfc::cuda::detail {

using ::pfc::gpu::detail::FaceSlabSpec;
using ::pfc::gpu::detail::getenv_truthy;
using ::pfc::gpu::detail::make_padded_face_slabs;

void launch_padded_pack_face(double *d_dst_contig, const double *d_pad, int ox,
                             int oy, int oz, int sx, int sy, int sz, int nxp,
                             int nyp, int nzp, cudaStream_t stream);

void launch_padded_unpack_face(double *d_pad, const double *d_src_contig, int ox,
                               int oy, int oz, int sx, int sy, int sz, int nxp,
                               int nyp, int nzp, cudaStream_t stream);

inline bool runtime_mpi_cuda_aware() {
#if defined(OpenPFC_MPI_CUDA_AWARE) && defined(OPEN_MPI) &&                         \
    defined(OPENPFC_HAVE_MPIX_QUERY_CUDA_SUPPORT)
  return MPIX_Query_cuda_support() == 1;
#else
  return false;
#endif
}

} // namespace pfc::cuda::detail

namespace pfc::cuda {

/** Wall-time buckets for `OPENPFC_CUDA_PROFILE_HALO=1` (CPU-side `MPI_Wtime`). */
struct CudaHaloExchangeCpuTimers {
  std::uint64_t n_calls = 0;
  double pre_stream_sync = 0;
  double gpu_aware_mpi = 0;
  /** `cudaDeviceSynchronize` (GPU-aware) or final `cudaStreamSynchronize` (packed).
   */
  double post_exchange_cuda_sync = 0;
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

struct CudaHaloOps {
  using stream_t = cudaStream_t;
  using space = pfc::CudaSpace;
  using timers_t = CudaHaloExchangeCpuTimers;
  using error_t = cudaError_t;

  static constexpr const char *kind = "CUDA";
  static constexpr const char *force_packed_env = "OPENPFC_CUDA_FORCE_PACKED_HALO";
  static constexpr const char *malloc_host_send = "cudaMallocHost halo send";
  static constexpr const char *malloc_host_recv = "cudaMallocHost halo recv";
  static constexpr const char *malloc_scratch =
      "cudaMalloc halo device scratch (pack/unpack)";
  static constexpr const char *sync_pre = "cudaStreamSynchronize pre halo";
  static constexpr const char *sync_post_aware =
      "cudaDeviceSynchronize post GPU-aware MPI";
  static constexpr const char *sync_post_packed =
      "cudaStreamSynchronize post packed halo";
  static constexpr const char *sync_self =
      "cudaStreamSynchronize after local self-neighbor halo copies";
  static constexpr const char *sync_self_packed =
      "cudaStreamSynchronize after packed self-neighbor halo copies";
  static constexpr const char *memcpy_d2h = "cudaMemcpyAsync pack face D2H";
  static constexpr const char *sync_pack = "cudaStreamSynchronize pack face";
  static constexpr const char *memcpy_h2d = "cudaMemcpyAsync unpack face H2D";
  static constexpr const char *sync_unpack = "cudaStreamSynchronize unpack H2D";
  static constexpr const char *malloc_full_scratch =
      "cudaMalloc full halo device scratch";
  static constexpr const char *sync_pre_full =
      "cudaStreamSynchronize pre full halo";
  static constexpr const char *sync_after_full_pass =
      "cudaDeviceSynchronize after full halo pass";
  static constexpr const char *sync_after_full_self =
      "cudaStreamSynchronize after full halo self-pack";
  static constexpr const char *print_rank_what =
      "MPI_Comm_rank in print_cuda_halo_exchange_cpu_timers";
  static constexpr const char *print_reduce_what =
      "MPI_Reduce in print_cuda_halo_exchange_cpu_timers (reduce_max)";
  static constexpr const char *print_ncalls_what =
      "MPI_Reduce in print_cuda_halo_exchange_cpu_timers (n_calls)";
  static constexpr const char *profile_summary = "OPENPFC_CUDA_PROFILE_HALO_SUMMARY";
  static constexpr const char *post_sync_key = "post_exchange_cuda_sync";

  static bool mpi_aware() { return detail::runtime_mpi_cuda_aware(); }
  static bool perf_enabled() { return cuda_halo_exchange_perf_enabled(); }
  static timers_t &timers() { return cuda_halo_exchange_cpu_timers(); }
  static double &post_sync(timers_t &t) { return t.post_exchange_cuda_sync; }

  static void check(error_t e, const char *what) { detail::cuda_check(e, what); }
  static void malloc_dev(void **p, std::size_t bytes, const char *what) {
    check(cudaMalloc(p, bytes), what);
  }
  static void malloc_host(void **p, std::size_t bytes, const char *what) {
    check(cudaMallocHost(p, bytes), what);
  }
  static void free_dev(void *p) { (void)cudaFree(p); }
  static void free_host(void *p) { (void)cudaFreeHost(p); }
  static void memcpy_async_d2h(void *dst, const void *src, std::size_t bytes,
                               stream_t stream, const char *what) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), what);
  }
  static void memcpy_async_h2d(void *dst, const void *src, std::size_t bytes,
                               stream_t stream, const char *what) {
    check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), what);
  }
  static void stream_sync(stream_t stream, const char *what) {
    check(cudaStreamSynchronize(stream), what);
  }
  static void device_sync(const char *what) {
    check(cudaDeviceSynchronize(), what);
  }
  static void pack_face(double *d_dst, const double *d_pad, int ox, int oy, int oz,
                        int sx, int sy, int sz, int nxp, int nyp, int nzp,
                        stream_t stream) {
    detail::launch_padded_pack_face(d_dst, d_pad, ox, oy, oz, sx, sy, sz, nxp, nyp,
                                    nzp, stream);
  }
  static void unpack_face(double *d_pad, const double *d_src, int ox, int oy,
                          int oz, int sx, int sy, int sz, int nxp, int nyp, int nzp,
                          stream_t stream) {
    detail::launch_padded_unpack_face(d_pad, d_src, ox, oy, oz, sx, sy, sz, nxp, nyp,
                                      nzp, stream);
  }
};

} // namespace pfc::cuda
#endif // OpenPFC_ENABLE_CUDA

#if defined(OpenPFC_ENABLE_HIP)
namespace pfc::hip::detail {

using ::pfc::gpu::detail::FaceSlabSpec;
using ::pfc::gpu::detail::getenv_truthy;
using ::pfc::gpu::detail::make_padded_face_slabs;

void launch_padded_pack_face(double *d_dst_contig, const double *d_pad, int ox,
                             int oy, int oz, int sx, int sy, int sz, int nxp,
                             int nyp, int nzp, hipStream_t stream);

void launch_padded_unpack_face(double *d_pad, const double *d_src_contig, int ox,
                               int oy, int oz, int sx, int sy, int sz, int nxp,
                               int nyp, int nzp, hipStream_t stream);

inline bool runtime_mpi_hip_aware() {
#if defined(OpenPFC_MPI_HIP_AWARE) && defined(OPEN_MPI) &&                          \
    defined(OPENPFC_HAVE_MPIX_QUERY_HIP_SUPPORT)
  return MPIX_Query_hip_support() == 1;
#else
  return false;
#endif
}

} // namespace pfc::hip::detail

namespace pfc::hip {

/** Wall-time buckets for `OPENPFC_HIP_PROFILE_HALO=1` (CPU-side `MPI_Wtime`). */
struct HipHaloExchangeCpuTimers {
  std::uint64_t n_calls = 0;
  double pre_stream_sync = 0;
  double gpu_aware_mpi = 0;
  /** `hipDeviceSynchronize` (GPU-aware) or final `hipStreamSynchronize` (packed).
   */
  double post_exchange_hip_sync = 0;
  double packed_face_pack_d2h_sync = 0;
  double packed_mpi_waitall = 0;
  double packed_face_h2d_unpack_sync = 0;
};

inline HipHaloExchangeCpuTimers &hip_halo_exchange_cpu_timers() {
  static HipHaloExchangeCpuTimers t;
  return t;
}

inline bool hip_halo_exchange_perf_enabled() {
  const char *v = std::getenv("OPENPFC_HIP_PROFILE_HALO");
  return v != nullptr && v[0] == '1';
}

struct HipHaloOps {
  using stream_t = hipStream_t;
  using space = pfc::HipSpace;
  using timers_t = HipHaloExchangeCpuTimers;
  using error_t = hipError_t;

  static constexpr const char *kind = "HIP";
  static constexpr const char *force_packed_env = "OPENPFC_HIP_FORCE_PACKED_HALO";
  static constexpr const char *malloc_host_send = "hipHostMalloc halo send";
  static constexpr const char *malloc_host_recv = "hipHostMalloc halo recv";
  static constexpr const char *malloc_scratch =
      "hipMalloc halo device scratch (pack/unpack)";
  static constexpr const char *sync_pre = "hipStreamSynchronize pre halo";
  static constexpr const char *sync_post_aware =
      "hipDeviceSynchronize post GPU-aware MPI";
  static constexpr const char *sync_post_packed =
      "hipStreamSynchronize post packed halo";
  static constexpr const char *sync_self =
      "hipStreamSynchronize after local self-neighbor halo copies";
  static constexpr const char *sync_self_packed =
      "hipStreamSynchronize after packed self-neighbor halo copies";
  static constexpr const char *memcpy_d2h = "hipMemcpyAsync pack face D2H";
  static constexpr const char *sync_pack = "hipStreamSynchronize pack face";
  static constexpr const char *memcpy_h2d = "hipMemcpyAsync unpack face H2D";
  static constexpr const char *sync_unpack = "hipStreamSynchronize unpack H2D";
  static constexpr const char *malloc_full_scratch =
      "hipMalloc full halo device scratch";
  static constexpr const char *sync_pre_full = "hipStreamSynchronize pre full halo";
  static constexpr const char *sync_after_full_pass =
      "hipDeviceSynchronize after full halo pass";
  static constexpr const char *sync_after_full_self =
      "hipStreamSynchronize after full halo self-pack";
  static constexpr const char *print_rank_what =
      "MPI_Comm_rank in print_hip_halo_exchange_cpu_timers";
  static constexpr const char *print_reduce_what =
      "MPI_Reduce in print_hip_halo_exchange_cpu_timers (reduce_max)";
  static constexpr const char *print_ncalls_what =
      "MPI_Reduce in print_hip_halo_exchange_cpu_timers (n_calls)";
  static constexpr const char *profile_summary = "OPENPFC_HIP_PROFILE_HALO_SUMMARY";
  static constexpr const char *post_sync_key = "post_exchange_hip_sync";

  static bool mpi_aware() { return detail::runtime_mpi_hip_aware(); }
  static bool perf_enabled() { return hip_halo_exchange_perf_enabled(); }
  static timers_t &timers() { return hip_halo_exchange_cpu_timers(); }
  static double &post_sync(timers_t &t) { return t.post_exchange_hip_sync; }

  static void check(error_t e, const char *what) { detail::hip_check(e, what); }
  static void malloc_dev(void **p, std::size_t bytes, const char *what) {
    check(hipMalloc(p, bytes), what);
  }
  static void malloc_host(void **p, std::size_t bytes, const char *what) {
    check(hipHostMalloc(p, bytes, hipHostMallocDefault), what);
  }
  static void free_dev(void *p) { (void)hipFree(p); }
  static void free_host(void *p) { (void)hipHostFree(p); }
  static void memcpy_async_d2h(void *dst, const void *src, std::size_t bytes,
                               stream_t stream, const char *what) {
    check(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream), what);
  }
  static void memcpy_async_h2d(void *dst, const void *src, std::size_t bytes,
                               stream_t stream, const char *what) {
    check(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream), what);
  }
  static void stream_sync(stream_t stream, const char *what) {
    check(hipStreamSynchronize(stream), what);
  }
  static void device_sync(const char *what) { check(hipDeviceSynchronize(), what); }
  static void pack_face(double *d_dst, const double *d_pad, int ox, int oy, int oz,
                        int sx, int sy, int sz, int nxp, int nyp, int nzp,
                        stream_t stream) {
    detail::launch_padded_pack_face(d_dst, d_pad, ox, oy, oz, sx, sy, sz, nxp, nyp,
                                    nzp, stream);
  }
  static void unpack_face(double *d_pad, const double *d_src, int ox, int oy,
                          int oz, int sx, int sy, int sz, int nxp, int nyp, int nzp,
                          stream_t stream) {
    detail::launch_padded_unpack_face(d_pad, d_src, ox, oy, oz, sx, sy, sz, nxp, nyp,
                                      nzp, stream);
  }
};

} // namespace pfc::hip
#endif // OpenPFC_ENABLE_HIP

namespace pfc::gpu {

template <typename Ops>
void print_halo_exchange_cpu_timers(MPI_Comm comm) {
  if (!Ops::perf_enabled()) {
    return;
  }
  int rank = 0;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_rank(comm, &rank), Ops::print_rank_what);
  auto &T = Ops::timers();
  if (T.n_calls == 0) {
    return;
  }
  auto reduce_max = [&](double local) {
    double mx = 0;
    pfc::mpi::throw_on_mpi_error(
        MPI_Reduce(&local, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, comm),
        Ops::print_reduce_what);
    return (rank == 0) ? mx : 0.0;
  };
  std::uint64_t nloc = T.n_calls;
  std::uint64_t nmax = 0;
  pfc::mpi::throw_on_mpi_error(
      MPI_Reduce(&nloc, &nmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm),
      Ops::print_ncalls_what);

  const double mx_pre = reduce_max(T.pre_stream_sync);
  const double mx_gw = reduce_max(T.gpu_aware_mpi);
  const double mx_post = reduce_max(Ops::post_sync(T));
  const double mx_pk = reduce_max(T.packed_face_pack_d2h_sync);
  const double mx_mpi = reduce_max(T.packed_mpi_waitall);
  const double mx_uu = reduce_max(T.packed_face_h2d_unpack_sync);
  const double mx_total = mx_pre + mx_gw + mx_post + mx_pk + mx_mpi + mx_uu;

  if (rank == 0) {
    std::cout << std::setprecision(17);
    std::cout << Ops::profile_summary << " n_exchange_calls_max=" << nmax
              << " wall_s_max_per_rank{"
              << "pre_stream_sync=" << mx_pre << " gpu_aware_mpi=" << mx_gw << " "
              << Ops::post_sync_key << "=" << mx_post
              << " packed_pack_d2h_sync=" << mx_pk
              << " packed_mpi_waitall=" << mx_mpi
              << " packed_h2d_unpack_sync=" << mx_uu
              << " total_halo_cpu_wall_s=" << mx_total << "}\n";
  }
}

/**
 * @brief MPI halo exchange for a padded Field-layout buffer on the device.
 *
 * Non-copyable; tie lifetime to the owning rank's padded device allocations.
 * Field overloads are the primary API; raw-pointer overloads remain for
 * backward compatibility.
 */
template <typename Ops>
class PaddedDeviceHaloExchangerImpl {
public:
  using Int3 = pfc::types::Int3;
  using stream_t = typename Ops::stream_t;

  PaddedDeviceHaloExchangerImpl(const decomposition::Decomposition &decomp, int rank,
                                int halo_width, MPI_Comm comm, int base_tag = 0)
      : PaddedDeviceHaloExchangerImpl(decomp, rank, halo_width, comm,
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
   */
  PaddedDeviceHaloExchangerImpl(const decomposition::Decomposition &decomp, int rank,
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

    m_nxp = pfc::gpu::detail::checked_padded_extent(Ops::kind, nx, hw);
    m_nyp = pfc::gpu::detail::checked_padded_extent(Ops::kind, ny, hw);
    m_nzp = pfc::gpu::detail::checked_padded_extent(Ops::kind, nz, hw);

    m_face_specs = pfc::gpu::detail::make_padded_face_slabs(nx, ny, nz, hw);

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
      if (m_active[i]) {
        m_scratch_elems = std::max(m_scratch_elems, c);
      }
    }

    const bool force_packed = pfc::gpu::detail::getenv_truthy(Ops::force_packed_env);
    m_use_gpu_aware = !force_packed && Ops::mpi_aware();

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
        Ops::malloc_host(reinterpret_cast<void **>(&m_h_send[i]), bytes,
                         Ops::malloc_host_send);
        Ops::malloc_host(reinterpret_cast<void **>(&m_h_recv[i]), bytes,
                         Ops::malloc_host_recv);
      }
    }
    if (m_scratch_elems > 0) {
      Ops::malloc_dev(reinterpret_cast<void **>(&m_d_scratch),
                      m_scratch_elems * sizeof(double), Ops::malloc_scratch);
    }
  }

  template <typename T>
  PaddedDeviceHaloExchangerImpl(const pfc::data::Field<T, typename Ops::space> &field,
                                const decomposition::Decomposition &decomp, int rank,
                                MPI_Comm comm,
                                halo::HaloDirectionSet dirs = halo::presets::Axes3D(),
                                int base_tag = 0,
                                halo::HaloDirectionSelector selector = {})
      : PaddedDeviceHaloExchangerImpl(decomp, rank, field.halo_width(), comm, dirs,
                                      base_tag, selector) {
    (void)field;
  }

  PaddedDeviceHaloExchangerImpl(const PaddedDeviceHaloExchangerImpl &) = delete;
  PaddedDeviceHaloExchangerImpl &
  operator=(const PaddedDeviceHaloExchangerImpl &) = delete;

  ~PaddedDeviceHaloExchangerImpl() { cleanup(); }

  [[nodiscard]] bool uses_gpu_aware_mpi() const { return m_use_gpu_aware; }

  [[nodiscard]] const halo::HaloDirectionSet &direction_set() const noexcept {
    return m_dirs;
  }

  template <typename T>
  void exchange_halos_device(pfc::data::Field<T, typename Ops::space> &field,
                             stream_t stream = nullptr) {
    exchange_halos_device_impl(field.data(), field.size(), stream);
  }

  void exchange_halos_device(double *d_padded, std::size_t padded_size,
                             stream_t stream = nullptr) {
    exchange_halos_device_impl(d_padded, padded_size, stream);
  }

private:
  void exchange_halos_device_impl(double *d_padded, std::size_t padded_size,
                                  stream_t stream) {
    (void)padded_size;
    const bool perf = Ops::perf_enabled();
    auto &H = Ops::timers();

    double t_mark = MPI_Wtime();
    Ops::stream_sync(stream, Ops::sync_pre);
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
      Ops::device_sync(Ops::sync_post_aware);
      if (perf) {
        Ops::post_sync(H) += MPI_Wtime() - t_mark;
      }
    } else {
      exchange_packed_fallback_(d_padded, stream);
      t_mark = MPI_Wtime();
      Ops::stream_sync(stream, Ops::sync_post_packed);
      if (perf) {
        Ops::post_sync(H) += MPI_Wtime() - t_mark;
      }
    }
    profiling::record_time(profiling::kProfilingRegionCommunication,
                           MPI_Wtime() - t0);
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

  void exchange_gpu_aware_(double *d_padded, stream_t stream) {
    void *buf = static_cast<void *>(d_padded);
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
        const auto &recv = m_face_specs[static_cast<std::size_t>(
                                            opposite_slot(static_cast<int>(i)))]
                               .second;
        Ops::pack_face(m_d_scratch, d_padded, send.ox, send.oy, send.oz, send.sx,
                       send.sy, send.sz, m_nxp, m_nyp, m_nzp, stream);
        Ops::unpack_face(d_padded, m_d_scratch, recv.ox, recv.oy, recv.oz, recv.sx,
                         recv.sy, recv.sz, m_nxp, m_nyp, m_nzp, stream);
      }
      Ops::stream_sync(stream, Ops::sync_self);
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

  void exchange_packed_fallback_(double *d_padded, stream_t stream) {
    const bool perf = Ops::perf_enabled();
    auto &H = Ops::timers();

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
        const auto &recv = m_face_specs[static_cast<std::size_t>(
                                            opposite_slot(static_cast<int>(i)))]
                               .second;
        Ops::pack_face(m_d_scratch, d_padded, send.ox, send.oy, send.oz, send.sx,
                       send.sy, send.sz, m_nxp, m_nyp, m_nzp, stream);
        Ops::unpack_face(d_padded, m_d_scratch, recv.ox, recv.oy, recv.oz, recv.sx,
                         recv.sy, recv.sz, m_nxp, m_nyp, m_nzp, stream);
      }
      Ops::stream_sync(stream, Ops::sync_self_packed);
    }

    std::size_t req_count = 0;
    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const int tag = m_base_tag + opposite_slot(static_cast<int>(i));
      const int face_count = pfc::mpi::ensure_mpi_int_count(
          m_face_elems[i], "PaddedDeviceHaloExchanger packed face");
      pfc::mpi::throw_on_mpi_error(
          MPI_Irecv(m_h_recv[i], face_count, MPI_DOUBLE, m_neighbors[i], tag, m_comm,
                    &m_requests[req_count]),
          "PaddedDeviceHaloExchanger packed-fallback MPI_Irecv");
      ++req_count;
    }

    for (std::size_t i = 0; i < 6; ++i) {
      if (!m_active[i] || m_neighbors[i] == m_rank) {
        continue;
      }
      const double t_face = perf ? MPI_Wtime() : 0.0;
      const auto &send = m_face_specs[i].first;
      Ops::pack_face(m_d_scratch, d_padded, send.ox, send.oy, send.oz, send.sx,
                     send.sy, send.sz, m_nxp, m_nyp, m_nzp, stream);
      Ops::memcpy_async_d2h(m_h_send[i], m_d_scratch,
                            m_face_elems[i] * sizeof(double), stream,
                            Ops::memcpy_d2h);
      Ops::stream_sync(stream, Ops::sync_pack);
      if (perf) {
        H.packed_face_pack_d2h_sync += MPI_Wtime() - t_face;
      }

      const int tag = m_base_tag + static_cast<int>(i);
      const int face_count = pfc::mpi::ensure_mpi_int_count(
          m_face_elems[i], "PaddedDeviceHaloExchanger packed face");
      pfc::mpi::throw_on_mpi_error(
          MPI_Isend(m_h_send[i], face_count, MPI_DOUBLE, m_neighbors[i], tag, m_comm,
                    &m_requests[req_count]),
          "PaddedDeviceHaloExchanger packed-fallback MPI_Isend");
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
      Ops::memcpy_async_h2d(m_d_scratch, m_h_recv[i],
                            m_face_elems[i] * sizeof(double), stream,
                            Ops::memcpy_h2d);
      Ops::stream_sync(stream, Ops::sync_unpack);

      const auto &recv = m_face_specs[i].second;
      Ops::unpack_face(d_padded, m_d_scratch, recv.ox, recv.oy, recv.oz, recv.sx,
                       recv.sy, recv.sz, m_nxp, m_nyp, m_nzp, stream);
      if (perf) {
        H.packed_face_h2d_unpack_sync += MPI_Wtime() - t_face;
      }
    }
  }

  void cleanup() {
    if (!m_use_gpu_aware) {
      for (std::size_t i = 0; i < 6; ++i) {
        if (m_h_send[i] != nullptr) {
          Ops::free_host(m_h_send[i]);
          m_h_send[i] = nullptr;
        }
        if (m_h_recv[i] != nullptr) {
          Ops::free_host(m_h_recv[i]);
          m_h_recv[i] = nullptr;
        }
      }
    }
    if (m_d_scratch != nullptr) {
      Ops::free_dev(m_d_scratch);
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

  std::array<std::pair<pfc::gpu::detail::FaceSlabSpec, pfc::gpu::detail::FaceSlabSpec>,
             6>
      m_face_specs{};
  std::array<halo::FaceTypes, 6> m_face_types{};
  std::array<bool, 6> m_active{};
  std::vector<int> m_neighbors;
  std::vector<MPI_Request> m_requests;

  std::array<std::size_t, 6> m_face_elems{};
  std::size_t m_scratch_elems = 0;

  bool m_use_gpu_aware = false;
  bool m_any_self_neighbor = false;
  std::array<double *, 6> m_h_send{};
  std::array<double *, 6> m_h_recv{};
  double *m_d_scratch = nullptr;
};

} // namespace pfc::gpu

#if defined(OpenPFC_ENABLE_CUDA)
namespace pfc::cuda {

inline void print_cuda_halo_exchange_cpu_timers(MPI_Comm comm) {
  ::pfc::gpu::print_halo_exchange_cpu_timers<CudaHaloOps>(comm);
}

using PaddedDeviceHaloExchanger =
    ::pfc::gpu::PaddedDeviceHaloExchangerImpl<CudaHaloOps>;

} // namespace pfc::cuda
#endif

#if defined(OpenPFC_ENABLE_HIP)
namespace pfc::hip {

inline void print_hip_halo_exchange_cpu_timers(MPI_Comm comm) {
  ::pfc::gpu::print_halo_exchange_cpu_timers<HipHaloOps>(comm);
}

using PaddedDeviceHaloExchanger =
    ::pfc::gpu::PaddedDeviceHaloExchangerImpl<HipHaloOps>;

} // namespace pfc::hip
#endif

#endif // OpenPFC_ENABLE_CUDA || OpenPFC_ENABLE_HIP
