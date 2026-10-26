// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file observable_reduce_gpu.hpp
 * @brief Fused owned-cell device reductions, returning compact global scalars.
 *
 * Instantiate in a CUDA/HIP translation unit. A device-callable map can read
 * other synchronized fields through captured device pointers. It receives the
 * primary field value and local owned coordinates; padding is never visited.
 */

#include <array>
#include <cmath>
#include <exception>
#include <type_traits>
#include <vector>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/mpi/collective_check.hpp>
#include <openpfc/runtime/gpu/databuffer_gpu.hpp>
#include <openpfc/runtime/gpu/gpu_api.hpp>
#include <openpfc/runtime/gpu/memory_space_gpu.hpp>

namespace pfc::sim::gpu {

enum class ReductionOp { Sum, Min, Max };

/// Device-safe fixed-width result of a per-cell map; no owning storage.
template <std::size_t N> struct ReductionValues {
  double values[N];
  OPENPFC_HD double &operator[](std::size_t i) noexcept { return values[i]; }
  OPENPFC_HD double operator[](std::size_t i) const noexcept { return values[i]; }
};

template <std::size_t N> struct ReductionOps {
  ReductionOp values[N];
};

namespace detail {
OPENPFC_HD inline double identity(ReductionOp op) {
  return op == ReductionOp::Sum ? 0.0
                                : (op == ReductionOp::Min ? INFINITY : -INFINITY);
}
OPENPFC_HD inline double combine(double a, double b, ReductionOp op) {
  if (op == ReductionOp::Sum) return a + b;
  return op == ReductionOp::Min ? (a < b ? a : b) : (a > b ? a : b);
}

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
template <std::size_t N, class T, class Map>
__global__ void reduce_owned_kernel(const T *data, int nx, int ny, std::size_t count,
                                    int halo, std::size_t sy, std::size_t sz,
                                    Map map, ReductionOps<N> ops, double *partials) {
  __shared__ double scratch[256][N];
  const unsigned lane = threadIdx.x;
  double local[N];
  for (std::size_t c = 0; c < N; ++c) local[c] = identity(ops.values[c]);
  for (std::size_t q = std::size_t(blockIdx.x) * blockDim.x + lane; q < count;
       q += std::size_t(blockDim.x) * gridDim.x) {
    const int i = q % nx, j = (q / nx) % ny, k = q / (std::size_t(nx) * ny);
    const auto offset = std::size_t(i + halo) + std::size_t(j + halo) * sy +
                        std::size_t(k + halo) * sz;
    const auto values = map(data[offset], i, j, k);
    for (std::size_t c = 0; c < N; ++c)
      local[c] = combine(local[c], values[c], ops.values[c]);
  }
  for (std::size_t c = 0; c < N; ++c) scratch[lane][c] = local[c];
  __syncthreads();
  for (unsigned stride = 128; stride > 0; stride /= 2) {
    if (lane < stride)
      for (std::size_t c = 0; c < N; ++c)
        scratch[lane][c] =
            combine(scratch[lane][c], scratch[lane + stride][c], ops.values[c]);
    __syncthreads();
  }
  if (lane == 0)
    for (std::size_t c = 0; c < N; ++c)
      partials[std::size_t(blockIdx.x) * N + c] = scratch[0][c];
}
#endif
} // namespace detail

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
/**
 * @brief Reduce several mapped expressions in one device pass, then across MPI.
 *
 * @tparam N Number of expressions, 1..16 (bounded block shared memory).
 * @param field Primary padded or unpadded device field; only owned cells count.
 * @param map Trivially copyable device callable `(T value,int i,int j,int k)`
 * returning ReductionValues<N>. Captured device buffers must remain valid and
 * synchronized through this blocking call. Every expression must avoid NaNs:
 * count invalid samples explicitly and supply the operation's identity instead.
 * @param ops Per-expression Sum, Min or Max; identical on all ranks.
 * @param comm All members call with the same N and operations, in the same order.
 * @return Global scalars. Empty input contributes 0, +infinity or -infinity.
 * @note Synchronizes stale field storage to the device. At most 256*N partial
 * scalars are read back; no field-sized host mirror is requested. MPI combines
 * scalars in at most three collectives, grouping expressions by operation.
 * Floating-point sums depend on device/MPI reduction order, not bitwise parity.
 * A collective success check precedes the value collectives: a local device
 * failure causes every rank to throw before combining values.
 * @throws std::runtime_error On device or MPI errors.
 */
template <std::size_t N, class T, class Space, class Map>
[[nodiscard]] ReductionValues<N> reduce_owned_multi(data::Field<T, Space> &field,
                                                    Map map, ReductionOps<N> ops,
                                                    MPI_Comm comm) {
  static_assert(N >= 1 && N <= 16);
  static_assert(!data::Field<T, Space>::is_host_space,
                "reduce_owned_multi requires device storage");
  static_assert(std::is_trivially_copyable_v<Map>);
  for (const auto op : ops.values)
    if (op != ReductionOp::Sum && op != ReductionOp::Min && op != ReductionOp::Max)
      throw std::invalid_argument("reduce_owned_multi: invalid operation");
  ReductionValues<N> result{};
  std::exception_ptr local_error;
  try {
    field.sync_to_device();
    const auto n = field.local_size();
    const std::size_t count = std::size_t(n[0]) * n[1] * n[2];
    const auto needed = (count + 255) / 256;
    const int blocks =
        static_cast<int>(needed < 1 ? 1 : (needed > 256 ? 256 : needed));
    using Tag = typename data::Field<T, Space>::backend_tag;
    core::DataBuffer<Tag, double> partials(std::size_t(blocks) * N);
    detail::reduce_owned_kernel<N><<<blocks, 256>>>(
        field.data(), n[0], n[1], count, field.storage_halo(),
        field.padded_extent(0),
        std::size_t(field.padded_extent(0)) * field.padded_extent(1), map, ops,
        partials.data());
    GPU_CHECK(pfc::gpuGetLastError());
    std::vector<double> host(std::size_t(blocks) * N);
    partials.copy_to_host(host.data(), host.size());
    for (std::size_t c = 0; c < N; ++c) {
      result[c] = detail::identity(ops.values[c]);
      for (int block = 0; block < blocks; ++block)
        result[c] = detail::combine(result[c], host[std::size_t(block) * N + c],
                                    ops.values[c]);
    }
  } catch (...) {
    local_error = std::current_exception();
  }
  if (!mpi::all_ranks_succeeded(!local_error, comm)) {
    if (local_error) std::rethrow_exception(local_error);
    throw std::runtime_error(
        "reduce_owned_multi: another rank's device reduction failed");
  }
  for (auto op : {ReductionOp::Sum, ReductionOp::Min, ReductionOp::Max}) {
    std::array<double, N> values{};
    std::array<std::size_t, N> columns{};
    int width = 0;
    for (std::size_t c = 0; c < N; ++c)
      if (ops.values[c] == op) {
        columns[width] = c;
        values[width++] = result[c];
      }
    if (width == 0) continue;
    const auto mpi_op = op == ReductionOp::Sum
                            ? MPI_SUM
                            : (op == ReductionOp::Min ? MPI_MIN : MPI_MAX);
    mpi::throw_on_mpi_error(
        MPI_Allreduce(MPI_IN_PLACE, values.data(), width, MPI_DOUBLE, mpi_op, comm),
        "MPI_Allreduce in reduce_owned_multi");
    for (int c = 0; c < width; ++c) result[columns[c]] = values[c];
  }
  return result;
}
#endif
} // namespace pfc::sim::gpu
