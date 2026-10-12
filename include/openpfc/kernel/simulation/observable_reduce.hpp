// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file observable_reduce.hpp
 * @brief Rank-local field sums and MPI allreduce for free-energy/observables.
 *
 * @details
 * Aluminum (M9) needs a global integral of a density. This header sums
 * owned cells (halo excluded), multiplies by the cell volume, and
 * `MPI_Allreduce`s with `MPI_SUM`.
 *
 * Device fields (`CUDASpace` / `HIPSpace`) pull a current host mirror via
 * `with_host_view` and sum the owned interior of that mirror. There is no
 * separate device reduction. Kernel-safe: no runtime GPU headers.
 *
 * `reduce_owned(field, comm, map)` maps one owned cell to one scalar
 * (`|u|`, `u*u`, a pointwise transform). It does not see neighbors or
 * other fields, so a gradient, a tensor invariant, or a multi-field
 * quantity stays in the caller. Materialize that scalar, then reduce it.
 *
 * Variance is the population second moment about the mean. Each rank
 * runs Welford's method on `x - x0`, where `x0` is that rank's first
 * owned sample, then adds `x0` back. A running mean of the raw values
 * cannot keep a small offset on top of a large baseline. Ranks combine
 * `(count, mean, M2)` with Chan's formula. A negative merged second
 * moment is rounding error and is replaced by zero, because that
 * moment is a sum of squares.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

namespace pfc::sim {

[[nodiscard]] inline double cell_volume(const Domain &domain) noexcept {
  const auto &s = domain::get_spacing(domain);
  return s[0] * s[1] * s[2];
}

/**
 * @brief Sum owned cells of @p field on this rank (halo excluded).
 *
 * Host fields iterate `for_each_owned`. Device fields refresh the host
 * mirror and sum the owned index box.
 */
template <class T, class MemorySpace = pfc::HostSpace>
[[nodiscard]] double sum_owned(pfc::data::Field<T, MemorySpace> &field) {
  double acc = 0.0;
  if constexpr (pfc::data::Field<T, MemorySpace>::is_host_space) {
    field.for_each_owned(
        [&](int i, int j, int k) { acc += static_cast<double>(field(i, j, k)); });
  } else {
    field.with_host_view([&](T *data, std::size_t) {
      const int nx = field.box().size[0];
      const int ny = field.box().size[1];
      const int nz = field.box().size[2];
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            acc += static_cast<double>(data[field.idx(i, j, k)]);
          }
        }
      }
    });
  }
  return acc;
}

[[nodiscard]] inline double allreduce_sum(double local, MPI_Comm comm) {
  double global = 0.0;
  const int err = MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce SUM in allreduce_sum");
  return global;
}

/**
 * @brief Element-wise `MPI_SUM` of `values` across `comm`, in place.
 *
 * Every rank must pass the same length. An empty span is a no-op
 * collective; a length above `INT_MAX` throws `std::length_error`.
 */
inline void allreduce_sum(std::span<double> values, MPI_Comm comm) {
  if (values.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error("allreduce_sum: span longer than INT_MAX");
  }
  const int err =
      MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                    MPI_DOUBLE, MPI_SUM, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce SUM in allreduce_sum(span)");
}

/**
 * @brief Global integral ∫ field dV ≈ (sum owned) × Δx Δy Δz, all ranks.
 */
template <class T, class MemorySpace = pfc::HostSpace>
[[nodiscard]] double integrate_owned(pfc::data::Field<T, MemorySpace> &field,
                                     MPI_Comm comm) {
  const double local = sum_owned(field) * cell_volume(field.domain());
  return allreduce_sum(local, comm);
}

/// Equal-weight moments of owned cells. Halo cells are not included.
struct FieldReduction {
  double sum{0.0};
  double min{0.0};
  double max{0.0};
  double mean{0.0};
  double l1{0.0};
  double l2{0.0};
  double rms{0.0};
  /// Population second moment about the mean, `M2 / count`.
  double variance{0.0};
  /// `sum * Δx Δy Δz`.
  double integral{0.0};
  std::uint64_t count{0};
};

namespace detail {

/// One rank's Welford state plus the plain sum and L1. `m2` is the sum of
/// squared deviations from `mean`.
struct PartialMoments {
  unsigned long long count{0};
  double mean{0.0};
  double m2{0.0};
  double sum{0.0};
  double l1{0.0};
  double minv{std::numeric_limits<double>::infinity()};
  double maxv{-std::numeric_limits<double>::infinity()};
};

static_assert(std::is_trivially_copyable_v<PartialMoments>);

/// `m.mean` is the mean of `x - origin` until the caller adds `origin`
/// back. `sum`, `l1`, and the bounds stay in the original units.
inline void consume(PartialMoments &m, double x, double origin) {
  const double y = x - origin;
  ++m.count;
  const double delta = y - m.mean;
  m.mean += delta / static_cast<double>(m.count);
  const double delta2 = y - m.mean;
  m.m2 += delta * delta2;
  m.sum += x;
  m.l1 += std::fabs(x);
  m.minv = std::min(m.minv, x);
  m.maxv = std::max(m.maxv, x);
}

/// Chan's pairwise merge. A negative `m2` is rounding error in a sum of
/// squares, so it is replaced by zero.
[[nodiscard]] inline PartialMoments merge_moments(PartialMoments a,
                                                  PartialMoments b) {
  if (a.count == 0) return b;
  if (b.count == 0) return a;
  PartialMoments c;
  c.count = a.count + b.count;
  const double nb = static_cast<double>(b.count);
  const double n = static_cast<double>(c.count);
  const double delta = b.mean - a.mean;
  c.mean = a.mean + delta * (nb / n);
  c.m2 = a.m2 + b.m2 + delta * delta * static_cast<double>(a.count) * nb / n;
  if (c.m2 < 0.0) c.m2 = 0.0;
  c.sum = a.sum + b.sum;
  c.l1 = a.l1 + b.l1;
  c.minv = std::min(a.minv, b.minv);
  c.maxv = std::max(a.maxv, b.maxv);
  return c;
}

template <class T, class MemorySpace, class Map>
[[nodiscard]] PartialMoments local_moments(pfc::data::Field<T, MemorySpace> &field,
                                           Map &&map) {
  PartialMoments m;
  double origin = 0.0;
  bool started = false;
  auto take = [&](double raw) {
    const double x = static_cast<double>(map(raw));
    if (!started) {
      origin = x;
      started = true;
    }
    consume(m, x, origin);
  };
  if constexpr (pfc::data::Field<T, MemorySpace>::is_host_space) {
    field.for_each_owned(
        [&](int i, int j, int k) { take(static_cast<double>(field(i, j, k))); });
  } else {
    field.with_host_view([&](T *data, std::size_t) {
      const int nx = field.box().size[0];
      const int ny = field.box().size[1];
      const int nz = field.box().size[2];
      for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
          for (int i = 0; i < nx; ++i) {
            take(static_cast<double>(data[field.idx(i, j, k)]));
          }
        }
      }
    });
  }
  if (m.count > 0) m.mean += origin;
  return m;
}

} // namespace detail

/**
 * @brief Global owned-cell reductions of `map(value)`.
 *
 * `map` receives one cell and returns one scalar. It does not receive
 * neighbors or another field. Empty fields reduce to a zero count and a
 * NaN min/max. Variance is `M2 / count` after a rank-order Chan merge, so
 * a large baseline does not cancel the fluctuation.
 */
template <class T, class MemorySpace, class Map>
[[nodiscard]] FieldReduction reduce_owned(pfc::data::Field<T, MemorySpace> &field,
                                          MPI_Comm comm, Map &&map) {
  if (comm == MPI_COMM_NULL) {
    throw std::invalid_argument("reduce_owned: pass an explicit communicator");
  }
  const detail::PartialMoments local =
      detail::local_moments(field, std::forward<Map>(map));
  int nproc = 1;
  pfc::mpi::throw_on_mpi_error(MPI_Comm_size(comm, &nproc),
                               "reduce_owned: MPI_Comm_size");
  std::vector<detail::PartialMoments> parts(static_cast<std::size_t>(nproc));
  pfc::mpi::throw_on_mpi_error(
      MPI_Allgather(&local, static_cast<int>(sizeof(local)), MPI_BYTE, parts.data(),
                    static_cast<int>(sizeof(local)), MPI_BYTE, comm),
      "reduce_owned: MPI_Allgather");
  detail::PartialMoments merged;
  for (const detail::PartialMoments &part : parts) {
    merged = detail::merge_moments(merged, part);
  }

  FieldReduction out;
  out.sum = merged.sum;
  out.l1 = merged.l1;
  out.count = static_cast<std::uint64_t>(merged.count);
  if (out.count == 0) {
    out.min = std::numeric_limits<double>::quiet_NaN();
    out.max = out.min;
    return out;
  }
  const double n = static_cast<double>(out.count);
  out.min = merged.minv;
  out.max = merged.maxv;
  out.mean = merged.mean;
  out.variance = merged.m2 / n;
  const double sum_sq = merged.m2 + n * merged.mean * merged.mean;
  out.l2 = std::sqrt(sum_sq);
  out.rms = std::sqrt(sum_sq / n);
  out.integral = out.sum * cell_volume(field.domain());
  return out;
}

template <class T, class MemorySpace = pfc::HostSpace>
[[nodiscard]] FieldReduction reduce_owned(pfc::data::Field<T, MemorySpace> &field,
                                          MPI_Comm comm) {
  return reduce_owned(field, comm, [](double v) { return v; });
}

} // namespace pfc::sim
