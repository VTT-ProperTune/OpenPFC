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
 * Device fields (`CudaSpace` / `HipSpace`) pull a current host mirror via
 * `with_host_view` and sum the owned interior of that mirror. Kernel-safe:
 * no runtime GPU headers.
 */

#include <cstddef>
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
    field.for_each_owned([&](int i, int j, int k) {
      acc += static_cast<double>(field(i, j, k));
    });
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
  const int err =
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  pfc::mpi::throw_on_mpi_error(err, "MPI_Allreduce SUM in allreduce_sum");
  return global;
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

} // namespace pfc::sim
