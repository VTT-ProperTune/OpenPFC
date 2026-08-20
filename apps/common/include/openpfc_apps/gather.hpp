// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

/**
 * @file gather.hpp
 * @brief Rank-0 XY gather and ordered field stats for FD demo apps.
 */

#include <algorithm>
#include <limits>
#include <vector>

#include <mpi.h>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>

namespace pfc::apps {

inline void pack_owned_xy0(const pfc::data::Field<double, pfc::HostSpace> &b,
                           std::vector<double> &out) {
  const int nx = b.local_size()[0];
  const int ny = b.local_size()[1];
  out.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      out[static_cast<std::size_t>(i) +
          static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)] = b(i, j, 0);
    }
  }
}

inline void gather_global_xy_rank0(const pfc::decomposition::Decomposition &decomp,
                                   int rank, int nproc, MPI_Comm comm,
                                   const std::vector<double> &local_owned_xy,
                                   int nx_glob, int ny_glob,
                                   std::vector<double> &global_out) {
  const int my_count = static_cast<int>(local_owned_xy.size());
  std::vector<int> counts(static_cast<std::size_t>(nproc));
  MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

  std::vector<int> displs(static_cast<std::size_t>(nproc));
  int total = 0;
  for (int r = 0; r < nproc; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }

  std::vector<double> gathered;
  if (rank == 0) {
    gathered.resize(static_cast<std::size_t>(total));
  }

  MPI_Gatherv(const_cast<double *>(local_owned_xy.data()), my_count, MPI_DOUBLE,
              rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(),
              MPI_DOUBLE, 0, comm);

  if (rank != 0) {
    return;
  }

  global_out.assign(static_cast<std::size_t>(nx_glob) *
                        static_cast<std::size_t>(ny_glob),
                    std::numeric_limits<double>::quiet_NaN());

  std::size_t offset = 0;
  for (int r = 0; r < nproc; ++r) {
    const auto &box = pfc::decomposition::local_box(decomp, r);
    auto lo = box.low;
    auto sz = box.size;
    const int nx = sz[0];
    const int ny = sz[1];
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t li =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx);
        const int gx = lo[0] + ix;
        const int gy = lo[1] + iy;
        global_out[static_cast<std::size_t>(gx) +
                   static_cast<std::size_t>(gy) *
                       static_cast<std::size_t>(nx_glob)] = gathered[offset + li];
      }
    }
    offset += static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  }
}

struct FieldStats {
  double sum = 0.0;
  double sumsq = 0.0;
  double min_v = 0.0;
  double max_v = 0.0;
};

inline FieldStats stats_global_ordered(const std::vector<double> &global_xy,
                                       int nx_glob, int ny_glob) {
  FieldStats s{};
  s.min_v = std::numeric_limits<double>::infinity();
  s.max_v = -std::numeric_limits<double>::infinity();
  for (int gy = 0; gy < ny_glob; ++gy) {
    for (int gx = 0; gx < nx_glob; ++gx) {
      const double v = global_xy[static_cast<std::size_t>(gx) +
                                 static_cast<std::size_t>(gy) *
                                     static_cast<std::size_t>(nx_glob)];
      s.sum += v;
      s.sumsq += v * v;
      s.min_v = std::min(s.min_v, v);
      s.max_v = std::max(s.max_v, v);
    }
  }
  return s;
}

} // namespace pfc::apps
