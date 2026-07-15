// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file verification_utilities.hpp
 * @brief Shared verification helper functions for Kobayashi drivers.
 *
 * This header contains backend-agnostic post-processing code used by CPU, HIP,
 * and CUDA Kobayashi drivers to produce KOBAYASHI_VERIFY output statistics.
 * These functions ensure cross-backend numerical comparison for regression
 * checking.
 */

#ifndef KOBAYASHI_VERIFICATION_UTILITIES_HPP
#define KOBAYASHI_VERIFICATION_UTILITIES_HPP

#include <limits>
#include <mpi.h>
#include <string>
#include <vector>

#include <openpfc/kernel/field/padded_brick.hpp>
#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>

namespace {

/**
 * @brief Extract the z=0 plane from a padded brick field into a flat vector.
 *
 * Copies the owned cells at z=0 from a PaddedBrick into a vector ordered by
 * local (x,y) coordinates: `out[i + j*nx] = b(i, j, 0)`.
 *
 * @param b Input padded brick field.
 * @param out Output vector resized to `nx * ny` and filled with z=0 values.
 */
void pack_owned_xy0(const pfc::field::PaddedBrick<double> &b, std::vector<double> &out) {
  const int nx = b.nx();
  const int ny = b.ny();
  out.resize(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      out[static_cast<std::size_t>(i) +
          static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)] = b(i, j, 0);
    }
  }
}

/**
 * @brief Write a PNG visualization of the phi field.
 *
 * Extracts the z=0 plane and writes it as a PNG using MPI-aware I/O.
 *
 * @param rank MPI rank of the calling process.
 * @param decomp Domain decomposition describing process layout.
 * @param phi Phase field to visualize.
 * @param path Output file path for the PNG.
 */
void write_phi_png(int rank, const pfc::decomposition::Decomposition &decomp,
                   const pfc::field::PaddedBrick<double> &phi, const std::string &path) {
  std::vector<double> local;
  pack_owned_xy0(phi, local);
  pfc::io::write_mpi_scalar_field_png_xy(MPI_COMM_WORLD, decomp, rank, local, path,
                                         0.0, 1.0);
}

/**
 * @brief Gather one z-layer of owned cell values onto rank 0 in global row-major order.
 *
 * Collects all MPI ranks' owned z=0 field data onto rank 0, arranged in global
 * (x,y) coordinates: `global_out[gx + gy * nx_glob]`. This matches the layout
 * used by PNG gather and enables deterministic cross-backend comparison.
 *
 * @param decomp Domain decomposition describing process layout.
 * @param rank MPI rank of the calling process.
 * @param nproc Total number of MPI processes.
 * @param comm MPI communicator (typically MPI_COMM_WORLD).
 * @param local_owned_xy Local z=0 field values from pack_owned_xy0.
 * @param nx_glob Global domain size in x direction.
 * @param ny_glob Global domain size in y direction.
 * @param global_out Output vector (rank 0 only) filled with global field data.
 */
void gather_global_xy_rank0(const pfc::decomposition::Decomposition &decomp,
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

  global_out.assign(static_cast<std::size_t>(nx_glob) * static_cast<std::size_t>(ny_glob),
                    std::numeric_limits<double>::quiet_NaN());

  std::size_t offset = 0;
  for (int r = 0; r < nproc; ++r) {
    const auto &sw = pfc::decomposition::get_subworld(decomp, r);
    auto lo = pfc::world::get_lower(sw);
    auto sz = pfc::world::get_size(sw);
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
                     static_cast<std::size_t>(gy) * static_cast<std::size_t>(nx_glob)] =
            gathered[offset + li];
      }
    }
    offset += static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  }
}

/**
 * @brief Field statistics for verification output.
 */
struct FieldStats {
  double sum = 0.0;
  double sumsq = 0.0;
  double min_v = 0.0;
  double max_v = 0.0;
};

/**
 * @brief Compute field statistics in lexicographic (gx, gy) order.
 *
 * Accumulates sum, sum of squares, min, and max across the global field.
 * The lexicographic traversal order ensures deterministic results given
 * identical input data, enabling cross-backend regression checking.
 *
 * @param global_xy Global field data in row-major order.
 * @param nx_glob Global domain size in x direction.
 * @param ny_glob Global domain size in y direction.
 * @return FieldStats containing sum, sumsq, min_v, and max_v.
 */
FieldStats stats_global_ordered(const std::vector<double> &global_xy,
                                int nx_glob, int ny_glob) {
  FieldStats s{};
  s.min_v = std::numeric_limits<double>::infinity();
  s.max_v = -std::numeric_limits<double>::infinity();
  for (int gy = 0; gy < ny_glob; ++gy) {
    for (int gx = 0; gx < nx_glob; ++gx) {
      const double v =
          global_xy[static_cast<std::size_t>(gx) +
                    static_cast<std::size_t>(gy) * static_cast<std::size_t>(nx_glob)];
      s.sum += v;
      s.sumsq += v * v;
      s.min_v = std::min(s.min_v, v);
      s.max_v = std::max(s.max_v, v);
    }
  }
  return s;
}

} // unnamed namespace

#endif // KOBAYASHI_VERIFICATION_UTILITIES_HPP
