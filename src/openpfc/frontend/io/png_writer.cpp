// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <openpfc/frontend/io/png_writer.hpp>

#include <openpfc/kernel/data/world_queries.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include <stb_image_write.h>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace pfc::io {

void write_png_grayscale_8(const std::string &path, int width, int height,
                           const unsigned char *row_major) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(
        "write_png_grayscale_8: width and height must be positive");
  }
  if (stbi_write_png(path.c_str(), width, height, 1, row_major, width) == 0) {
    throw std::runtime_error("write_png_grayscale_8: failed to write " + path);
  }
}

void write_png_grayscale_from_doubles(const std::string &path, int width, int height,
                                      const double *row_major, double vmin,
                                      double vmax) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument(
        "write_png_grayscale_from_doubles: invalid dimensions");
  }
  const std::size_t n =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  std::vector<unsigned char> px(n);
  double lo = vmin;
  double hi = vmax;
  if (!(hi > lo)) {
    const double mid = 0.5 * (lo + hi);
    lo = mid - 1.0;
    hi = mid + 1.0;
  }
  const double inv = 1.0 / (hi - lo);
  for (std::size_t i = 0; i < n; ++i) {
    double t = (row_major[i] - lo) * inv;
    if (t < 0.0) {
      t = 0.0;
    }
    if (t > 1.0) {
      t = 1.0;
    }
    px[i] = static_cast<unsigned char>(std::lround(255.0 * t));
  }
  write_png_grayscale_8(path, width, height, px.data());
}

void write_mpi_scalar_field_png_xy(MPI_Comm comm,
                                   const pfc::decomposition::Decomposition &decomp,
                                   int rank, const std::vector<double> &local_field,
                                   const std::string &path) {
  int nproc = 1;
  MPI_Comm_size(comm, &nproc);

  const auto &gw = pfc::decomposition::get_world(decomp);
  auto gsz = pfc::world::get_size(gw);
  if (gsz[2] != 1) {
    throw std::invalid_argument(
        "write_mpi_scalar_field_png_xy: global nz must be 1");
  }
  const int nx_glob = gsz[0];
  const int ny_glob = gsz[1];

  for (int r = 0; r < nproc; ++r) {
    const auto &sw = pfc::decomposition::get_subworld(decomp, r);
    auto sz = pfc::world::get_size(sw);
    if (sz[2] != 1) {
      throw std::invalid_argument(
          "write_mpi_scalar_field_png_xy: each rank must have nz==1");
    }
  }

  const int my_count = static_cast<int>(local_field.size());
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

  MPI_Gatherv(const_cast<double *>(local_field.data()), my_count, MPI_DOUBLE,
              rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(),
              MPI_DOUBLE, 0, comm);

  if (rank != 0) {
    return;
  }

  std::vector<double> global(static_cast<std::size_t>(nx_glob) *
                             static_cast<std::size_t>(ny_glob));
  std::fill(global.begin(), global.end(), std::numeric_limits<double>::quiet_NaN());

  std::size_t offset = 0;
  for (int r = 0; r < nproc; ++r) {
    const auto &sw = pfc::decomposition::get_subworld(decomp, r);
    auto lo = pfc::world::get_lower(sw);
    auto sz = pfc::world::get_size(sw);
    const int nx = sz[0];
    const int ny = sz[1];
    const int nz = sz[2];
    (void)nz;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const std::size_t li =
            static_cast<std::size_t>(ix) +
            static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx);
        const int gx = lo[0] + ix;
        const int gy = lo[1] + iy;
        const std::size_t gi =
            static_cast<std::size_t>(gx) +
            static_cast<std::size_t>(gy) * static_cast<std::size_t>(nx_glob);
        global[gi] = gathered[offset + li];
      }
    }
    offset += static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  }

  double vmin = std::numeric_limits<double>::infinity();
  double vmax = -std::numeric_limits<double>::infinity();
  for (double v : global) {
    if (!std::isfinite(v)) {
      continue;
    }
    vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  if (!std::isfinite(vmin) || !std::isfinite(vmax)) {
    throw std::runtime_error(
        "write_mpi_scalar_field_png_xy: no valid samples in global field");
  }

  write_png_grayscale_from_doubles(path, nx_glob, ny_glob, global.data(), vmin,
                                   vmax);
}

} // namespace pfc::io
