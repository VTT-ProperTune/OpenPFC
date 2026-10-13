// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/**
 * @file distance.hpp
 * @brief Periodic Euclidean distance of a binary Cartesian mask.
 *
 * Voxel `(i, j, k)` is the sample at its centre,
 * `((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h)`, on one isotropic
 * spacing `h`. The distance is the Euclidean distance between those
 * centres, taking the nearest periodic image on every axis. It is not a
 * reconstructed surface distance, and it is not a Hamilton–Jacobi
 * redistancing of an existing level set.
 *
 * `periodic_distance` is zero on an occupied voxel and positive elsewhere.
 * `periodic_signed_distance` follows the implicit-geometry convention:
 * negative on occupied voxels, positive on empty voxels. The magnitude on
 * an occupied voxel is the distance to the nearest empty voxel centre, so
 * a voxel next to an empty face neighbour has inside value `-h`.
 *
 * The mask is stored x-fastest. A nonzero byte is occupied. The grid is
 * periodic and isotropic. An empty mask has no distance. A mask that fills
 * the grid has no outside, so the signed distance is rejected.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace pfc::geometry {

namespace distance_detail {

inline constexpr double kNoFeature = 1.0e20;

inline void squared_distance_1d(const double *f, double *d, int n) {
  std::vector<int> v(static_cast<std::size_t>(n));
  std::vector<double> z(static_cast<std::size_t>(n) + 1);
  int k = 0;
  v[0] = 0;
  z[0] = -kNoFeature;
  z[1] = kNoFeature;
  for (int q = 1; q < n; ++q) {
    double s = 0.0;
    while (true) {
      const int vk = v[static_cast<std::size_t>(k)];
      s = ((f[q] + static_cast<double>(q) * q) -
           (f[vk] + static_cast<double>(vk) * vk)) /
          (2.0 * q - 2.0 * vk);
      if (k > 0 && s <= z[static_cast<std::size_t>(k)]) {
        --k;
      } else {
        break;
      }
    }
    ++k;
    v[static_cast<std::size_t>(k)] = q;
    z[static_cast<std::size_t>(k)] = s;
    z[static_cast<std::size_t>(k) + 1] = kNoFeature;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[static_cast<std::size_t>(k) + 1] < q) ++k;
    const int vk = v[static_cast<std::size_t>(k)];
    d[q] = static_cast<double>(q - vk) * (q - vk) + f[vk];
  }
}

inline void squared_distance_periodic(std::vector<double> &f) {
  const int n = static_cast<int>(f.size());
  bool any = false;
  for (double value : f) any = any || value < kNoFeature * 0.5;
  if (!any || n == 0) return;
  std::vector<double> pad(static_cast<std::size_t>(3 * n));
  std::vector<double> out(pad.size());
  for (int i = 0; i < 3 * n; ++i)
    pad[static_cast<std::size_t>(i)] = f[static_cast<std::size_t>(i % n)];
  squared_distance_1d(pad.data(), out.data(), 3 * n);
  for (int i = 0; i < n; ++i)
    f[static_cast<std::size_t>(i)] = out[static_cast<std::size_t>(n + i)];
}

inline std::size_t index_of(int x, int y, int z, int nx, int ny) {
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(nx) *
             (static_cast<std::size_t>(y) +
              static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
}

inline void require_grid(std::span<const std::uint8_t> mask, int nx, int ny, int nz,
                         double spacing) {
  // The periodic sweep stores a triple copy of one axis in an int index.
  constexpr int kMaxAxis = std::numeric_limits<int>::max() / 3;
  if (nx < 1 || ny < 1 || nz < 1 || nx > kMaxAxis || ny > kMaxAxis ||
      nz > kMaxAxis || !(spacing > 0.0) || !std::isfinite(spacing)) {
    throw std::invalid_argument("distance: grid and spacing must be positive");
  }
  const auto ux = static_cast<std::size_t>(nx);
  const auto uy = static_cast<std::size_t>(ny);
  const auto uz = static_cast<std::size_t>(nz);
  if (ux > std::numeric_limits<std::size_t>::max() / uy ||
      ux * uy > std::numeric_limits<std::size_t>::max() / uz) {
    throw std::invalid_argument("distance: grid is too large");
  }
  const auto n = ux * uy * uz;
  if (mask.size() != n)
    throw std::invalid_argument("distance: mask size does not match the grid");
}

inline std::vector<double> squared_voxels(std::span<const std::uint8_t> feature,
                                          int nx, int ny, int nz) {
  const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                        static_cast<std::size_t>(nz);
  std::vector<double> dist(n, kNoFeature);
  std::vector<double> line;
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      line.resize(static_cast<std::size_t>(nx));
      for (int x = 0; x < nx; ++x) {
        line[static_cast<std::size_t>(x)] =
            feature[index_of(x, y, z, nx, ny)] ? 0.0 : kNoFeature;
      }
      squared_distance_periodic(line);
      for (int x = 0; x < nx; ++x)
        dist[index_of(x, y, z, nx, ny)] = line[static_cast<std::size_t>(x)];
    }
  }
  for (int z = 0; z < nz; ++z) {
    for (int x = 0; x < nx; ++x) {
      line.resize(static_cast<std::size_t>(ny));
      for (int y = 0; y < ny; ++y)
        line[static_cast<std::size_t>(y)] = dist[index_of(x, y, z, nx, ny)];
      squared_distance_periodic(line);
      for (int y = 0; y < ny; ++y)
        dist[index_of(x, y, z, nx, ny)] = line[static_cast<std::size_t>(y)];
    }
  }
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      line.resize(static_cast<std::size_t>(nz));
      for (int z = 0; z < nz; ++z)
        line[static_cast<std::size_t>(z)] = dist[index_of(x, y, z, nx, ny)];
      squared_distance_periodic(line);
      for (int z = 0; z < nz; ++z)
        dist[index_of(x, y, z, nx, ny)] = line[static_cast<std::size_t>(z)];
    }
  }
  return dist;
}

} // namespace distance_detail

/// Euclidean distance, in length units, to the nearest occupied voxel centre.
[[nodiscard]] inline std::vector<double>
periodic_distance(std::span<const std::uint8_t> occupied, int nx, int ny, int nz,
                  double spacing) {
  distance_detail::require_grid(occupied, nx, ny, nz, spacing);
  bool any = false;
  for (std::uint8_t value : occupied) any = any || value != 0;
  if (!any) throw std::invalid_argument("distance: mask is empty");
  const auto squared = distance_detail::squared_voxels(occupied, nx, ny, nz);
  std::vector<double> out(squared.size());
  for (std::size_t i = 0; i < squared.size(); ++i) {
    out[i] = std::sqrt(std::max(0.0, squared[i])) * spacing;
  }
  return out;
}

/// Negative inside the occupied set, positive outside. See the file comment.
[[nodiscard]] inline std::vector<double>
periodic_signed_distance(std::span<const std::uint8_t> occupied, int nx, int ny,
                         int nz, double spacing) {
  distance_detail::require_grid(occupied, nx, ny, nz, spacing);
  const std::size_t n = occupied.size();
  std::vector<std::uint8_t> empty(n);
  bool any_occupied = false;
  bool any_empty = false;
  for (std::size_t i = 0; i < n; ++i) {
    empty[i] = occupied[i] ? 0 : 1;
    any_occupied = any_occupied || occupied[i] != 0;
    any_empty = any_empty || occupied[i] == 0;
  }
  if (!any_occupied) throw std::invalid_argument("distance: mask is empty");
  if (!any_empty) throw std::invalid_argument("distance: mask fills the grid");
  const auto to_occupied = distance_detail::squared_voxels(occupied, nx, ny, nz);
  const auto to_empty = distance_detail::squared_voxels(empty, nx, ny, nz);
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double magnitude =
        std::sqrt(std::max(0.0, occupied[i] ? to_empty[i] : to_occupied[i])) *
        spacing;
    out[i] = occupied[i] ? -magnitude : magnitude;
  }
  return out;
}

} // namespace pfc::geometry
