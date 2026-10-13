// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/geometry/distance.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using pfc::geometry::periodic_distance;
using pfc::geometry::periodic_signed_distance;

namespace {

std::size_t at(int x, int y, int z, int nx, int ny) {
  return static_cast<std::size_t>(x) +
         static_cast<std::size_t>(nx) *
             (static_cast<std::size_t>(y) +
              static_cast<std::size_t>(ny) * static_cast<std::size_t>(z));
}

int wrap_delta(int a, int b, int n) {
  int d = std::abs(a - b);
  if (d > n - d) d = n - d;
  return d;
}

double brute_distance(const std::vector<std::uint8_t> &mask, int nx, int ny, int nz,
                      double spacing, int x, int y, int z) {
  double best = 1.0e300;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        if (!mask[at(i, j, k, nx, ny)]) continue;
        const double dx = wrap_delta(x, i, nx) * spacing;
        const double dy = wrap_delta(y, j, ny) * spacing;
        const double dz = wrap_delta(z, k, nz) * spacing;
        best = std::min(best, dx * dx + dy * dy + dz * dz);
      }
    }
  }
  return std::sqrt(best);
}

} // namespace

TEST_CASE("one occupied voxel has a periodic distance", "[geometry][distance]") {
  constexpr int nx = 8;
  constexpr int ny = 6;
  constexpr int nz = 5;
  constexpr double h = 1.5;
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(nx) * ny * nz, 0);
  mask[at(0, 2, 3, nx, ny)] = 1;
  const auto distance = periodic_distance(mask, nx, ny, nz, h);
  const auto signed_distance = periodic_signed_distance(mask, nx, ny, nz, h);

  REQUIRE(distance[at(0, 2, 3, nx, ny)] == 0.0);
  REQUIRE(distance[at(1, 2, 3, nx, ny)] == h);
  REQUIRE(distance[at(nx - 1, 2, 3, nx, ny)] == h);
  REQUIRE(distance[at(4, 2, 3, nx, ny)] == 4.0 * h);
  REQUIRE(std::abs(distance[at(1, 3, 3, nx, ny)] - h * std::sqrt(2.0)) < 1.0e-12);

  REQUIRE(signed_distance[at(0, 2, 3, nx, ny)] == -h);
  REQUIRE(signed_distance[at(1, 2, 3, nx, ny)] == h);
  REQUIRE(signed_distance[at(nx - 1, 2, 3, nx, ny)] == h);

  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
        const auto i = at(x, y, z, nx, ny);
        REQUIRE(std::abs(distance[i] -
                         brute_distance(mask, nx, ny, nz, h, x, y, z)) < 1.0e-9);
      }
    }
  }
}

TEST_CASE("a flat interface away from the seam is a plane distance",
          "[geometry][distance]") {
  constexpr int nx = 16;
  constexpr int ny = 4;
  constexpr int nz = 3;
  constexpr double h = 0.5;
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(nx) * ny * nz, 0);
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < 8; ++x) mask[at(x, y, z, nx, ny)] = 1;
    }
  }
  const auto distance = periodic_distance(mask, nx, ny, nz, h);
  const auto signed_distance = periodic_signed_distance(mask, nx, ny, nz, h);
  REQUIRE(distance[at(7, 1, 1, nx, ny)] == 0.0);
  REQUIRE(distance[at(8, 1, 1, nx, ny)] == h);
  REQUIRE(distance[at(10, 1, 1, nx, ny)] == 3.0 * h);
  REQUIRE(signed_distance[at(6, 1, 1, nx, ny)] == -2.0 * h);
  REQUIRE(signed_distance[at(8, 1, 1, nx, ny)] == h);
  REQUIRE(signed_distance[at(9, 2, 0, nx, ny)] == 2.0 * h);
}

TEST_CASE("a feature on a periodic face is one spacing from the opposite face",
          "[geometry][distance]") {
  constexpr int n = 9;
  constexpr double h = 2.0;
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(n) * n * n, 0);
  mask[at(0, 0, n - 1, n, n)] = 4;
  const auto distance = periodic_distance(mask, n, n, n, h);
  REQUIRE(distance[at(n - 1, 0, n - 1, n, n)] == h);
  REQUIRE(distance[at(0, n - 1, n - 1, n, n)] == h);
  REQUIRE(distance[at(0, 0, 0, n, n)] == h);
  REQUIRE(std::abs(distance[at(n - 1, n - 1, 0, n, n)] - h * std::sqrt(3.0)) <
          1.0e-12);

  for (int z = 0; z < n; ++z) {
    for (int y = 0; y < n; ++y) {
      for (int x = 0; x < n; ++x) {
        REQUIRE(std::abs(distance[at(x, y, z, n, n)] -
                         brute_distance(mask, n, n, n, h, x, y, z)) < 1.0e-9);
      }
    }
  }
}

TEST_CASE("an empty or solid mask is rejected", "[geometry][distance]") {
  std::vector<std::uint8_t> mask(8, 0);
  REQUIRE_THROWS_AS(periodic_distance(mask, 2, 2, 2, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_signed_distance(mask, 2, 2, 2, 1.0),
                    std::invalid_argument);
  for (auto &value : mask) value = 1;
  REQUIRE(periodic_distance(mask, 2, 2, 2, 1.0)[0] == 0.0);
  REQUIRE_THROWS_AS(periodic_signed_distance(mask, 2, 2, 2, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_distance(mask, 2, 2, 1, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_distance(mask, 2, 2, 2, 0.0), std::invalid_argument);
}

TEST_CASE("one-dimensional and flat grids match a periodic brute force",
          "[geometry][distance]") {
  std::vector<std::uint8_t> line(13, 0);
  line[0] = 1;
  line[5] = 1;
  line[12] = 1;
  const auto line_distance = periodic_distance(line, 13, 1, 1, 0.25);
  const auto line_signed = periodic_signed_distance(line, 13, 1, 1, 0.25);
  for (int x = 0; x < 13; ++x) {
    REQUIRE(std::abs(line_distance[static_cast<std::size_t>(x)] -
                     brute_distance(line, 13, 1, 1, 0.25, x, 0, 0)) < 1.0e-9);
    const double outside = brute_distance(line, 13, 1, 1, 0.25, x, 0, 0);
    const double inside_or_out = line[static_cast<std::size_t>(x)] ? -0.25 : outside;
    // A 1D occupied site's nearest empty centre is one spacing away.
    if (line[static_cast<std::size_t>(x)]) {
      REQUIRE(line_signed[static_cast<std::size_t>(x)] == -0.25);
    } else {
      REQUIRE(std::abs(line_signed[static_cast<std::size_t>(x)] - inside_or_out) <
              1.0e-9);
    }
  }

  constexpr int nx = 7;
  constexpr int ny = 9;
  std::vector<std::uint8_t> plane(static_cast<std::size_t>(nx * ny), 0);
  plane[at(0, 0, 0, nx, ny)] = 1;
  plane[at(nx - 1, ny - 1, 0, nx, ny)] = 1;
  plane[at(3, 1, 0, nx, ny)] = 3;
  const auto plane_distance = periodic_distance(plane, nx, ny, 1, 2.0);
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      REQUIRE(std::abs(plane_distance[at(x, y, 0, nx, ny)] -
                       brute_distance(plane, nx, ny, 1, 2.0, x, y, 0)) < 1.0e-9);
    }
  }
}

TEST_CASE("small randomized masks match an independent brute force",
          "[geometry][distance]") {
  std::mt19937 rng(520);
  std::uniform_int_distribution<int> dim(1, 6);
  std::uniform_int_distribution<int> bit(0, 2);
  for (int trial = 0; trial < 24; ++trial) {
    const int nx = dim(rng);
    const int ny = dim(rng);
    const int nz = dim(rng);
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(nx) * ny * nz);
    for (auto &value : mask) value = bit(rng) == 0 ? 1 : 0;
    const int occupied = std::count(mask.begin(), mask.end(), 1);
    if (occupied == 0) {
      REQUIRE_THROWS_AS(periodic_distance(mask, nx, ny, nz, 1.0),
                        std::invalid_argument);
      continue;
    }
    const auto first = periodic_distance(mask, nx, ny, nz, 1.0);
    const auto second = periodic_distance(mask, nx, ny, nz, 1.0);
    REQUIRE(first == second);
    for (int z = 0; z < nz; ++z) {
      for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
          REQUIRE(std::abs(first[at(x, y, z, nx, ny)] -
                           brute_distance(mask, nx, ny, nz, 1.0, x, y, z)) < 1.0e-8);
        }
      }
    }
    if (occupied != nx * ny * nz) {
      const auto signed_distance = periodic_signed_distance(mask, nx, ny, nz, 1.0);
      for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
          for (int x = 0; x < nx; ++x) {
            const auto i = at(x, y, z, nx, ny);
            if (mask[i]) {
              REQUIRE(signed_distance[i] < 0.0);
            } else {
              REQUIRE(signed_distance[i] == first[i]);
            }
          }
        }
      }
    } else {
      REQUIRE_THROWS_AS(periodic_signed_distance(mask, nx, ny, nz, 1.0),
                        std::invalid_argument);
    }
  }
}

TEST_CASE("distance rejects degenerate extents and a size overflow",
          "[geometry][distance]") {
  const std::vector<std::uint8_t> one{1};
  REQUIRE_THROWS_AS(periodic_distance(one, 0, 1, 1, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_distance(one, 1, -2, 1, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_distance(one, 1, 1, 1, -1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(
      periodic_distance(one, 1, 1, 1, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      periodic_distance(one, std::numeric_limits<int>::max(), 1, 1, 1.0),
      std::invalid_argument);
  // 3e6^3 does not fit in size_t. The call must fail before any allocation
  // of that product.
  constexpr int big = 3000000;
  REQUIRE_THROWS_AS(periodic_distance(one, big, big, big, 1.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_signed_distance(one, big, big, big, 1.0),
                    std::invalid_argument);
}
