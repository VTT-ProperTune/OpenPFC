// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cmath>
#include <set>
#include <tuple>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/brick_iteration.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

namespace {

pfc::data::Field<double, pfc::HostSpace> make_brick(int n, int hw) {
  auto domain = pfc::domain::create(pfc::GridSize({n, n, n}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(domain, 1);
  return pfc::data::field_from_subdomain<double>(decomp, /*rank=*/0, hw);
}

} // namespace

TEST_CASE("for_each_owned visits every owned cell exactly once",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/1);
  std::set<std::tuple<int, int, int>> seen;
  field::for_each_owned(u, [&](int i, int j, int k) { seen.insert({i, j, k}); });
  const auto sz = u.local_size();
  REQUIRE(seen.size() == static_cast<std::size_t>(sz[0] * sz[1] * sz[2]));
  REQUIRE(*seen.begin() == std::tuple{0, 0, 0});
  REQUIRE(*seen.rbegin() == std::tuple{sz[0] - 1, sz[1] - 1, sz[2] - 1});
}

TEST_CASE("for_each yields Int3 in row-major order over every owned cell",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/1);
  std::vector<pfc::Int3> seen;
  field::for_each(u, [&](const auto &idx) { seen.push_back(idx); });

  const auto sz = u.local_size();
  REQUIRE(seen.size() == static_cast<std::size_t>(sz[0] * sz[1] * sz[2]));
  REQUIRE(seen.front() == pfc::Int3{0, 0, 0});
  REQUIRE(seen.back() == pfc::Int3{sz[0] - 1, sz[1] - 1, sz[2] - 1});

  // Check the k-outer / j-middle / i-inner ordering: the i index must
  // monotonically advance until it wraps, then j, then k.
  bool ordering_matches = true;
  for (std::size_t s = 1; s < seen.size(); ++s) {
    const auto &p = seen[s - 1];
    const auto &c = seen[s];
    const bool i_advance = (c[0] == p[0] + 1) && c[1] == p[1] && c[2] == p[2];
    const bool j_advance = (c[0] == 0) && c[1] == p[1] + 1 && c[2] == p[2];
    const bool k_advance = (c[0] == 0) && c[1] == 0 && c[2] == p[2] + 1;
    ordering_matches &= i_advance || j_advance || k_advance;
  }
  REQUIRE(ordering_matches);

  // Confirm the body can write through brick(idx) without any (i, j, k)
  // unpacking — this is the workflow the heat3d_fd driver will use.
  field::for_each(u, [&](const auto &idx) { u(idx) = idx[0] + 10 * idx[1]; });
  bool values_match = true;
  for (int k = 0; k < sz[2]; ++k)
    for (int j = 0; j < sz[1]; ++j)
      for (int i = 0; i < sz[0]; ++i) values_match &= u(i, j, k) == i + 10 * j;
  REQUIRE(values_match);
}

TEST_CASE("for_each_inner stays in [r, n-r) and obeys r=0 -> entire owned",
          "[field][brick_iteration]") {
  auto u = make_brick(5, /*hw=*/2);
  const auto sz = u.local_size();
  int count_r0 = 0, count_r1 = 0, count_r2 = 0;
  field::for_each_inner(u, 0, [&](int, int, int) { ++count_r0; });
  field::for_each_inner(u, 1, [&](int, int, int) { ++count_r1; });
  field::for_each_inner(u, 2, [&](int, int, int) { ++count_r2; });
  REQUIRE(count_r0 == sz[0] * sz[1] * sz[2]);
  REQUIRE(count_r1 == 3 * 3 * 3);
  REQUIRE(count_r2 == 1);
}

TEST_CASE("for_each_inner is a no-op when n <= 2*r", "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/2);
  int count = 0;
  field::for_each_inner(u, 2, [&](int, int, int) { ++count; });
  REQUIRE(count == 0);
}

TEST_CASE("for_each_border covers owned-minus-inner exactly once",
          "[field][brick_iteration]") {
  bool all_regions_are_valid = true;
  for (int n : {4, 5, 6, 8, 24, 32}) {
    for (int r : {1, 2, 4, 6, 10}) {
      if (n <= 2 * r) continue;
      auto u = make_brick(n, /*hw=*/r);
      std::set<std::tuple<int, int, int>> border;
      bool border_is_unique = true;
      field::for_each_border(u, r, [&](int i, int j, int k) {
        const auto added = border.emplace(i, j, k).second;
        border_is_unique &= added;
      });

      std::set<std::tuple<int, int, int>> inner;
      field::for_each_inner(u, r,
                            [&](int i, int j, int k) { inner.emplace(i, j, k); });

      all_regions_are_valid &=
          border.size() + inner.size() == static_cast<std::size_t>(n * n * n);

      bool regions_are_disjoint = true;
      for (const auto &c : border) regions_are_disjoint &= !inner.contains(c);

      bool border_is_on_faces = true;
      for (const auto &b : border) {
        const int i = std::get<0>(b);
        const int j = std::get<1>(b);
        const int k = std::get<2>(b);
        const bool on_face = (i < r) || (i >= n - r) || (j < r) || (j >= n - r) ||
                             (k < r) || (k >= n - r);
        border_is_on_faces &= on_face;
      }
      const bool border_is_valid =
          border_is_unique && regions_are_disjoint && border_is_on_faces;
      all_regions_are_valid &= border_is_valid;
    }
  }
  REQUIRE(all_regions_are_valid);
}

TEST_CASE("for_each_border falls back to the whole owned region when n <= 2*r",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/2);
  const auto sz = u.local_size();
  int count = 0;
  field::for_each_border(u, 2, [&](int, int, int) { ++count; });
  REQUIRE(count == sz[0] * sz[1] * sz[2]);
}

TEST_CASE("stencil over inner region only reads owned cells (no halo dependency)",
          "[field][brick_iteration]") {
  auto u = make_brick(6, /*hw=*/1);
  const auto sz = u.local_size();
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) u(i, j, k) = i + 10 * j + 100 * k;
    }
  }

  double accum = 0.0;
  field::for_each_inner(u, 1, [&](int i, int j, int k) {
    const double xx = u(i + 1, j, k) - 2 * u(i, j, k) + u(i - 1, j, k);
    const double yy = u(i, j + 1, k) - 2 * u(i, j, k) + u(i, j - 1, k);
    const double zz = u(i, j, k + 1) - 2 * u(i, j, k) + u(i, j, k - 1);
    accum += xx + yy + zz;
  });
  REQUIRE(accum == Catch::Approx(0.0));
}

namespace {

double apply_d2(const pfc::data::Field<double, pfc::HostSpace> &u, int i, int j,
                int k, const pfc::field::fd::EvenCentralD2View &st) {
  const int r = st.half_width;
  const double inv = 1.0 / static_cast<double>(st.denom);
  auto axis = [&](int di, int dj, int dk) {
    double s = st.coeffs[0] * u(i, j, k);
    for (int o = 1; o <= r; ++o) {
      s += st.coeffs[o] * (u(i + o * di, j + o * dj, k + o * dk) +
                           u(i - o * di, j - o * dj, k - o * dk));
    }
    return s * inv;
  };
  return axis(1, 0, 0) + axis(0, 1, 0) + axis(0, 0, 1);
}

void fill_periodic_owned_and_halo(pfc::data::Field<double, pfc::HostSpace> &u,
                                  int r) {
  const auto sz = u.local_size();
  const int n = sz[0];
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        u(i, j, k) = static_cast<double>(i + 3 * j + 5 * k);
      }
    }
  }
  for (int k = -r; k < n + r; ++k) {
    for (int j = -r; j < n + r; ++j) {
      for (int i = -r; i < n + r; ++i) {
        const bool halo = i < 0 || i >= n || j < 0 || j >= n || k < 0 || k >= n;
        if (!halo) continue;
        const int ii = (i % n + n) % n;
        const int jj = (j % n + n) % n;
        const int kk = (k % n + n) % n;
        u(i, j, k) = u(ii, jj, kk);
      }
    }
  }
}

} // namespace

TEST_CASE("inner D2 at widths 1,2,4,6,10 does not read poisoned halo",
          "[field][brick_iteration][fd108]") {
  for (int r : {1, 2, 4, 6, 10}) {
    const int n = 32;
    auto u = make_brick(n, r);
    pfc::field::fd::EvenCentralD2View st{};
    REQUIRE(pfc::field::fd::lookup_even_central_d2(2 * r, &st));
    for (int k = 0; k < n; ++k)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) u(i, j, k) = static_cast<double>(i + j + k);
    const double nan = std::nan("");
    for (int k = -r; k < n + r; ++k) {
      for (int j = -r; j < n + r; ++j) {
        for (int i = -r; i < n + r; ++i) {
          const bool halo = i < 0 || i >= n || j < 0 || j >= n || k < 0 || k >= n;
          if (halo) u(i, j, k) = nan;
        }
      }
    }
    bool finite = true;
    field::for_each_inner(u, r, [&](int i, int j, int k) {
      finite &= std::isfinite(apply_d2(u, i, j, k, st));
    });
    REQUIRE(finite);
  }
}

TEST_CASE("inner+border D2 matches unsplit owned stencil at #108 halo widths",
          "[field][brick_iteration][fd108]") {
  for (int r : {1, 2, 4, 6, 10}) {
    const int n = 32;
    auto u = make_brick(n, r);
    pfc::field::fd::EvenCentralD2View st{};
    REQUIRE(pfc::field::fd::lookup_even_central_d2(2 * r, &st));
    fill_periodic_owned_and_halo(u, r);
    std::vector<double> split(static_cast<std::size_t>(n * n * n), 0.0);
    std::vector<double> unsplit(split.size(), 0.0);
    auto lin = [n](int i, int j, int k) {
      return static_cast<std::size_t>(i + n * (j + n * k));
    };
    field::for_each_inner(u, r, [&](int i, int j, int k) {
      split[lin(i, j, k)] = apply_d2(u, i, j, k, st);
    });
    field::for_each_border(u, r, [&](int i, int j, int k) {
      split[lin(i, j, k)] = apply_d2(u, i, j, k, st);
    });
    field::for_each_owned(u, [&](int i, int j, int k) {
      unsplit[lin(i, j, k)] = apply_d2(u, i, j, k, st);
    });
    REQUIRE(split == unsplit);
  }
}

TEST_CASE("for_each_owned_omp visits same set as serial counterpart",
          "[field][brick_iteration]") {
  auto u = make_brick(4, /*hw=*/0);
  const auto sz = u.local_size();
  std::vector<int> hits(sz[0] * sz[1] * sz[2], 0);
  field::for_each_owned_omp(u, [&](int i, int j, int k) {
    const std::size_t lin = static_cast<std::size_t>(i) +
                            static_cast<std::size_t>(j) * sz[0] +
                            static_cast<std::size_t>(k) * sz[0] * sz[1];
    hits[lin] = 1;
  });
  bool all_cells_visited = true;
  for (int v : hits) all_cells_visited &= v == 1;
  REQUIRE(all_cells_visited);
}

TEST_CASE("for_each_coords fills every owned cell from physical coordinates",
          "[field][brick_iteration]") {
  auto u = make_brick(3, /*hw=*/1);
  bool values_match = true;
  field::for_each_coords(
      u, [](double x, double y, double z, double &v) { v = x + 2.0 * y + 3.0 * z; });
  field::for_each_owned(u, [&](int i, int j, int k) {
    const auto xyz = u.coords(i, j, k);
    values_match &=
        u(i, j, k) == Catch::Approx(xyz[0] + 2.0 * xyz[1] + 3.0 * xyz[2]);
  });
  REQUIRE(values_match);
}

TEST_CASE("data::Field operator() for Int3", "[field][brick_iteration]") {
  auto u = make_brick(3, /*hw=*/1);
  const pfc::Int3 idx{1, 2, 0};
  u(idx) = 42.0;
  REQUIRE(u(idx) == Catch::Approx(42.0));
  u(idx) = -1.5;
  REQUIRE(u(idx) == Catch::Approx(-1.5));
}
