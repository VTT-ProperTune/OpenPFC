// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * Manufactured even/odd reflection for padded fields.
 *
 * The expected value is the mirror map written out below. It does not call
 * the fill. Multi-rank cases go through StagePreparationService so the
 * exchange runs before the fill. A faces-only exchange leaves the physical
 * × periodic corner untouched; that corner is the check that reflection
 * reads the exchanged tangential halo rather than an earlier poison value.
 *
 * Even reflection is checked against a zero normal difference. A separate
 * case shows a nonzero tangential face current on that same field: the
 * scalar mirror is not `WallPolicy::Zero`.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/decomposition_neighbors.hpp>
#include <openpfc/kernel/decomposition/dirichlet_ghosts.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/kernel/field/face_current.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using pfc::HostSpace;
using pfc::communication::apply_reflection_ghosts;
using pfc::communication::BoundaryCentering;
using pfc::communication::reflection_walls;
using pfc::communication::ReflectionExtent;
using pfc::communication::ReflectionParity;
using pfc::communication::ReflectionWalls;

namespace {

struct AxisMap {
  int index = 0;
  int sign = 1;
};

int wrap_index(int g, int n) {
  const int r = g % n;
  return r < 0 ? r + n : r;
}

/// Independent mirror. Periodic axes wrap. Odd parity contributes a sign.
AxisMap map_axis(int g, int n, bool periodic, BoundaryCentering centering,
                 ReflectionParity parity) {
  if (g >= 0 && g < n) {
    return {g, 1};
  }
  if (periodic) {
    return {wrap_index(g, n), 1};
  }
  const bool cell = centering == BoundaryCentering::CellFace;
  const int mirrored =
      g < 0 ? (cell ? -1 - g : -g) : (cell ? 2 * n - 1 - g : 2 * (n - 1) - g);
  const int sign = parity == ReflectionParity::Odd ? -1 : 1;
  return {mirrored, sign};
}

double sample_value(int x, int y, int z) {
  return static_cast<double>(3 * x + 5 * y + 7 * z);
}

double expected_value(int gx, int gy, int gz, pfc::Int3 n, pfc::Bool3 periodic,
                      BoundaryCentering centering, ReflectionParity parity) {
  const AxisMap x = map_axis(gx, n[0], periodic[0], centering, parity);
  const AxisMap y = map_axis(gy, n[1], periodic[1], centering, parity);
  const AxisMap z = map_axis(gz, n[2], periodic[2], centering, parity);
  return static_cast<double>(x.sign * y.sign * z.sign) *
         sample_value(x.index, y.index, z.index);
}

void poison_ghosts(pfc::data::Field<double, HostSpace> &u) {
  const int hw = u.storage_halo();
  const auto n = u.local_size();
  for (int k = -hw; k < n[2] + hw; ++k) {
    for (int j = -hw; j < n[1] + hw; ++j) {
      for (int i = -hw; i < n[0] + hw; ++i) {
        const bool owned =
            i >= 0 && i < n[0] && j >= 0 && j < n[1] && k >= 0 && k < n[2];
        if (!owned) {
          u(i, j, k) = -1234567.0;
        }
      }
    }
  }
}

void fill_owned(pfc::data::Field<double, HostSpace> &u) {
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = u.global(i, j, k);
        u(i, j, k) = sample_value(g[0], g[1], g[2]);
      }
    }
  }
}

void require_matches(const pfc::data::Field<double, HostSpace> &u,
                     BoundaryCentering centering, ReflectionParity parity,
                     pfc::Bool3 periodic) {
  const int hw = u.storage_halo();
  const auto n = u.local_size();
  const auto gsz = u.global_size();
  for (int k = -hw; k < n[2] + hw; ++k) {
    for (int j = -hw; j < n[1] + hw; ++j) {
      for (int i = -hw; i < n[0] + hw; ++i) {
        const auto g = u.global(i, j, k);
        const double expect =
            expected_value(g[0], g[1], g[2], gsz, periodic, centering, parity);
        INFO("local " << i << " " << j << " " << k << " global " << g[0] << " "
                      << g[1] << " " << g[2]);
        REQUIRE(u(i, j, k) == expect);
      }
    }
  }
}

void require_owned_unchanged(const pfc::data::Field<double, HostSpace> &u) {
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        const auto g = u.global(i, j, k);
        REQUIRE(u(i, j, k) == sample_value(g[0], g[1], g[2]));
      }
    }
  }
}

pfc::data::Field<double, HostSpace> make_serial_field(pfc::Bool3 periodic, int hw) {
  auto domain = pfc::domain::create(pfc::GridSize({8, 6, 5}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}), periodic);
  auto decomp = pfc::decomposition::create(domain, 1);
  return pfc::data::field_from_subdomain<double>(decomp, 0, hw);
}

} // namespace

TEST_CASE("reflection ghosts match the mirror map at several halo widths",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  const pfc::Bool3 closed{false, false, false};
  for (const int hw : {1, 3}) {
    for (const auto centering :
         {BoundaryCentering::Node, BoundaryCentering::CellFace}) {
      for (const auto parity : {ReflectionParity::Even, ReflectionParity::Odd}) {
        auto u = make_serial_field(closed, hw);
        fill_owned(u);
        poison_ghosts(u);
        const double owned_checksum = u(1, 1, 1);
        apply_reflection_ghosts(u, reflection_walls<double>(parity, centering, 0.0));
        require_matches(u, centering, parity, closed);
        require_owned_unchanged(u);
        REQUIRE(u(1, 1, 1) == owned_checksum);
      }
    }
  }
}

TEST_CASE("odd corner composes X then Y through nonzero wall values",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  auto u = make_serial_field(pfc::Bool3{false, false, true}, 2);
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        u(i, j, k) = 1000.0 * static_cast<double>(i) + static_cast<double>(j);
      }
    }
  }
  poison_ghosts(u);
  ReflectionWalls<double> walls;
  walls.xm = {true, ReflectionParity::Odd, BoundaryCentering::Node, 3.0};
  walls.ym = {true, ReflectionParity::Odd, BoundaryCentering::Node, 5.0};
  apply_reflection_ghosts(u, walls);

  // Source of (-1,-1) is (1,1) = 1001. X then Y:
  // 2*5 - (2*3 - 1001) = 1005.
  REQUIRE(u(-1, -1, 0) == 1005.0);
  REQUIRE(u(-1, 0, 0) == 2.0 * 3.0 - u(1, 0, 0));
  // j = -1 is the enabled low wall. The disabled high wall stays poison.
  REQUIRE(u(2, -1, 0) == 2.0 * 5.0 - u(2, 1, 0));
  REQUIRE(u(2, n[1], 0) == -1234567.0);
}

TEST_CASE("node centering rejects a halo as wide as the axis",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  auto domain = pfc::domain::create(
      pfc::GridSize({4, 4, 4}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, true, true});
  auto decomp = pfc::decomposition::create(domain, 1);
  auto node = pfc::data::field_from_subdomain<double>(decomp, 0, 4);
  fill_owned(node);
  REQUIRE_THROWS_AS(apply_reflection_ghosts(
                        node, reflection_walls<double>(ReflectionParity::Even,
                                                       BoundaryCentering::Node)),
                    std::out_of_range);

  auto cell = pfc::data::field_from_subdomain<double>(decomp, 0, 4);
  fill_owned(cell);
  poison_ghosts(cell);
  apply_reflection_ghosts(
      cell,
      reflection_walls<double>(ReflectionParity::Even, BoundaryCentering::CellFace));
  REQUIRE(cell(-4, 0, 0) == cell(3, 0, 0));
  REQUIRE(cell(4, 0, 0) == cell(3, 0, 0));
}

TEST_CASE("a disabled physical side and a periodic axis are left to the exchange",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  auto u = make_serial_field(pfc::Bool3{false, true, true}, 1);
  fill_owned(u);
  poison_ghosts(u);
  ReflectionWalls<double> walls;
  walls.xp = {true, ReflectionParity::Even, BoundaryCentering::Node, 0.0};
  apply_reflection_ghosts(u, walls);
  REQUIRE(u(-1, 0, 0) == -1234567.0);
  REQUIRE(u(8, 0, 0) == u(6, 0, 0));
  REQUIRE(u(0, -1, 0) == -1234567.0);
  REQUIRE(u(0, 0, -1) == -1234567.0);
}

TEST_CASE("even reflection zeroes the normal difference and not a tangential flux",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }

  constexpr int n = 8;
  auto domain = pfc::domain::create(
      pfc::GridSize({n, n, 4}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, false, true});
  auto decomp = pfc::decomposition::create(domain, 1);
  auto u = pfc::data::field_from_subdomain<double>(decomp, 0, 1);
  const auto sz = u.local_size();
  for (int k = 0; k < sz[2]; ++k) {
    for (int j = 0; j < sz[1]; ++j) {
      for (int i = 0; i < sz[0]; ++i) {
        const double x = static_cast<double>(i) / static_cast<double>(n - 1);
        u(i, j, k) = std::cos(std::numbers::pi * x) + 0.5 * static_cast<double>(j);
      }
    }
  }
  ReflectionWalls<double> walls;
  walls.xm = {true, ReflectionParity::Even, BoundaryCentering::Node, 0.0};
  walls.xp = {true, ReflectionParity::Even, BoundaryCentering::Node, 0.0};
  apply_reflection_ghosts(u, walls);
  REQUIRE(u(-1, 2, 0) == u(1, 2, 0));
  REQUIRE(u(n, 2, 0) == u(n - 2, 2, 0));
  REQUIRE(u(1, 2, 0) - u(-1, 2, 0) == 0.0);
  REQUIRE(u(n, 2, 0) - u(n - 2, 2, 0) == 0.0);

  // Tangential difference on the reflected low-x sample. The normal
  // difference above is zero. This current is not.
  const double tangential = u(0, 3, 0) - u(0, 2, 0);
  REQUIRE(tangential == 0.5);

  const int nx = n;
  const int ny = n;
  const int nz = 4;
  std::vector<double> gx(pfc::field::fd::n_xfaces(nx, ny, nz), tangential);
  std::vector<double> gy(pfc::field::fd::n_yfaces(nx, ny, nz), 0.0);
  std::vector<double> gz(pfc::field::fd::n_zfaces(nx, ny, nz), 0.0);
  REQUIRE(gx[pfc::field::fd::xface_index(nx, ny, 0, 2, 0)] == 0.5);
  pfc::field::fd::apply_walls(gx, gy, gz, nx, ny, nz, {true, true, true},
                              pfc::field::fd::closed_walls());
  REQUIRE(gx[pfc::field::fd::xface_index(nx, ny, 0, 2, 0)] == 0.0);
}

TEST_CASE("stage preparation reflects after the halo exchange",
          "[MPI][reflection]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 && size != 2 && size != 4) {
    return;
  }

  const pfc::Int3 grid = size == 4   ? pfc::Int3{2, 2, 1}
                         : size == 2 ? pfc::Int3{2, 1, 1}
                                     : pfc::Int3{1, 1, 1};
  const pfc::Bool3 periodic{false, false, true};
  auto domain = pfc::domain::create(pfc::GridSize({8, 8, 4}),
                                    pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                    pfc::GridSpacing({1.0, 1.0, 1.0}), periodic);
  auto decomp = pfc::decomposition::create(domain, grid);
  constexpr int hw = 2;

  auto run = [&](pfc::comm::HaloConnectivity connectivity,
                 BoundaryCentering centering, ReflectionParity parity,
                 bool boundary) {
    auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
    fill_owned(u);
    poison_ghosts(u);
    pfc::comm::HaloExchangeOptions opt;
    opt.connectivity = connectivity;
    // Physical faces have no MPI neighbour. The exchanger's direction
    // agreement rejects that peer, so those faces stay out of the set and
    // reflection fills them after the exchange.
    const auto base = connectivity == pfc::comm::HaloConnectivity::Full
                          ? pfc::halo::presets::Full3D()
                          : pfc::halo::presets::Axes3D();
    opt.selector = [&](int r) {
      std::vector<pfc::Int3> kept;
      kept.reserve(base.dirs.size());
      for (const auto &d : base.dirs) {
        if (pfc::decomposition::get_neighbor_rank(decomp, r, d) >= 0) {
          kept.push_back(d);
        }
      }
      return pfc::halo::HaloDirectionSet(std::move(kept));
    };
    pfc::comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                    opt);
    pfc::communication::StagePreparationService<double> prep;
    prep.bind("u", halo);
    prep.set_boundary_hook([&](std::string_view) { poison_ghosts(u); });
    if (boundary) {
      prep.set_reflection("u", reflection_walls<double>(parity, centering, 0.0), u);
    }
    pfc::communication::StagePreparationRequirements req;
    req.needs_halo_exchange = true;
    req.needs_boundary_update = true;
    req.ordering = pfc::communication::BoundaryHaloOrder::BoundaryThenHalo;
    const std::string_view fields[] = {"u"};
    prep.prepare(req, fields);
    return u;
  };

  auto faces = run(pfc::comm::HaloConnectivity::Faces, BoundaryCentering::Node,
                   ReflectionParity::Even, true);
  const auto n = faces.local_size();
  const auto low = faces.box().low;
  if (low[0] == 0) {
    const int j = 0;
    const int k_halo = -1;
    const auto g = faces.global(1, j, k_halo);
    const double expect = expected_value(
        faces.global(-1, j, k_halo)[0], g[1], g[2], faces.global_size(), periodic,
        BoundaryCentering::Node, ReflectionParity::Even);
    INFO("faces corner rank " << rank);
    REQUIRE(faces(-1, j, k_halo) == expect);
    REQUIRE(faces(-1, j, k_halo) != -1234567.0);
  }
  if (low[0] + n[0] == faces.global_size()[0]) {
    const int i_ghost = n[0];
    const auto gsrc = faces.global(n[0] - 2, 0, -1);
    const double expect = expected_value(
        faces.global(i_ghost, 0, -1)[0], gsrc[1], gsrc[2], faces.global_size(),
        periodic, BoundaryCentering::Node, ReflectionParity::Even);
    REQUIRE(faces(i_ghost, 0, -1) == expect);
  }

  for (const auto centering :
       {BoundaryCentering::Node, BoundaryCentering::CellFace}) {
    for (const auto parity : {ReflectionParity::Even, ReflectionParity::Odd}) {
      auto full = run(pfc::comm::HaloConnectivity::Full, centering, parity, true);
      require_matches(full, centering, parity, periodic);
      require_owned_unchanged(full);
    }
  }

  auto untouched = run(pfc::comm::HaloConnectivity::Full, BoundaryCentering::Node,
                       ReflectionParity::Even, false);
  if (low[0] == 0) {
    REQUIRE(untouched(-1, 0, 0) == -1234567.0);
  }
}

TEST_CASE("linear cell-face extension reproduces an affine profile",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;
  for (int hw : {1, 2, 4}) {
    auto u = make_serial_field({false, true, true}, hw);
    poison_ghosts(u);
    const auto n = u.local_size();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i) u(i, j, k) = 3.0 + 2.0 * (i + 0.5);
    ReflectionWalls<double> walls;
    walls.xm = {true, ReflectionParity::Odd, BoundaryCentering::CellFace, 3.0,
                pfc::communication::BoundaryExtension::Linear};
    walls.xp = walls.xm;
    walls.xp.value = 3.0 + 2.0 * n[0];
    apply_reflection_ghosts(u, walls);
    for (int i = -hw; i < n[0] + hw; ++i)
      REQUIRE(u(i, 0, 0) == 3.0 + 2.0 * (i + 0.5));

    const std::vector<double> accepted(u.data(), u.data() + u.size());
    walls.xm.centering = BoundaryCentering::Node;
    REQUIRE_THROWS_AS(apply_reflection_ghosts(u, walls), std::invalid_argument);
    REQUIRE(std::vector<double>(u.data(), u.data() + u.size()) == accepted);
    walls.xm.centering = BoundaryCentering::CellFace;
    walls.xp.parity = ReflectionParity::Even;
    REQUIRE_THROWS_AS(apply_reflection_ghosts(u, walls), std::invalid_argument);
    REQUIRE(std::vector<double>(u.data(), u.data() + u.size()) == accepted);
  }
}

TEST_CASE("linear staged walls match tensor-product extrapolation",
          "[MPI][reflection]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1 && size != 2 && size != 4) return;
  const pfc::Int3 grid = size == 4   ? pfc::Int3{2, 2, 1}
                         : size == 2 ? pfc::Int3{2, 1, 1}
                                     : pfc::Int3{1, 1, 1};
  auto domain = pfc::domain::create(
      pfc::GridSize({8, 8, 4}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, false, false});
  auto decomp = pfc::decomposition::create(domain, grid);
  for (int hw : {1, 2, 4}) {
    auto u = pfc::data::field_from_subdomain<double>(decomp, rank, hw);
    fill_owned(u);
    poison_ghosts(u);
    pfc::comm::HaloExchangeOptions opt;
    opt.connectivity = pfc::comm::HaloConnectivity::Full;
    opt.selector = [&](int r) {
      std::vector<pfc::Int3> kept;
      for (const auto &d : pfc::halo::presets::Full3D().dirs)
        if (pfc::decomposition::get_neighbor_rank(decomp, r, d) >= 0)
          kept.push_back(d);
      return pfc::halo::HaloDirectionSet(std::move(kept));
    };
    pfc::comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD,
                                                    opt);
    auto walls =
        reflection_walls<double>(ReflectionParity::Odd, BoundaryCentering::CellFace);
    auto set = [](auto &side, double value) {
      side.extension = pfc::communication::BoundaryExtension::Linear;
      side.value = value;
    };
    set(walls.xm, 2);
    set(walls.xp, 3);
    set(walls.ym, 5);
    set(walls.yp, 7);
    set(walls.zm, 11);
    set(walls.zp, 13);
    pfc::communication::StagePreparationService<double> prep;
    prep.bind("u", halo);
    prep.set_reflection("u", walls, u);
    pfc::communication::StagePreparationRequirements req;
    req.needs_halo_exchange = req.needs_boundary_update = true;
    const std::string_view fields[] = {"u"};
    prep.prepare(req, fields);
    const auto n = u.local_size();
    const auto ng = u.global_size();
    const double wall_values[3][2] = {{2, 3}, {5, 7}, {11, 13}};
    for (int k = -hw; k < n[2] + hw; ++k)
      for (int j = -hw; j < n[1] + hw; ++j)
        for (int i = -hw; i < n[0] + hw; ++i) {
          const auto g = u.global(i, j, k);
          int nearest[3];
          double a[3] = {1, 1, 1}, b[3] = {0, 0, 0};
          for (int axis = 0; axis < 3; ++axis) {
            nearest[axis] = std::clamp(g[axis], 0, ng[axis] - 1);
            // Lagrange interpolation through x=0 (wall) and x=1/2 (sample).
            // The ghost coordinate is 1/2-layer; its sample weight is 1-2*layer.
            if (g[axis] != nearest[axis]) {
              const int layer = std::abs(g[axis] - nearest[axis]);
              a[axis] = 1 - 2 * layer;
              b[axis] = 2 * layer * wall_values[axis][g[axis] >= ng[axis]];
            }
          }
          const double oracle =
              a[0] * a[1] * a[2] * sample_value(nearest[0], nearest[1], nearest[2]) +
              b[0] * a[1] * a[2] + b[1] * a[2] + b[2];
          INFO("rank " << rank << " local " << i << "," << j << "," << k);
          REQUIRE(u(i, j, k) == oracle);
        }
    require_owned_unchanged(u);
    // This prescribed wall is incompatible with the owned affine gradient.
    if (u.box().low[0] == 0 && hw >= 2) {
      REQUIRE(u(-2, 0, 0) == 8.0 - 3.0 * u(0, 0, 0));
      REQUIRE(u(-2, 0, 0) != 4.0 - u(1, 0, 0));
    }
  }
}

TEST_CASE("linear and mirror walls preserve a sequential cell-face donor",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) return;
  constexpr double ub = -0.375;
  for (int hw : {2, 4}) {
    for (bool quadrant : {false, true}) {
      auto expected = make_serial_field({false, false, true}, hw);
      auto actual = make_serial_field({false, false, true}, hw);
      const auto n = actual.local_size();
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          actual(i, j, 0) = expected(i, j, 0) = i * i + 2 * j * j * j + 0.5 * i * j;
      // Frozen two-pass donor: first X for owned Y, then Y across padded X.
      for (int j = 0; j < n[1]; ++j)
        for (int k = 1; k <= hw; ++k) {
          expected(-k, j, 0) = quadrant
                                   ? expected(k - 1, j, 0)
                                   : 2 * k * ub - (2 * k - 1) * expected(0, j, 0);
          expected(n[0] - 1 + k, j, 0) =
              2 * k * ub - (2 * k - 1) * expected(n[0] - 1, j, 0);
        }
      for (int i = -hw; i < n[0] + hw; ++i)
        for (int k = 1; k <= hw; ++k) {
          expected(i, -k, 0) = quadrant
                                   ? expected(i, k - 1, 0)
                                   : 2 * k * ub - (2 * k - 1) * expected(i, 0, 0);
          expected(i, n[1] - 1 + k, 0) =
              2 * k * ub - (2 * k - 1) * expected(i, n[1] - 1, 0);
        }
      ReflectionWalls<double> walls;
      walls.xp =
          walls.yp = {true, ReflectionParity::Odd, BoundaryCentering::CellFace, ub,
                      pfc::communication::BoundaryExtension::Linear};
      if (quadrant) {
        walls.xm =
            walls.ym = {true, ReflectionParity::Even, BoundaryCentering::CellFace};
      } else {
        walls.xm = walls.ym = walls.xp;
      }
      apply_reflection_ghosts(actual, walls);
      for (int j = -hw; j < n[1] + hw; ++j)
        for (int i = -hw; i < n[0] + hw; ++i)
          REQUIRE(actual(i, j, 0) == expected(i, j, 0));
    }
  }
}

TEST_CASE("reflection slabs match a full-volume sample",
          "[kernel][field][reflection][unit]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return;
  }
  const auto volume = [](pfc::data::Field<double, HostSpace> &u,
                         const ReflectionWalls<double> &walls,
                         ReflectionExtent extent, bool skip_disabled) {
    auto geom =
        pfc::communication::detail::build_reflection_geometry(u, walls, extent);
    geom.skip_disabled = skip_disabled ? 1 : 0;
    pfc::communication::detail::validate_reflection_geometry(geom);
    const int hw = geom.halo;
    for (int k = -hw; k < geom.nloc[2] + hw; ++k) {
      for (int j = -hw; j < geom.nloc[1] + hw; ++j) {
        for (int i = -hw; i < geom.nloc[0] + hw; ++i) {
          const auto sample =
              pfc::communication::detail::reflection_sample(geom, i, j, k);
          if (!sample.write) {
            continue;
          }
          u(i, j, k) =
              sample.scale * u(sample.si, sample.sj, sample.sk) + sample.offset;
        }
      }
    }
  };
  const auto same = [](const pfc::data::Field<double, HostSpace> &a,
                       const pfc::data::Field<double, HostSpace> &b) {
    REQUIRE(a.size() == b.size());
    REQUIRE(std::equal(a.data(), a.data() + a.size(), b.data()));
  };

  for (int hw : {1, 2, 4}) {
    auto domain = pfc::domain::create(
        pfc::GridSize({16, 12, 1}), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
        pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, false, false});
    auto decomp = pfc::decomposition::create(domain, 1);
    auto slab = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
    auto walked = pfc::data::field_from_subdomain<double>(decomp, 0, hw);
    fill_owned(slab);
    fill_owned(walked);
    poison_ghosts(slab);
    poison_ghosts(walked);
    ReflectionWalls<double> walls;
    const pfc::communication::ReflectionSide<double> even{
        true, ReflectionParity::Even, BoundaryCentering::CellFace, 0.0,
        pfc::communication::BoundaryExtension::Reflection};
    const pfc::communication::ReflectionSide<double> far{
        true, ReflectionParity::Odd, BoundaryCentering::CellFace, -0.375,
        pfc::communication::BoundaryExtension::Linear};
    walls.xm = walls.ym = even;
    walls.xp = walls.yp = far;
    volume(walked, walls, ReflectionExtent::PhysicalFaces, false);
    apply_reflection_ghosts(slab, walls);
    same(slab, walked);
    REQUIRE(slab(0, 0, -1) == -1234567.0);
    REQUIRE(slab(-1, 0, -1) == -1234567.0);
    REQUIRE(slab(0, 0, 1) == -1234567.0);
    const auto owned = slab.local_size();
    REQUIRE(slab(owned[0] / 2, owned[1] / 2, 0) ==
            sample_value(owned[0] / 2, owned[1] / 2, 0));
  }

  {
    auto box = make_serial_field({false, false, false}, 2);
    auto walked = make_serial_field({false, false, false}, 2);
    fill_owned(box);
    fill_owned(walked);
    poison_ghosts(box);
    poison_ghosts(walked);
    const auto walls =
        reflection_walls<double>(ReflectionParity::Even, BoundaryCentering::Node);
    volume(walked, walls, ReflectionExtent::PhysicalFaces, false);
    apply_reflection_ghosts(box, walls);
    same(box, walked);
  }

  {
    auto direct = make_serial_field({true, false, true}, 3);
    auto walked = make_serial_field({true, false, true}, 3);
    fill_owned(direct);
    fill_owned(walked);
    poison_ghosts(direct);
    poison_ghosts(walked);
    ReflectionWalls<double> walls;
    walls.ym = {true, ReflectionParity::Odd, BoundaryCentering::Node, 1.25};
    walls.yp = {true, ReflectionParity::Odd, BoundaryCentering::Node, -0.5};
    volume(walked, walls, ReflectionExtent::AllOutside, true);
    pfc::communication::apply_dirichlet_ghosts(direct, 1, 1.25, -0.5);
    same(direct, walked);
  }
}
