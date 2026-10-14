// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file bench_reflection_ghosts.cpp
 * @brief Slab reflection against the previous full padded walk.
 *
 * The volume walk is the old host loop, kept here so a release run can
 * report both costs from one binary. It is not a second library path.
 *
 *   ./openpfc-tests "[reflection][benchmark]"
 */

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/reflection_ghosts.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using pfc::HostSpace;

namespace {

void fill_owned(pfc::data::Field<double, HostSpace> &u) {
  const auto n = u.local_size();
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      for (int i = 0; i < n[0]; ++i) {
        u(i, j, k) = 0.1 * i + 0.01 * j + 0.001 * k;
      }
    }
  }
}

pfc::data::Field<double, HostSpace> make_field(pfc::Int3 n, int halo) {
  auto domain = pfc::domain::create(
      pfc::GridSize(n), pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
      pfc::GridSpacing({1.0, 1.0, 1.0}), pfc::Bool3{false, false, false});
  auto decomp = pfc::decomposition::create(domain, 1);
  auto field = pfc::data::field_from_subdomain<double>(decomp, 0, halo);
  fill_owned(field);
  return field;
}

void volume_walk(pfc::data::Field<double, HostSpace> &u,
                 const pfc::communication::ReflectionWalls<double> &walls) {
  const auto geom = pfc::communication::detail::build_reflection_geometry(
      u, walls, pfc::communication::ReflectionExtent::PhysicalFaces);
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
}

pfc::communication::ReflectionWalls<double> closed_xy() {
  pfc::communication::ReflectionWalls<double> walls;
  const pfc::communication::ReflectionSide<double> even{
      true, pfc::communication::ReflectionParity::Even,
      pfc::communication::BoundaryCentering::CellFace, 0.0,
      pfc::communication::BoundaryExtension::Reflection};
  const pfc::communication::ReflectionSide<double> far{
      true, pfc::communication::ReflectionParity::Odd,
      pfc::communication::BoundaryCentering::CellFace, -0.375,
      pfc::communication::BoundaryExtension::Linear};
  walls.xm = walls.ym = even;
  walls.xp = walls.yp = far;
  return walls;
}

} // namespace

TEST_CASE("reflection slab fill versus the full padded walk",
          "[reflection][benchmark]") {
  const auto walls = closed_xy();
  auto thin = make_field({64, 64, 1}, 4);
  auto cube = make_field({48, 48, 48}, 2);
  BENCHMARK("volume walk 64x64x1 halo 4") {
    volume_walk(thin, walls);
    return thin(-1, 0, 0);
  };
  BENCHMARK("slab fill 64x64x1 halo 4") {
    pfc::communication::apply_reflection_ghosts(thin, walls);
    return thin(-1, 0, 0);
  };
  const auto box = pfc::communication::reflection_walls<double>(
      pfc::communication::ReflectionParity::Even,
      pfc::communication::BoundaryCentering::CellFace);
  BENCHMARK("volume walk 48^3 halo 2") {
    volume_walk(cube, box);
    return cube(-1, -1, -1);
  };
  BENCHMARK("slab fill 48^3 halo 2") {
    pfc::communication::apply_reflection_ghosts(cube, box);
    return cube(-1, -1, -1);
  };
}
