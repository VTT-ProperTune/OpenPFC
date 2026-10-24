// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/frontend/ui/from_json_world_time.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>

namespace full_halo_fd6 {
constexpr int width = 3;
constexpr double poison = -987654321.0;
inline double pattern(int field, int stage, const pfc::Int3 &g) {
  return 1.0 + field * 0.25 + stage * 16777216.0 + g[0] + 128.0 * g[1] +
         16384.0 * g[2];
}
inline pfc::Int3 process_grid(int ranks) {
  switch (ranks) {
  case 1: return {1, 1, 1};
  case 2: return {2, 1, 1};
  case 4: return {2, 2, 1};
  case 8: return {2, 2, 2};
  default: throw std::invalid_argument("Full FD6 test requires 1/2/4/8 ranks");
  }
}
template <class Space>
void initialize(pfc::data::Field<double, Space> &u, int field, int stage) {
  u.with_host_view([&](double *data, std::size_t) {
    std::fill(data, data + u.size(), poison);
    const auto n = u.size3();
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
          data[u.idx(i, j, k)] = pattern(field, stage, u.global(i, j, k));
  });
}

// Independent global-coordinate oracle: no exchange geometry/helper is used.
// Physical exterior ghosts remain caller-owned; periodic ghosts wrap globally.
template <class Space>
void verify(pfc::data::Field<double, Space> &u, int field, int stage, bool slab) {
  std::array<int, 27> covered{};
  std::size_t mismatch = 0;
  u.with_host_view([&](double *data, std::size_t count) {
    const auto n = u.size3();
    for (int k = -width; k < n[2] + width; ++k)
      for (int j = -width; j < n[1] + width; ++j)
        for (int i = -width; i < n[0] + width; ++i) {
          auto g = u.global(i, j, k);
          bool physical_exterior = slab && k != 0;
          const pfc::Int3 local{i, j, k};
          pfc::Int3 dir{};
          for (int axis = 0; axis < 3; ++axis) {
            dir[axis] = local[axis] < 0 ? -1 : local[axis] >= n[axis] ? 1 : 0;
            if (u.domain().periodic[axis])
              g[axis] = (g[axis] % u.global_size()[axis] + u.global_size()[axis]) %
                        u.global_size()[axis];
            else if (g[axis] < 0 || g[axis] >= u.global_size()[axis])
              physical_exterior = true;
          }
          const auto index = u.idx(i, j, k);
          REQUIRE(index < count);
          const double expected =
              physical_exterior ? poison : pattern(field, stage, g);
          mismatch += data[index] != expected;
          ++covered[(dir[0] + 1) * 9 + (dir[1] + 1) * 3 + dir[2] + 1];
        }
  });
  INFO("field=" << field << " stage=" << stage << " mismatched cells=" << mismatch);
  REQUIRE(mismatch == 0);
  for (int visits : covered) REQUIRE(visits > 0);
}

template <class Space, class Producer>
void run(const pfc::Bool3 &periodic, bool slab, Producer producer) {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  auto grid = process_grid(ranks);
  if (slab) grid = {ranks, 1, 1};
  // Uneven rank extents and unequal physical spacing; smallest owned axis >=3.
  const pfc::Int3 extent =
      slab ? pfc::Int3{7 * ranks + 1, 11, 1} : pfc::Int3{19, 15, 11};
  auto domain = pfc::domain::with_spacing(extent, {0.13, 0.27, 0.41}, periodic);
  if (!slab) {
    pfc::Real3 upper{};
    for (int axis = 0; axis < 3; ++axis)
      upper[axis] = domain.spacing[axis] * (extent[axis] - (periodic[axis] ? 0 : 1));
    const nlohmann::json config{{"domain", {{"size", extent}, {"lower", {0,0,0}},
                                            {"upper", upper}, {"periodic", periodic}}}};
    domain = pfc::ui::from_json<pfc::Domain>(config);
  }
  auto decomp = pfc::decomposition::create(domain, grid);
  pfc::data::Field<double, Space> u(
      domain, pfc::decomposition::local_box(decomp, rank), width);
  pfc::data::Field<double, Space> v(
      domain, pfc::decomposition::local_box(decomp, rank), width);
  pfc::comm::HaloExchangeOptions options;
  options.connectivity = pfc::comm::HaloConnectivity::Full;
  if (slab) options.directions = pfc::halo::presets::Full2D();
  pfc::comm::HaloExchange<Space, double> halo({&u, &v}, decomp, rank, MPI_COMM_WORLD,
                                              options);
  REQUIRE_THROWS_AS(halo.start(), std::logic_error);
  // Four distinct stage-owned values catch stale halos and cross-field tags.
  for (int stage = 0; stage < 4; ++stage) {
    initialize(u, 0, stage);
    initialize(v, 1, stage);
    producer(u, 0, stage);
    producer(v, 1, stage);
    halo.exchange();
    verify(u, 0, stage, slab);
    verify(v, 1, stage, slab);
  }
}

template <class Space> void thin_rejection() {
  int rank = 0, ranks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  auto domain = pfc::domain::create({2 * ranks, 7, 5});
  auto decomp = pfc::decomposition::create(domain, {ranks, 1, 1});
  pfc::data::Field<double, Space> u(
      domain, pfc::decomposition::local_box(decomp, rank), width);
  pfc::comm::HaloExchangeOptions options;
  options.connectivity = pfc::comm::HaloConnectivity::Full;
  REQUIRE_THROWS_AS((pfc::comm::HaloExchange<Space, double>(
                        u, decomp, rank, MPI_COMM_WORLD, options)),
                    std::invalid_argument);
}
} // namespace full_halo_fd6
