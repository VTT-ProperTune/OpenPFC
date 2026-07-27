// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_padded_halo_exchange.cpp
 * @brief Integration tests for `pfc::PaddedHaloExchanger<T>`.
 *
 * Each test fills the owned core of a `pfc::data::Field<double, HostSpace>`
 * with a known per-rank value, runs the in-place padded halo exchange,
 * and asserts the appropriate halo ring received the neighbour's value
 * (or the rank's own value, for self-wrap on periodic boundaries).
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

namespace {

void fill_owned(data::Field<double, HostSpace> &u, double val) {
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) u(i, j, k) = val;
}

bool halo_layer_x_matches(const data::Field<double, HostSpace> &u, int i,
                          double expected) {
  bool matches = true;
  const auto n = u.size3();
  const int hw = u.storage_halo();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_layer_y_matches(const data::Field<double, HostSpace> &u, int j,
                          double expected) {
  bool matches = true;
  const auto n = u.size3();
  const int hw = u.storage_halo();
  for (int k = 0; k < n[2]; ++k)
    for (int i = 0; i < n[0]; ++i) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_layer_z_matches(const data::Field<double, HostSpace> &u, int k,
                          double expected) {
  bool matches = true;
  const auto n = u.size3();
  const int hw = u.storage_halo();
  for (int j = 0; j < n[1]; ++j)
    for (int i = 0; i < n[0]; ++i) matches &= u(i, j, k) == expected;
  return matches;
}

} // namespace

TEST_CASE("PaddedHaloExchanger: single-rank periodic wrap fills all 6 halos",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  auto world = world::create(GridSize({8, 6, 4}).to_vector3());
  auto decomp = decomposition::create(world, 1);

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  fill_owned(u, 7.0);

  auto domain = decomposition::domain(decomp);
  auto subdomain_box = decomposition::local_box(decomp, rank);
  PaddedHaloExchanger<double> halo(subdomain_box, domain, decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());

  const bool halos_match =
      halo_layer_x_matches(u, -1, 7.0) && halo_layer_x_matches(u, u.local_size()[0], 7.0) &&
      halo_layer_y_matches(u, -1, 7.0) && halo_layer_y_matches(u, u.local_size()[1], 7.0) &&
      halo_layer_z_matches(u, -1, 7.0) && halo_layer_z_matches(u, u.local_size()[2], 7.0);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: two-rank X-split fills +X / -X with neighbour",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}).to_vector3());
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 2;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  auto domain = decomposition::domain(decomp);
  auto subdomain_box = decomposition::local_box(decomp, rank);
  PaddedHaloExchanger<double> halo(subdomain_box, domain, decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());

  bool halos_match = true;
  const auto n = u.local_size();
  for (int d = 1; d <= hw; ++d)
    halos_match &= halo_layer_x_matches(u, -d, other) &&
                   halo_layer_x_matches(u, n[0] + d - 1, other) &&
                   halo_layer_y_matches(u, -d, mine) &&
                   halo_layer_y_matches(u, n[1] + d - 1, mine) &&
                   halo_layer_z_matches(u, -d, mine) &&
                   halo_layer_z_matches(u, n[2] + d - 1, mine);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: non-blocking start/finish overlaps with inner work",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}).to_vector3());
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  auto domain = decomposition::domain(decomp);
  auto subdomain_box = decomposition::local_box(decomp, rank);
  PaddedHaloExchanger<double> halo(subdomain_box, domain, decomp, rank, hw, MPI_COMM_WORLD);
  halo.start_halo_exchange(u.data(), u.size());

  double inner_sum = 0.0;
  const auto n = u.local_size();
  for (int k = hw; k < n[2] - hw; ++k)
    for (int j = hw; j < n[1] - hw; ++j)
      for (int i = hw; i < n[0] - hw; ++i) inner_sum += u(i, j, k);
  const bool inner_sum_positive = inner_sum > 0.0;
  const bool mine_positive = mine > 0.0;
  REQUIRE(inner_sum_positive == mine_positive);

  halo.finish_halo_exchange();

  const bool halos_match =
      halo_layer_x_matches(u, -1, other) && halo_layer_x_matches(u, n[0], other) &&
      halo_layer_y_matches(u, -1, mine) && halo_layer_z_matches(u, -1, mine);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: 2x2x1 grid fills X and Y with right neighbours",
          "[MPI][padded_halo][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) return;

  auto world = world::create(GridSize({16, 16, 4}).to_vector3());
  auto decomp = decomposition::create(world, {2, 2, 1});

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  fill_owned(u, mine);

  auto domain = decomposition::domain(decomp);
  auto subdomain_box = decomposition::local_box(decomp, rank);
  PaddedHaloExchanger<double> halo(subdomain_box, domain, decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());

  const int rank_x = rank % 2;
  const int rank_y = rank / 2;
  const int xpos_neighbor = ((rank_x + 1) % 2) + rank_y * 2;
  const int xneg_neighbor = ((rank_x - 1 + 2) % 2) + rank_y * 2;
  const int ypos_neighbor = rank_x + ((rank_y + 1) % 2) * 2;
  const int yneg_neighbor = rank_x + ((rank_y - 1 + 2) % 2) * 2;

  const auto n = u.local_size();
  const bool halos_match =
      halo_layer_x_matches(u, n[0], static_cast<double>(xpos_neighbor)) &&
      halo_layer_x_matches(u, -1, static_cast<double>(xneg_neighbor)) &&
      halo_layer_y_matches(u, n[1], static_cast<double>(ypos_neighbor)) &&
      halo_layer_y_matches(u, -1, static_cast<double>(yneg_neighbor)) &&
      halo_layer_z_matches(u, -1, mine) && halo_layer_z_matches(u, n[2], mine);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: Axes2D direction set skips ±Z halos",
          "[MPI][padded_halo][halo_directions]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  auto world = pfc::world::create(GridSize({8, 6, 4}).to_vector3());
  auto decomp = pfc::decomposition::create(world, 1);

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  // Pre-fill every cell (owned + halo ring) with sentinel; then overwrite
  // owned region with 7.0. After Axes2D exchange ±Z halos must remain
  // sentinel; ±X and ±Y halos should self-wrap to 7.0.
  const double sentinel = -999.0;
  const auto n = u.local_size();
  for (int k = -hw; k < n[2] + hw; ++k)
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i) u(i, j, k) = sentinel;
  fill_owned(u, 7.0);

  auto domain = decomposition::domain(decomp);
  auto subdomain_box = decomposition::local_box(decomp, rank);
  PaddedHaloExchanger<double> halo(subdomain_box, domain, decomp, rank, hw, MPI_COMM_WORLD,
                                   pfc::halo::presets::Axes2D());
  REQUIRE(halo.num_directions() == 4);
  halo.exchange_halos(u.data(), u.size());

  const bool halos_match =
      halo_layer_x_matches(u, -1, 7.0) && halo_layer_x_matches(u, n[0], 7.0) &&
      halo_layer_y_matches(u, -1, 7.0) && halo_layer_y_matches(u, n[1], 7.0) &&
      // ±Z stay at sentinel — Axes2D excludes them.
      halo_layer_z_matches(u, -1, sentinel) &&
      halo_layer_z_matches(u, n[2], sentinel);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: brick-binding ctor + free start/finish wrappers",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}).to_vector3());
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 2;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  // The Field-binding ctor requires explicit decompress, rank, hw parameters.
  PaddedHaloExchanger<double> halo(u, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(halo.is_bound());

  // Drive via the free wrappers — equivalent to halo.start() / halo.finish().
  start_exchange(halo);
  finish_exchange(halo);

  bool halos_match = true;
  const auto n = u.local_size();
  for (int d = 1; d <= hw; ++d)
    halos_match &= halo_layer_x_matches(u, -d, other) &&
                   halo_layer_x_matches(u, n[0] + d - 1, other);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: exchange(halo) matches start+finish",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}).to_vector3());
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 2;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  pfc::communication::PaddedHaloExchanger<double> halo(u, decomp, rank, MPI_COMM_WORLD);
  pfc::communication::exchange(halo);

  bool halos_match = true;
  const auto n = u.local_size();
  for (int d = 1; d <= hw; ++d)
    halos_match &= halo_layer_x_matches(u, -d, other) &&
                   halo_layer_x_matches(u, n[0] + d - 1, other);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: unbound start() throws std::logic_error",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  auto world = world::create(GridSize({4, 4, 4}).to_vector3());
  auto decomp = decomposition::create(world, 1);

  auto domain = decomposition::domain(decomp);
  auto subdomain_box = decomposition::local_box(decomp, rank);
  PaddedHaloExchanger<double> halo(subdomain_box, domain, decomp, rank, /*hw=*/1, MPI_COMM_WORLD);
  REQUIRE_FALSE(halo.is_bound());
  REQUIRE_THROWS_AS(halo.start(), std::logic_error);
  REQUIRE_THROWS_AS(halo.finish(), std::logic_error);
}
