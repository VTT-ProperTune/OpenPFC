// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_padded_halo_exchange.cpp
 * @brief Integration tests for `pfc::PaddedHaloExchanger<T>`.
 *
 * Each test fills the owned core of a `pfc::field::PaddedBrick<double>`
 * with a known per-rank value, runs the in-place padded halo exchange,
 * and asserts the appropriate halo ring received the neighbour's value
 * (or the rank's own value, for self-wrap on periodic boundaries).
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;

namespace {

void fill_owned(field::PaddedBrick<double> &u, double val) {
  for (int k = 0; k < u.nz(); ++k)
    for (int j = 0; j < u.ny(); ++j)
      for (int i = 0; i < u.nx(); ++i) u(i, j, k) = val;
}

bool halo_layer_x_matches(const field::PaddedBrick<double> &u, int i,
                          double expected) {
  bool matches = true;
  for (int k = 0; k < u.nz(); ++k)
    for (int j = 0; j < u.ny(); ++j) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_layer_y_matches(const field::PaddedBrick<double> &u, int j,
                          double expected) {
  bool matches = true;
  for (int k = 0; k < u.nz(); ++k)
    for (int i = 0; i < u.nx(); ++i) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_layer_z_matches(const field::PaddedBrick<double> &u, int k,
                          double expected) {
  bool matches = true;
  for (int j = 0; j < u.ny(); ++j)
    for (int i = 0; i < u.nx(); ++i) matches &= u(i, j, k) == expected;
  return matches;
}

} // namespace

TEST_CASE("PaddedHaloExchanger: single-rank periodic wrap fills all 6 halos",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  auto world = world::create(GridSize({8, 6, 4}));
  auto decomp = decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  fill_owned(u, 7.0);

  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());

  const bool halos_match =
      halo_layer_x_matches(u, -1, 7.0) && halo_layer_x_matches(u, u.nx(), 7.0) &&
      halo_layer_y_matches(u, -1, 7.0) && halo_layer_y_matches(u, u.ny(), 7.0) &&
      halo_layer_z_matches(u, -1, 7.0) && halo_layer_z_matches(u, u.nz(), 7.0);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: two-rank X-split fills +X / -X with neighbour",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 2;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());

  bool halos_match = true;
  for (int d = 1; d <= hw; ++d)
    halos_match &= halo_layer_x_matches(u, -d, other) &&
                   halo_layer_x_matches(u, u.nx() + d - 1, other) &&
                   halo_layer_y_matches(u, -d, mine) &&
                   halo_layer_y_matches(u, u.ny() + d - 1, mine) &&
                   halo_layer_z_matches(u, -d, mine) &&
                   halo_layer_z_matches(u, u.nz() + d - 1, mine);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: non-blocking start/finish overlaps with inner work",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);
  halo.start_halo_exchange(u.data(), u.size());

  double inner_sum = 0.0;
  for (int k = hw; k < u.nz() - hw; ++k)
    for (int j = hw; j < u.ny() - hw; ++j)
      for (int i = hw; i < u.nx() - hw; ++i) inner_sum += u(i, j, k);
  const bool inner_sum_positive = inner_sum > 0.0;
  const bool mine_positive = mine > 0.0;
  REQUIRE(inner_sum_positive == mine_positive);

  halo.finish_halo_exchange();

  const bool halos_match =
      halo_layer_x_matches(u, -1, other) && halo_layer_x_matches(u, u.nx(), other) &&
      halo_layer_y_matches(u, -1, mine) && halo_layer_z_matches(u, -1, mine);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: 2x2x1 grid fills X and Y with right neighbours",
          "[MPI][padded_halo][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) return;

  auto world = world::create(GridSize({16, 16, 4}));
  auto decomp = decomposition::create(world, {2, 2, 1});

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  fill_owned(u, mine);

  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD);
  halo.exchange_halos(u.data(), u.size());

  const int rank_x = rank % 2;
  const int rank_y = rank / 2;
  const int xpos_neighbor = ((rank_x + 1) % 2) + rank_y * 2;
  const int xneg_neighbor = ((rank_x - 1 + 2) % 2) + rank_y * 2;
  const int ypos_neighbor = rank_x + ((rank_y + 1) % 2) * 2;
  const int yneg_neighbor = rank_x + ((rank_y - 1 + 2) % 2) * 2;

  const bool halos_match =
      halo_layer_x_matches(u, u.nx(), static_cast<double>(xpos_neighbor)) &&
      halo_layer_x_matches(u, -1, static_cast<double>(xneg_neighbor)) &&
      halo_layer_y_matches(u, u.ny(), static_cast<double>(ypos_neighbor)) &&
      halo_layer_y_matches(u, -1, static_cast<double>(yneg_neighbor)) &&
      halo_layer_z_matches(u, -1, mine) && halo_layer_z_matches(u, u.nz(), mine);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: Axes2D direction set skips ±Z halos",
          "[MPI][padded_halo][halo_directions]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  auto world = pfc::world::create(GridSize({8, 6, 4}));
  auto decomp = pfc::decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  // Pre-fill every cell (owned + halo ring) with sentinel; then overwrite
  // owned region with 7.0. After Axes2D exchange ±Z halos must remain
  // sentinel; ±X and ±Y halos should self-wrap to 7.0.
  const double sentinel = -999.0;
  for (int k = -hw; k < u.nz() + hw; ++k)
    for (int j = -hw; j < u.ny() + hw; ++j)
      for (int i = -hw; i < u.nx() + hw; ++i) u(i, j, k) = sentinel;
  fill_owned(u, 7.0);

  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD,
                                   pfc::halo::presets::Axes2D());
  REQUIRE(halo.num_directions() == 4);
  halo.exchange_halos(u.data(), u.size());

  const bool halos_match =
      halo_layer_x_matches(u, -1, 7.0) && halo_layer_x_matches(u, u.nx(), 7.0) &&
      halo_layer_y_matches(u, -1, 7.0) && halo_layer_y_matches(u, u.ny(), 7.0) &&
      // ±Z stay at sentinel — Axes2D excludes them.
      halo_layer_z_matches(u, -1, sentinel) &&
      halo_layer_z_matches(u, u.nz(), sentinel);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: brick-binding ctor + free start/finish wrappers",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 2;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  // No decomp/rank/hw repeated here: the brick already carries them, the
  // exchanger picks them up plus binds u.data() once.
  PaddedHaloExchanger<double> halo(u, MPI_COMM_WORLD);
  REQUIRE(halo.is_bound());

  // Drive via the free wrappers — equivalent to halo.start() / halo.finish().
  start_exchange(halo);
  finish_exchange(halo);

  bool halos_match = true;
  for (int d = 1; d <= hw; ++d)
    halos_match &= halo_layer_x_matches(u, -d, other) &&
                   halo_layer_x_matches(u, u.nx() + d - 1, other);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: exchange(halo) matches start+finish",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = world::create(GridSize({16, 8, 4}));
  auto decomp = decomposition::create(world, {2, 1, 1});

  const int hw = 2;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  pfc::communication::PaddedHaloExchanger<double> halo(u, MPI_COMM_WORLD);
  pfc::communication::exchange(halo);

  bool halos_match = true;
  for (int d = 1; d <= hw; ++d)
    halos_match &= halo_layer_x_matches(u, -d, other) &&
                   halo_layer_x_matches(u, u.nx() + d - 1, other);
  REQUIRE(halos_match);
}

TEST_CASE("PaddedHaloExchanger: unbound start() throws std::logic_error",
          "[MPI][padded_halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  auto world = world::create(GridSize({4, 4, 4}));
  auto decomp = decomposition::create(world, 1);

  PaddedHaloExchanger<double> halo(decomp, rank, /*hw=*/1, MPI_COMM_WORLD);
  REQUIRE_FALSE(halo.is_bound());
  REQUIRE_THROWS_AS(halo.start(), std::logic_error);
  REQUIRE_THROWS_AS(halo.finish(), std::logic_error);
}
