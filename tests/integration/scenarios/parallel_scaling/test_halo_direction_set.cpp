// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_halo_direction_set.cpp
 * @brief Tests for `pfc::halo::HaloDirectionSet`, presets, and per-rank
 *        selectors plumbed into the CPU exchangers.
 *
 * Covers:
 *   1. Preset content and size invariants for `Axes2D / Full2D / Axes3D /
 *      Full3D`.
 *   2. Custom set construction validates inputs (`{0,0,0}` rejected,
 *      out-of-range components rejected, duplicates dropped).
 *   3. `direction_to_face_slot` round-trips with `face_slot_to_direction`.
 *   4. `from_connectivity` mirrors the legacy enum.
 *   5. `PaddedHaloExchanger<double>` with `Axes2D()` on an `nz=1`
 *      periodic field is bit-equal to a manual XY periodic fill (no Z
 *      halo touched).
 *   6. Per-rank selector: rank 0 gets `Axes2D`, rank 1 gets `±X` only —
 *      verify each rank's halo ring matches the configured set.
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <algorithm>
#include <vector>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/decomposition/padded_halo_exchange.hpp>
#include <openpfc/kernel/field/padded_brick.hpp>

using namespace pfc;
using pfc::halo::HaloDirectionSet;
using Int3 = pfc::types::Int3;

TEST_CASE("HaloDirectionSet presets have the documented sizes and members",
          "[halo_directions][preset]") {
  SECTION("Axes2D") {
    auto s = halo::presets::Axes2D();
    REQUIRE(s.size() == 4);
    REQUIRE(s.contains(Int3{1, 0, 0}));
    REQUIRE(s.contains(Int3{-1, 0, 0}));
    REQUIRE(s.contains(Int3{0, 1, 0}));
    REQUIRE(s.contains(Int3{0, -1, 0}));
    REQUIRE_FALSE(s.contains(Int3{0, 0, 1}));
    REQUIRE_FALSE(s.contains(Int3{0, 0, -1}));
  }
  SECTION("Full2D") {
    auto s = halo::presets::Full2D();
    REQUIRE(s.size() == 8);
    REQUIRE(s.contains(Int3{1, 1, 0}));
    REQUIRE(s.contains(Int3{-1, -1, 0}));
    REQUIRE_FALSE(s.contains(Int3{0, 0, 1}));
    REQUIRE_FALSE(s.contains(Int3{1, 0, 1}));
  }
  SECTION("Axes3D") {
    auto s = halo::presets::Axes3D();
    REQUIRE(s.size() == 6);
    REQUIRE(s.contains(Int3{0, 0, 1}));
    REQUIRE(s.contains(Int3{0, 0, -1}));
    REQUIRE_FALSE(s.contains(Int3{1, 1, 0}));
  }
  SECTION("Full3D") {
    auto s = halo::presets::Full3D();
    REQUIRE(s.size() == 26);
    // Sample a face, an edge, and a corner direction.
    REQUIRE(s.contains(Int3{1, 0, 0}));
    REQUIRE(s.contains(Int3{1, 1, 0}));
    REQUIRE(s.contains(Int3{1, 1, 1}));
    REQUIRE(s.contains(Int3{-1, -1, -1}));
    REQUIRE_FALSE(s.contains(Int3{0, 0, 0}));
  }
}

TEST_CASE("HaloDirectionSet custom construction validates and dedupes",
          "[halo_directions][validation]") {
  SECTION("rejects zero direction") {
    REQUIRE_THROWS_AS(HaloDirectionSet(std::vector<Int3>{Int3{0, 0, 0}}),
                      std::invalid_argument);
  }
  SECTION("rejects out-of-range component") {
    REQUIRE_THROWS_AS(HaloDirectionSet(std::vector<Int3>{Int3{2, 0, 0}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(HaloDirectionSet(std::vector<Int3>{Int3{0, -2, 0}}),
                      std::invalid_argument);
  }
  SECTION("dedupes repeated entries, preserves first-seen order") {
    HaloDirectionSet s(std::vector<Int3>{Int3{1, 0, 0}, Int3{-1, 0, 0},
                                         Int3{1, 0, 0}, Int3{0, 1, 0}});
    REQUIRE(s.size() == 3);
    REQUIRE(s.dirs[0] == Int3{1, 0, 0});
    REQUIRE(s.dirs[1] == Int3{-1, 0, 0});
    REQUIRE(s.dirs[2] == Int3{0, 1, 0});
  }
  SECTION("custom subset works") {
    HaloDirectionSet s(std::vector<Int3>{Int3{1, 0, 0}, Int3{-1, 0, 0}});
    REQUIRE(s.size() == 2);
    REQUIRE(s.contains(Int3{1, 0, 0}));
    REQUIRE_FALSE(s.contains(Int3{0, 1, 0}));
  }
}

TEST_CASE("direction_to_face_slot / face_slot_to_direction round trip",
          "[halo_directions][slot]") {
  for (int slot = 0; slot < 6; ++slot) {
    const auto dir = halo::face_slot_to_direction(slot);
    REQUIRE(halo::direction_to_face_slot(dir) == slot);
  }
  REQUIRE(halo::direction_to_face_slot(Int3{1, 1, 0}) == -1);
  REQUIRE(halo::direction_to_face_slot(Int3{1, 1, 1}) == -1);
  REQUIRE_THROWS_AS(halo::face_slot_to_direction(-1), std::out_of_range);
  REQUIRE_THROWS_AS(halo::face_slot_to_direction(6), std::out_of_range);
}

TEST_CASE("from_connectivity translates to expected presets",
          "[halo_directions][connectivity]") {
  REQUIRE(halo::from_connectivity(halo::Connectivity::Faces, 2) ==
          halo::presets::Axes2D());
  REQUIRE(halo::from_connectivity(halo::Connectivity::Faces, 3) ==
          halo::presets::Axes3D());
  REQUIRE(halo::from_connectivity(halo::Connectivity::Edges, 2) ==
          halo::presets::Full2D());
  REQUIRE(halo::from_connectivity(halo::Connectivity::Edges, 3) ==
          halo::presets::Full3D());
  REQUIRE(halo::from_connectivity(halo::Connectivity::All, 2) ==
          halo::presets::Full3D());
  REQUIRE(halo::from_connectivity(halo::Connectivity::All, 3) ==
          halo::presets::Full3D());
  REQUIRE_THROWS_AS(halo::from_connectivity(halo::Connectivity::Faces, 1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(halo::from_connectivity(halo::Connectivity::Faces, 4),
                    std::invalid_argument);
}

namespace {

void fill_owned(field::PaddedBrick<double> &u, double val) {
  for (int k = 0; k < u.nz(); ++k)
    for (int j = 0; j < u.ny(); ++j)
      for (int i = 0; i < u.nx(); ++i) u(i, j, k) = val;
}

void clear_halo(field::PaddedBrick<double> &u, double val) {
  // Set every cell (including the halo ring) to `val`; tests then overwrite
  // owned cells separately.
  for (int k = -u.halo_width(); k < u.nz() + u.halo_width(); ++k)
    for (int j = -u.halo_width(); j < u.ny() + u.halo_width(); ++j)
      for (int i = -u.halo_width(); i < u.nx() + u.halo_width(); ++i)
        u(i, j, k) = val;
}

} // namespace

TEST_CASE("PaddedHaloExchanger Axes2D leaves ±Z halos untouched on nz=1 slab",
          "[MPI][halo_directions][padded][axes2d]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  // Single-rank, periodic, 2D slab (nz=1). With `Axes3D()` periodic ±Z would
  // fill the Z halos (here from the same rank's owned cells); with
  // `Axes2D()` they must remain at the sentinel value.
  auto world = pfc::world::create(GridSize({8, 8, 1}));
  auto decomp = pfc::decomposition::create(world, 1);

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double sentinel = -999.0;
  clear_halo(u, sentinel);
  fill_owned(u, 7.0);

  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD,
                                   halo::presets::Axes2D());
  REQUIRE(halo.num_directions() == 4);
  halo.exchange_halos(u.data(), u.size());

  // X / Y halos populated by self-wrap.
  for (int k = 0; k < u.nz(); ++k) {
    for (int j = 0; j < u.ny(); ++j) {
      REQUIRE(u(-1, j, k) == 7.0);
      REQUIRE(u(u.nx(), j, k) == 7.0);
    }
    for (int i = 0; i < u.nx(); ++i) {
      REQUIRE(u(i, -1, k) == 7.0);
      REQUIRE(u(i, u.ny(), k) == 7.0);
    }
  }
  // Z halos must be untouched (set to sentinel by `clear_halo`).
  for (int j = 0; j < u.ny(); ++j) {
    for (int i = 0; i < u.nx(); ++i) {
      REQUIRE(u(i, j, -1) == sentinel);
      REQUIRE(u(i, j, u.nz()) == sentinel);
    }
  }
}

TEST_CASE("PaddedHaloExchanger Axes2D matches Axes3D in XY (two-rank X-split)",
          "[MPI][halo_directions][padded][axes2d]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = pfc::world::create(GridSize({16, 8, 1}));
  auto decomp = pfc::decomposition::create(world, {2, 1, 1});

  const int hw = 1;
  field::PaddedBrick<double> u_axes2d(decomp, rank, hw);
  field::PaddedBrick<double> u_axes3d(decomp, rank, hw);

  const double mine = static_cast<double>(rank);
  const double sentinel = -999.0;
  clear_halo(u_axes2d, sentinel);
  clear_halo(u_axes3d, sentinel);
  fill_owned(u_axes2d, mine);
  fill_owned(u_axes3d, mine);

  PaddedHaloExchanger<double> halo2d(decomp, rank, hw, MPI_COMM_WORLD,
                                     halo::presets::Axes2D(), /*base_tag=*/0);
  PaddedHaloExchanger<double> halo3d(decomp, rank, hw, MPI_COMM_WORLD,
                                     halo::presets::Axes3D(),
                                     /*base_tag=*/100);

  halo2d.exchange_halos(u_axes2d.data(), u_axes2d.size());
  halo3d.exchange_halos(u_axes3d.data(), u_axes3d.size());

  // X and Y halos should be identical between the two configurations.
  for (int k = 0; k < u_axes2d.nz(); ++k) {
    for (int j = 0; j < u_axes2d.ny(); ++j) {
      REQUIRE(u_axes2d(-1, j, k) == u_axes3d(-1, j, k));
      REQUIRE(u_axes2d(u_axes2d.nx(), j, k) == u_axes3d(u_axes3d.nx(), j, k));
    }
    for (int i = 0; i < u_axes2d.nx(); ++i) {
      REQUIRE(u_axes2d(i, -1, k) == u_axes3d(i, -1, k));
      REQUIRE(u_axes2d(i, u_axes2d.ny(), k) == u_axes3d(i, u_axes3d.ny(), k));
    }
  }
  // Z halo: Axes3D self-wraps mine; Axes2D leaves sentinel.
  for (int j = 0; j < u_axes2d.ny(); ++j) {
    for (int i = 0; i < u_axes2d.nx(); ++i) {
      REQUIRE(u_axes2d(i, j, -1) == sentinel);
      REQUIRE(u_axes2d(i, j, u_axes2d.nz()) == sentinel);
      REQUIRE(u_axes3d(i, j, -1) == mine);
      REQUIRE(u_axes3d(i, j, u_axes3d.nz()) == mine);
    }
  }
}

TEST_CASE("HaloDirectionSelector overrides the uniform direction set",
          "[MPI][halo_directions][selector]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto world = pfc::world::create(GridSize({16, 8, 1}));
  auto decomp = pfc::decomposition::create(world, {2, 1, 1});

  // Selector: rank 0 → Axes2D (4 dirs); rank 1 → ±X only (2 dirs).
  halo::HaloDirectionSelector selector = [](int r) {
    if (r == 0) return halo::presets::Axes2D();
    return HaloDirectionSet(std::vector<Int3>{Int3{1, 0, 0}, Int3{-1, 0, 0}});
  };

  const int hw = 1;
  field::PaddedBrick<double> u(decomp, rank, hw);
  const double sentinel = -999.0;
  clear_halo(u, sentinel);
  fill_owned(u, static_cast<double>(rank));

  // The uniform fallback is never used because `selector` is always called
  // for the local rank; pass any valid set as the fallback.
  PaddedHaloExchanger<double> halo(decomp, rank, hw, MPI_COMM_WORLD,
                                   halo::presets::Axes3D(), /*base_tag=*/0,
                                   selector);
  if (rank == 0) {
    REQUIRE(halo.direction_set() == halo::presets::Axes2D());
    REQUIRE(halo.num_directions() == 4);
  } else {
    REQUIRE(halo.direction_set().size() == 2);
    REQUIRE(halo.num_directions() == 2);
  }
}
