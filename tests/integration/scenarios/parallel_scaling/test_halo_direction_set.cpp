// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_halo_direction_set.cpp
 * @brief Tests for `pfc::halo::HaloDirectionSet`, presets, and
 *        neighbour-boundary agreement at HaloExchange construction.
 *
 * Covers:
 *   1. Preset content and size invariants for `Axes2D / Full2D / Axes3D /
 *      Full3D`.
 *   2. Custom set construction validates inputs (`{0,0,0}` rejected,
 *      out-of-range components rejected, duplicates dropped).
 *   3. `direction_to_face_slot` round-trips with `face_slot_to_direction`.
 *   4. `from_connectivity` mirrors the legacy enum.
 *   5. `HaloExchange` with `Axes2D()` on an `nz=1` periodic field is
 *      bit-equal to a manual XY periodic fill (no Z halo touched).
 *   6. Neighbour agreement: mismatched direction sets that disagree on a
 *      shared face throw `std::runtime_error`; uniform Axes2D still
 *      constructs Faces and persistent `HaloExchange`.
 *   Per-rank `HaloDirectionSelector` remains old-API-only until the
 *   facade grows a selector knob.
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_direction_agreement.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

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
  bool round_trip_matches = true;
  for (int slot = 0; slot < 6; ++slot) {
    const auto dir = halo::face_slot_to_direction(slot);
    round_trip_matches &= halo::direction_to_face_slot(dir) == slot;
  }
  REQUIRE(round_trip_matches);
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

void fill_owned(data::Field<double, HostSpace> &u, double val) {
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) u(i, j, k) = val;
}

void clear_halo(data::Field<double, HostSpace> &u, double val) {
  // Set every cell (including the halo ring) to `val`; tests then overwrite
  // owned cells separately.
  const auto n = u.size3();
  const int hw = u.storage_halo();
  for (int k = -hw; k < n[2] + hw; ++k)
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i) u(i, j, k) = val;
}

} // namespace

TEST_CASE("HaloExchange Axes2D leaves ±Z halos untouched on nz=1 slab",
          "[MPI][halo_directions][padded][axes2d]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) return;

  // Single-rank, periodic, 2D slab (nz=1). With `Axes3D()` periodic ±Z would
  // fill the Z halos (here from the same rank's owned cells); with
  // `Axes2D()` they must remain at the sentinel value.
  auto global_domain = pfc::domain::create({8, 8, 1});
  auto decomp = pfc::decomposition::create(global_domain, 1);

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double sentinel = -999.0;
  clear_halo(u, sentinel);
  fill_owned(u, 7.0);

  comm::HaloExchangeOptions opt;
  opt.directions = halo::presets::Axes2D();
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  halo.exchange();

  bool halos_match = true;
  const auto n = u.size3();
  // X / Y halos populated by self-wrap.
  for (int k = 0; k < n[2]; ++k) {
    for (int j = 0; j < n[1]; ++j) {
      halos_match &= u(-1, j, k) == 7.0 && u(n[0], j, k) == 7.0;
    }
    for (int i = 0; i < n[0]; ++i) {
      halos_match &= u(i, -1, k) == 7.0 && u(i, n[1], k) == 7.0;
    }
  }
  // Z halos must be untouched (set to sentinel by `clear_halo`).
  for (int j = 0; j < n[1]; ++j) {
    for (int i = 0; i < n[0]; ++i) {
      halos_match &= u(i, j, -1) == sentinel && u(i, j, n[2]) == sentinel;
    }
  }
  REQUIRE(halos_match);
}

TEST_CASE("HaloExchange Axes2D matches Axes3D in XY (two-rank X-split)",
          "[MPI][halo_directions][padded][axes2d]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 1});
  auto decomp = pfc::decomposition::create(global_domain, {2, 1, 1});

  const int hw = 1;
  auto u_axes2d = data::field_from_subdomain<double>(decomp, rank, hw);
  auto u_axes3d = data::field_from_subdomain<double>(decomp, rank, hw);

  const double mine = static_cast<double>(rank);
  const double sentinel = -999.0;
  clear_halo(u_axes2d, sentinel);
  clear_halo(u_axes3d, sentinel);
  fill_owned(u_axes2d, mine);
  fill_owned(u_axes3d, mine);

  comm::HaloExchangeOptions opt2d;
  opt2d.directions = halo::presets::Axes2D();
  comm::HaloExchangeOptions opt3d;
  opt3d.directions = halo::presets::Axes3D();
  opt3d.exchange_base = 1;
  comm::HaloExchange<HostSpace, double> halo2d(u_axes2d, decomp, rank,
                                               MPI_COMM_WORLD, opt2d);
  comm::HaloExchange<HostSpace, double> halo3d(u_axes3d, decomp, rank,
                                               MPI_COMM_WORLD, opt3d);

  halo2d.exchange();
  halo3d.exchange();

  bool halos_match = true;
  const auto n2d = u_axes2d.size3();
  const auto n3d = u_axes3d.size3();
  // X and Y halos should be identical between the two configurations.
  for (int k = 0; k < n2d[2]; ++k) {
    for (int j = 0; j < n2d[1]; ++j) {
      halos_match &= u_axes2d(-1, j, k) == u_axes3d(-1, j, k) &&
                     u_axes2d(n2d[0], j, k) == u_axes3d(n3d[0], j, k);
    }
    for (int i = 0; i < n2d[0]; ++i) {
      halos_match &= u_axes2d(i, -1, k) == u_axes3d(i, -1, k) &&
                     u_axes2d(i, n2d[1], k) == u_axes3d(i, n3d[1], k);
    }
  }
  // Z halo: Axes3D self-wraps mine; Axes2D leaves sentinel.
  for (int j = 0; j < n2d[1]; ++j) {
    for (int i = 0; i < n2d[0]; ++i) {
      halos_match &= u_axes2d(i, j, -1) == sentinel &&
                     u_axes2d(i, j, n2d[2]) == sentinel &&
                     u_axes3d(i, j, -1) == mine && u_axes3d(i, j, n3d[2]) == mine;
    }
  }
  REQUIRE(halos_match);
}

TEST_CASE("Mismatched HaloDirectionSet throws at neighbour agreement",
          "[MPI][halo_directions][agreement]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 1});
  auto decomp = pfc::decomposition::create(global_domain, {2, 1, 1});

  // Shared X face disagrees: rank 0 posts ±X (Axes2D); rank 1 has ±Y only.
  const auto dirs =
      (rank == 0)
          ? halo::presets::Axes2D()
          : HaloDirectionSet(std::vector<Int3>{Int3{0, 1, 0}, Int3{0, -1, 0}});
  REQUIRE_THROWS_AS((halo::validate_neighbour_direction_agreement(
                        MPI_COMM_WORLD, decomp, rank, dirs)),
                    std::runtime_error);
}

TEST_CASE("Agreeing Axes2D constructs Faces and persistent HaloExchange",
          "[MPI][halo_directions][agreement]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 1});
  auto decomp = pfc::decomposition::create(global_domain, {2, 1, 1});
  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);

  comm::HaloExchangeOptions faces;
  faces.directions = halo::presets::Axes2D();
  REQUIRE_NOTHROW((comm::HaloExchange<HostSpace, double>(u, decomp, rank,
                                                         MPI_COMM_WORLD, faces)));

  comm::HaloExchangeOptions persist;
  persist.directions = halo::presets::Axes2D();
  persist.persistent = true;
  REQUIRE_NOTHROW((comm::HaloExchange<HostSpace, double>(u, decomp, rank,
                                                         MPI_COMM_WORLD, persist)));
}
