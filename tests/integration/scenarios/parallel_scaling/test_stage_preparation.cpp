// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_stage_preparation.cpp
 * @brief Multi-rank Catch2 coverage for `StagePreparationService`.
 *
 * Proves scalar and two-field padded prepare, halo no-op, reject/retry
 * re-prepare (no rollback API), and boundary-hook ordering relative to
 * exchange.
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <string>
#include <string_view>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/stage_preparation.hpp>
#include <openpfc/kernel/field/field_factory.hpp>
#include <openpfc/kernel/integrator/stage_context.hpp>

using namespace pfc;

namespace {

void fill_owned(data::Field<double, HostSpace> &u, double val) {
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) u(i, j, k) = val;
}

void poison_halos(data::Field<double, HostSpace> &u, double poison) {
  const int hw = u.storage_halo();
  const auto n = u.size3();
  for (int d = 1; d <= hw; ++d) {
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j) {
        u(-d, j, k) = poison;
        u(n[0] + d - 1, j, k) = poison;
      }
    for (int k = 0; k < n[2]; ++k)
      for (int i = 0; i < n[0]; ++i) {
        u(i, -d, k) = poison;
        u(i, n[1] + d - 1, k) = poison;
      }
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) {
        u(i, j, -d) = poison;
        u(i, j, n[2] + d - 1) = poison;
      }
  }
}

// The exchanger is face-only (see HaloExchange Faces docs): each ghost ring
// only covers the *owned* extent of the two orthogonal axes, not the padded
// extent, so corner cells (where two ghost rings would overlap) are never
// written. These checks must match that contract and stick to the owned
// range on the orthogonal axes.
bool halo_layer_x_matches(const data::Field<double, HostSpace> &u, int i,
                          double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_layer_y_matches(const data::Field<double, HostSpace> &u, int j,
                          double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int i = 0; i < n[0]; ++i) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_layer_z_matches(const data::Field<double, HostSpace> &u, int k,
                          double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int j = 0; j < n[1]; ++j)
    for (int i = 0; i < n[0]; ++i) matches &= u(i, j, k) == expected;
  return matches;
}

bool x_split_halos_match(const data::Field<double, HostSpace> &u, double mine,
                         double other) {
  const int hw = u.storage_halo();
  bool ok = true;
  const auto n = u.size3();
  for (int d = 1; d <= hw; ++d)
    ok &= halo_layer_x_matches(u, -d, other) &&
          halo_layer_x_matches(u, n[0] + d - 1, other) &&
          halo_layer_y_matches(u, -d, mine) &&
          halo_layer_y_matches(u, n[1] + d - 1, mine) &&
          halo_layer_z_matches(u, -d, mine) &&
          halo_layer_z_matches(u, n[2] + d - 1, mine);
  return ok;
}

bool x_halos_all_equal(const data::Field<double, HostSpace> &u, double expected) {
  const int hw = u.storage_halo();
  bool ok = true;
  const auto n = u.size3();
  for (int d = 1; d <= hw; ++d)
    ok &= halo_layer_x_matches(u, -d, expected) &&
          halo_layer_x_matches(u, n[0] + d - 1, expected);
  return ok;
}

} // namespace

TEST_CASE("StagePreparationService: scalar prepare fills ±X ghosts",
          "[MPI][stage_preparation]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 4});
  auto decomp = decomposition::create(global_domain, {2, 1, 1});

  const int hw = 2;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  communication::StagePreparationService<double> prep;
  prep.bind("u", halo);

  communication::StagePreparationRequirements req{
      .region_kind = communication::RegionKind::All,
      .needs_halo_exchange = true,
      .needs_boundary_update = false,
  };
  const std::string_view fields[] = {"u"};
  prep.prepare(req, fields);

  REQUIRE(x_split_halos_match(u, mine, other));
}

TEST_CASE("StagePreparationService: two-field prepare fills both ghost rings",
          "[MPI][stage_preparation]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 4});
  auto decomp = decomposition::create(global_domain, {2, 1, 1});

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  auto v = data::field_from_subdomain<double>(decomp, rank, hw);
  const double u_mine = 10.0 + static_cast<double>(rank);
  const double u_other = 10.0 + static_cast<double>(1 - rank);
  const double v_mine = 20.0 + static_cast<double>(rank);
  const double v_other = 20.0 + static_cast<double>(1 - rank);
  fill_owned(u, u_mine);
  fill_owned(v, v_mine);

  comm::HaloExchange<HostSpace, double> halo_u(u, decomp, rank, MPI_COMM_WORLD);
  comm::HaloExchange<HostSpace, double> halo_v(v, decomp, rank, MPI_COMM_WORLD);
  communication::StagePreparationService<double> prep;
  prep.bind("u", halo_u);
  prep.bind("v", halo_v);

  communication::StagePreparationRequirements req{
      .region_kind = communication::RegionKind::All,
      .needs_halo_exchange = true,
      .needs_boundary_update = false,
  };
  const std::string_view fields[] = {"u", "v"};
  prep.prepare(req, fields);

  REQUIRE(x_split_halos_match(u, u_mine, u_other));
  REQUIRE(x_split_halos_match(v, v_mine, v_other));
}

TEST_CASE("StagePreparationService: needs_halo=false leaves ghosts untouched",
          "[MPI][stage_preparation]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 4});
  auto decomp = decomposition::create(global_domain, {2, 1, 1});

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  fill_owned(u, static_cast<double>(rank));
  constexpr double poison = -999.0;
  poison_halos(u, poison);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  communication::StagePreparationService<double> prep;
  prep.bind("u", halo);

  communication::StagePreparationRequirements req{
      .region_kind = communication::RegionKind::Interior,
      .needs_halo_exchange = false,
      .needs_boundary_update = false,
  };
  const std::string_view fields[] = {"u"};
  prep.prepare(req, fields);

  REQUIRE(x_halos_all_equal(u, poison));
}

TEST_CASE("StagePreparationService: reject/retry re-prepare restores ghosts",
          "[MPI][stage_preparation]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 4});
  auto decomp = decomposition::create(global_domain, {2, 1, 1});

  const int hw = 2;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  communication::StagePreparationService<double> prep;
  prep.bind("u", halo);

  communication::StagePreparationRequirements req{
      .region_kind = communication::RegionKind::All,
      .needs_halo_exchange = true,
      .needs_boundary_update = false,
  };
  const std::string_view fields[] = {"u"};
  prep.prepare(req, fields);
  REQUIRE(x_split_halos_match(u, mine, other));

  // Simulate reject: poison ghosts only; owned core stays accepted.
  poison_halos(u, -42.0);
  REQUIRE_FALSE(x_split_halos_match(u, mine, other));

  // No rollback API — re-prepare from accepted owned state.
  prep.prepare(req, fields);
  REQUIRE(x_split_halos_match(u, mine, other));
}

TEST_CASE("StagePreparationService: boundary hook ordering vs halo",
          "[MPI][stage_preparation]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) return;

  auto global_domain = pfc::domain::create({16, 8, 4});
  auto decomp = decomposition::create(global_domain, {2, 1, 1});

  const int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double base = 100.0 + static_cast<double>(rank);
  fill_owned(u, base);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  communication::StagePreparationService<double> prep;
  prep.bind("u", halo);

  std::vector<std::string> hook_log;
  constexpr double mutated = 777.0;
  prep.set_boundary_hook([&](std::string_view name) {
    hook_log.emplace_back(std::string("bc:") + std::string(name));
    // Mutate owned core so exchange can publish the new face values.
    fill_owned(u, mutated);
  });

  // BoundaryThenHalo: neighbor must receive mutated owned faces.
  {
    communication::StagePreparationRequirements req{
        .region_kind = communication::RegionKind::All,
        .needs_halo_exchange = true,
        .needs_boundary_update = true,
        .ordering = communication::BoundaryHaloOrder::BoundaryThenHalo,
    };
    hook_log.clear();
    fill_owned(u, base);
    poison_halos(u, -1.0);
    const std::string_view fields[] = {"u"};
    prep.prepare(req, fields);

    REQUIRE(hook_log.size() == 1);
    REQUIRE(hook_log[0] == "bc:u");
    REQUIRE(x_split_halos_match(u, mutated, mutated));
  }

  // HaloThenBoundary: exchange runs on pre-hook owned values; neighbor sees
  // base, then local owned is mutated after exchange.
  {
    communication::StagePreparationRequirements req{
        .region_kind = communication::RegionKind::All,
        .needs_halo_exchange = true,
        .needs_boundary_update = true,
        .ordering = communication::BoundaryHaloOrder::HaloThenBoundary,
    };
    hook_log.clear();
    fill_owned(u, base);
    poison_halos(u, -1.0);
    const std::string_view fields[] = {"u"};
    prep.prepare(req, fields);

    REQUIRE(hook_log.size() == 1);
    REQUIRE(hook_log[0] == "bc:u");
    const double other_base = 100.0 + static_cast<double>(1 - rank);
    // Local owned was mutated after exchange; X ghosts still hold neighbor's
    // pre-hook owned value.
    REQUIRE(x_halos_all_equal(u, other_base));
    REQUIRE(u(0, 0, 0) == mutated);
  }

  // requirements_from maps StageContext flags into prepare requirements.
  {
    integrator::StageContext ctx{
        .time = 0.0,
        .dt = 0.01,
        .stage_index = 0,
        .region_kind = integrator::StageContext::RegionKind::Interior,
        .needs_boundary_update = true,
        .needs_halo_exchange = false,
    };
    auto mapped = integrator::requirements_from(ctx);
    REQUIRE(mapped.region_kind == communication::RegionKind::Interior);
    REQUIRE(mapped.needs_boundary_update);
    REQUIRE_FALSE(mapped.needs_halo_exchange);
  }
}
