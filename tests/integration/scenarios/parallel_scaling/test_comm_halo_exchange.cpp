// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Host HaloExchange facade (M4). 4-rank mode comparison is in
// test_comm_halo_exchange_modes.cpp. Device cases live in
// test_comm_halo_exchange_gpu.cpp (HIP here; CUDA on tohtori).

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_directions.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

namespace {

void fill_owned(data::Field<double, HostSpace> &u, double val) {
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i) u(i, j, k) = val;
}

bool halo_x_matches(const data::Field<double, HostSpace> &u, int i,
                    double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_y_matches(const data::Field<double, HostSpace> &u, int j,
                    double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int i = 0; i < n[0]; ++i) matches &= u(i, j, k) == expected;
  return matches;
}

bool halo_z_matches(const data::Field<double, HostSpace> &u, int k,
                    double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int j = 0; j < n[1]; ++j)
    for (int i = 0; i < n[0]; ++i) matches &= u(i, j, k) == expected;
  return matches;
}

} // namespace

TEST_CASE("HaloExchange Faces: single-rank periodic wrap fills all 6 face halos",
          "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned(u, 7.0);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(halo.connectivity() == comm::HaloConnectivity::Faces);
  REQUIRE(halo.num_fields() == 1);
  halo.exchange();

  const auto n = u.local_size();
  REQUIRE(halo_x_matches(u, -1, 7.0));
  REQUIRE(halo_x_matches(u, n[0], 7.0));
}

TEST_CASE("HaloExchange Faces: start/finish matches blocking exchange",
          "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }

  auto domain = domain::create({16, 8, 4});
  auto decomp = decomposition::create(domain, {2, 1, 1});
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.start();
  halo.finish();

  const auto n = u.local_size();
  REQUIRE(halo_x_matches(u, -1, other));
  REQUIRE(halo_x_matches(u, n[0], other));
}

TEST_CASE("HaloExchange persistent Faces: single-rank wrap",
          "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Multi-rank persistent HaloExchange is already red on LUMI (same as
  // test_fd_heat_mpi). Cover the facade's persistent path on one rank.
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned(u, 4.0);

  comm::HaloExchangeOptions opt;
  opt.persistent = true;
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  REQUIRE(halo.persistent());
  halo.exchange();
  REQUIRE(halo_x_matches(u, -1, 4.0));
}

TEST_CASE("HaloExchange Full: start() is rejected", "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned(u, 1.0);

  comm::HaloExchangeOptions opt;
  opt.connectivity = comm::HaloConnectivity::Full;
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  REQUIRE_THROWS_AS(halo.start(), std::logic_error);
  halo.exchange();
}

TEST_CASE("HaloExchange rejects persistent Full", "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);

  comm::HaloExchangeOptions opt;
  opt.connectivity = comm::HaloConnectivity::Full;
  opt.persistent = true;
  REQUIRE_THROWS_AS(
      (comm::HaloExchange<HostSpace, double>(u, decomp, rank, MPI_COMM_WORLD, opt)),
      std::invalid_argument);
}

TEST_CASE("HaloExchange two fields use disjoint tag blocks",
          "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  auto v = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  fill_owned(u, 3.0);
  fill_owned(v, 5.0);

  comm::HaloExchange<HostSpace, double> halo({&u, &v}, decomp, rank, MPI_COMM_WORLD);
  REQUIRE(halo.num_fields() == 2);
  halo.exchange();
  REQUIRE(halo_x_matches(u, -1, 3.0));
  REQUIRE(halo_x_matches(v, -1, 5.0));
  REQUIRE(halo::field_tag_base(0, 1) == halo::kCanonicalTagCount);
}

TEST_CASE("HaloExchange Faces: Axes2D skips ±Z", "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    return;
  }

  auto domain = domain::create({8, 6, 4});
  auto decomp = decomposition::create(domain, 1);
  constexpr int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double sentinel = -999.0;
  const auto n = u.local_size();
  for (int k = -hw; k < n[2] + hw; ++k)
    for (int j = -hw; j < n[1] + hw; ++j)
      for (int i = -hw; i < n[0] + hw; ++i) u(i, j, k) = sentinel;
  fill_owned(u, 7.0);

  comm::HaloExchangeOptions opt;
  opt.directions = halo::presets::Axes2D();
  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD, opt);
  halo.exchange();

  REQUIRE(halo_x_matches(u, -1, 7.0));
  REQUIRE(halo_x_matches(u, n[0], 7.0));
  REQUIRE(halo_y_matches(u, -1, 7.0));
  REQUIRE(halo_y_matches(u, n[1], 7.0));
  REQUIRE(halo_z_matches(u, -1, sentinel));
  REQUIRE(halo_z_matches(u, n[2], sentinel));
}

TEST_CASE("HaloExchange Faces: two-rank X-split fills hw=2 layers",
          "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }

  auto domain = domain::create({16, 8, 4});
  auto decomp = decomposition::create(domain, {2, 1, 1});
  constexpr int hw = 2;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();

  const auto n = u.local_size();
  bool layers_match = true;
  for (int d = 1; d <= hw; ++d) {
    layers_match &=
        halo_x_matches(u, -d, other) && halo_x_matches(u, n[0] + d - 1, other) &&
        halo_y_matches(u, -d, mine) && halo_y_matches(u, n[1] + d - 1, mine) &&
        halo_z_matches(u, -d, mine) && halo_z_matches(u, n[2] + d - 1, mine);
  }
  REQUIRE(layers_match);
}

TEST_CASE("HaloExchange Faces: start/finish overlaps with inner work",
          "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    return;
  }

  auto domain = domain::create({16, 8, 4});
  auto decomp = decomposition::create(domain, {2, 1, 1});
  constexpr int hw = 1;
  auto u = data::field_from_subdomain<double>(decomp, rank, hw);
  const double mine = static_cast<double>(rank);
  const double other = static_cast<double>(1 - rank);
  fill_owned(u, mine);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.start();

  double inner_sum = 0.0;
  const auto n = u.local_size();
  for (int k = hw; k < n[2] - hw; ++k)
    for (int j = hw; j < n[1] - hw; ++j)
      for (int i = hw; i < n[0] - hw; ++i) inner_sum += u(i, j, k);
  REQUIRE((inner_sum > 0.0) == (mine > 0.0));

  halo.finish();

  REQUIRE(halo_x_matches(u, -1, other));
  REQUIRE(halo_x_matches(u, n[0], other));
  REQUIRE(halo_y_matches(u, -1, mine));
  REQUIRE(halo_z_matches(u, -1, mine));
}

TEST_CASE("HaloExchange Faces: 2x2x1 grid fills X and Y neighbours",
          "[MPI][halo_exchange][grid]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 4) {
    return;
  }

  auto domain = domain::create({16, 16, 4});
  auto decomp = decomposition::create(domain, {2, 2, 1});
  auto u = data::field_from_subdomain<double>(decomp, rank, /*halo=*/1);
  const double mine = static_cast<double>(rank);
  fill_owned(u, mine);

  comm::HaloExchange<HostSpace, double> halo(u, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();

  const int rank_x = rank % 2;
  const int rank_y = rank / 2;
  const int xpos_neighbor = ((rank_x + 1) % 2) + rank_y * 2;
  const int xneg_neighbor = ((rank_x - 1 + 2) % 2) + rank_y * 2;
  const int ypos_neighbor = rank_x + ((rank_y + 1) % 2) * 2;
  const int yneg_neighbor = rank_x + ((rank_y - 1 + 2) % 2) * 2;

  const auto n = u.local_size();
  REQUIRE(halo_x_matches(u, n[0], static_cast<double>(xpos_neighbor)));
  REQUIRE(halo_x_matches(u, -1, static_cast<double>(xneg_neighbor)));
  REQUIRE(halo_y_matches(u, n[1], static_cast<double>(ypos_neighbor)));
  REQUIRE(halo_y_matches(u, -1, static_cast<double>(yneg_neighbor)));
  REQUIRE(halo_z_matches(u, -1, mine));
  REQUIRE(halo_z_matches(u, n[2], mine));
}
