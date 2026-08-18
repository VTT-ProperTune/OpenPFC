// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Host HaloExchange facade (M4). Device HaloExchange is remaining M4 work;
// CUDA execution of that half is not available on LUMI.

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <stdexcept>
#include <vector>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/decomposition/halo_geometry.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

namespace {

void fill_owned(data::Field<double, HostSpace> &u, double val) {
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      for (int i = 0; i < n[0]; ++i)
        u(i, j, k) = val;
}

bool halo_x_matches(const data::Field<double, HostSpace> &u, int i, double expected) {
  bool matches = true;
  const auto n = u.size3();
  for (int k = 0; k < n[2]; ++k)
    for (int j = 0; j < n[1]; ++j)
      matches &= u(i, j, k) == expected;
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

TEST_CASE("HaloExchange persistent Faces: single-rank wrap", "[MPI][halo_exchange]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Multi-rank PersistentHaloExchanger is already red on LUMI (same as
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

TEST_CASE("HaloExchange two fields use disjoint tag blocks", "[MPI][halo_exchange]") {
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
