// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_halo_exchange_driver.cpp
 * @brief Integration tests for HaloExchange (two-rank face sync).
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/comm_halo_exchange.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;

TEST_CASE("HaloExchange syncs face values across ranks",
          "[integration][mpi][halo][driver]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    return;
  }

  // 2x1x1 decomposition: rank 0 = left half, rank 1 = right half in X
  auto domain = domain::create(GridSize({24, 24, 24}), PhysicalOrigin({0.0, 0.0, 0.0}), GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(domain, {2, 1, 1});

  constexpr int hw = 1;
  auto field = data::field_from_subdomain<double>(decomp, rank, hw);
  const auto fill = static_cast<double>(rank);
  field.for_each_owned([&](int i, int j, int k) { field(i, j, k) = fill; });

  comm::HaloExchange<HostSpace, double> halo(field, decomp, rank, MPI_COMM_WORLD);
  halo.exchange();

  const auto n = field.local_size();
  const double other = static_cast<double>(1 - rank);
  bool face_matches = true;
  for (int z = 0; z < n[2]; ++z) {
    for (int y = 0; y < n[1]; ++y) {
      face_matches &= field(-hw, y, z) == other;
      face_matches &= field(n[0], y, z) == other;
    }
  }
  REQUIRE(face_matches);
}
