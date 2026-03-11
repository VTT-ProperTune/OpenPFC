// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_halo_exchange_driver.cpp
 * @brief Integration tests for HaloExchanger (non-blocking face exchange)
 */

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_exchange.hpp>

using namespace pfc;

TEST_CASE("HaloExchanger exchange_halos syncs face values across ranks",
          "[integration][mpi][halo][driver]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    return; // need at least 2 ranks
  }

  // 2x1x1 decomposition: rank 0 = left half, rank 1 = right half in X
  auto world = world::uniform(24, 1.0);
  auto decomp = decomposition::create(world, {2, 1, 1});

  auto local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  size_t local_total = static_cast<size_t>(local_size[0]) *
                       static_cast<size_t>(local_size[1]) *
                       static_cast<size_t>(local_size[2]);

  std::vector<double> field(local_total);
  double fill = static_cast<double>(rank);
  for (size_t i = 0; i < local_total; ++i) {
    field[i] = fill;
  }

  HaloExchanger<double> exchanger(decomp, rank, 1, MPI_COMM_WORLD);

  exchanger.exchange_halos(field.data(), field.size());

  // Recv for +X is leftmost face (x=0); recv for -X is rightmost face (x=nx-1).
  // Rank 0 receives from +X (rank 1) into left face -> rank 0's x=0 layer = 1.0.
  // Rank 1 receives from -X (rank 0) into right face -> rank 1's x=nx-1 layer = 0.0.
  int nx = local_size[0], ny = local_size[1], nz = local_size[2];
  if (rank == 0) {
    for (int z = 0; z < nz; ++z) {
      for (int y = 0; y < ny; ++y) {
        size_t idx = static_cast<size_t>(z) * static_cast<size_t>(ny) *
                         static_cast<size_t>(nx) +
                     static_cast<size_t>(y) * static_cast<size_t>(nx) + 0;
        REQUIRE(field[idx] == 1.0);
      }
    }
  } else if (rank == 1) {
    for (int z = 0; z < nz; ++z) {
      for (int y = 0; y < ny; ++y) {
        size_t idx = static_cast<size_t>(z) * static_cast<size_t>(ny) *
                         static_cast<size_t>(nx) +
                     static_cast<size_t>(y) * static_cast<size_t>(nx) +
                     static_cast<size_t>(nx - 1);
        REQUIRE(field[idx] == 0.0);
      }
    }
  }
}
