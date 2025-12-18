// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>

using namespace pfc;

TEST_CASE("Global coverage equals sum of local coverage",
          "[integration][mpi][decomposition]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = world::uniform(24, 1.0);
  auto decomp = decomposition::create(world, size);

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  long long local_cells = static_cast<long long>(local_size[0]) *
                          static_cast<long long>(local_size[1]) *
                          static_cast<long long>(local_size[2]);

  long long global_sum = 0;
  MPI_Allreduce(&local_cells, &global_sum, 1, MPI_LONG_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  auto global_size = world::get_size(world);
  long long global_cells = static_cast<long long>(global_size[0]) *
                           static_cast<long long>(global_size[1]) *
                           static_cast<long long>(global_size[2]);

  if (rank == 0) {
    REQUIRE(global_sum == global_cells);
  }

  // Every rank should have non-zero local coverage when size > 1
  if (size > 1) {
    REQUIRE(local_cells > 0);
  } else {
    SUCCEED("Single rank run - multi-rank coverage check skipped");
  }
}
