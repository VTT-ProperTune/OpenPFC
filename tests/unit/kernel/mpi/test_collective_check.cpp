// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/collective_check.hpp>

TEST_CASE("collective acceptance uses the supplied communicator",
          "[MPI][collective_check]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm group = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &group);
  int local_rank = 0, size = 0;
  MPI_Comm_rank(group, &local_rank);
  MPI_Comm_size(group, &size);
  CHECK(pfc::mpi::all_ranks_succeeded(true, group));
  CHECK_FALSE(pfc::mpi::all_ranks_succeeded(local_rank != size - 1, group));
  CHECK_NOTHROW(pfc::mpi::require_all_ranks(true, group, "accepted"));
  CHECK_THROWS_AS(
      pfc::mpi::require_all_ranks(local_rank != size - 1, group, "rejected"),
      std::runtime_error);
  MPI_Comm_free(&group);
}
