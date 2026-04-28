// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>

using namespace pfc;

TEST_CASE("SimulationContext is_rank0 matches MPI_COMM_WORLD",
          "[simulation_context][unit]") {
  const SimulationContext ctx{MPI_COMM_WORLD};
  REQUIRE(ctx.mpi_comm() == MPI_COMM_WORLD);
  REQUIRE(ctx.is_rank0() == (mpi::get_rank(MPI_COMM_WORLD) == 0));
}

TEST_CASE("SimulationContext default constructor uses MPI_COMM_WORLD",
          "[simulation_context][unit]") {
  const SimulationContext ctx;
  REQUIRE(ctx.mpi_comm() == MPI_COMM_WORLD);
  REQUIRE(ctx.is_rank0() == (mpi::get_rank(MPI_COMM_WORLD) == 0));
}

TEST_CASE("SimulationContext null communicator is not rank 0",
          "[simulation_context][unit]") {
  const SimulationContext ctx{MPI_COMM_NULL};
  REQUIRE_FALSE(ctx.is_rank0());
}

TEST_CASE("mpi_comm_rank_is_zero matches MPI_Comm_rank",
          "[simulation_context][unit]") {
  REQUIRE(mpi_comm_rank_is_zero(MPI_COMM_WORLD) ==
          (mpi::get_rank(MPI_COMM_WORLD) == 0));
  REQUIRE_FALSE(mpi_comm_rank_is_zero(MPI_COMM_NULL));
}
