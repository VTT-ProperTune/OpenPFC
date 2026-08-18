// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/mpi/communicator.hpp>

TEST_CASE("communicator::duplicate isolates a copy of WORLD",
          "[mpi][communicator]") {
  pfc::mpi::communicator world;
  const auto dup = world.duplicate();
  REQUIRE(dup.size() == world.size());
  REQUIRE(dup.rank() == world.rank());
  REQUIRE(static_cast<MPI_Comm>(dup) != MPI_COMM_WORLD);
  REQUIRE(static_cast<MPI_Comm>(dup) != MPI_COMM_NULL);
}

TEST_CASE("communicator wrapping WORLD does not free WORLD",
          "[mpi][communicator]") {
  pfc::mpi::communicator world;
  REQUIRE(static_cast<MPI_Comm>(world) == MPI_COMM_WORLD);
  REQUIRE(world.size() >= 1);
}
