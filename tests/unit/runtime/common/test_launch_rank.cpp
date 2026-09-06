// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>

#include <openpfc/runtime/common/launch_rank.hpp>

namespace {

void clear_launch_rank_env() {
  unsetenv("SLURM_LOCALID");
  unsetenv("OMPI_COMM_WORLD_LOCAL_RANK");
  unsetenv("MPI_LOCALRANKID");
  unsetenv("PALS_LOCAL_RANKID");
}

} // namespace

TEST_CASE("local_rank_from_launch_env reads SLURM_LOCALID",
          "[runtime][launch_rank]") {
  clear_launch_rank_env();
  REQUIRE(pfc::runtime::local_rank_from_launch_env() == -1);
  REQUIRE(setenv("SLURM_LOCALID", "3", 1) == 0);
  REQUIRE(pfc::runtime::local_rank_from_launch_env() == 3);
  REQUIRE(setenv("SLURM_LOCALID", "0", 1) == 0);
  REQUIRE(pfc::runtime::local_rank_from_launch_env() == 0);
  REQUIRE(setenv("SLURM_LOCALID", "not-a-rank", 1) == 0);
  REQUIRE(pfc::runtime::local_rank_from_launch_env() == -1);
  clear_launch_rank_env();
}

TEST_CASE("local_rank_from_launch_env prefers SLURM over Open MPI",
          "[runtime][launch_rank]") {
  clear_launch_rank_env();
  REQUIRE(setenv("OMPI_COMM_WORLD_LOCAL_RANK", "7", 1) == 0);
  REQUIRE(pfc::runtime::local_rank_from_launch_env() == 7);
  REQUIRE(setenv("SLURM_LOCALID", "1", 1) == 0);
  REQUIRE(pfc::runtime::local_rank_from_launch_env() == 1);
  clear_launch_rank_env();
}
