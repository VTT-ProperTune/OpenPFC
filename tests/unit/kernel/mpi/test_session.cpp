// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <array>
#include <cstdint>

namespace {

void require_same_session(std::uint32_t case_id) {
  const std::array<std::uint32_t, 2> local{case_id, Catch::getSeed()};
  std::array<std::uint32_t, 2> low{}, high{};
  MPI_Allreduce(local.data(), low.data(), 2, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(local.data(), high.data(), 2, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);
  INFO("Ranks must enter the same case with the same random seed");
  CHECK(low == high);
}

} // namespace

TEST_CASE("test session agrees across ranks in its first case",
          "[MPI][test_session]") {
  require_same_session(1u);
}

TEST_CASE("test session agrees across ranks in its second case",
          "[MPI][test_session]") {
  require_same_session(2u);
}
