// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>

TEST_CASE("ensure_mpi_int_count accepts INT_MAX and below",
          "[mpi][int_count]") {
  REQUIRE(pfc::mpi::ensure_mpi_int_count(0u, "t") == 0);
  REQUIRE(pfc::mpi::ensure_mpi_int_count(static_cast<std::size_t>(INT_MAX),
                                         "t") == INT_MAX);
}

TEST_CASE("ensure_mpi_int_count throws above INT_MAX", "[mpi][int_count]") {
  const std::size_t oversize = static_cast<std::size_t>(INT_MAX) + 1u;
  REQUIRE_THROWS_AS(
      pfc::mpi::ensure_mpi_int_count(oversize, "exchange::send"),
      std::overflow_error);

  try {
    (void)pfc::mpi::ensure_mpi_int_count(oversize, "exchange::send");
    FAIL("expected std::overflow_error");
  } catch (const std::overflow_error &ex) {
    const std::string msg = ex.what();
    REQUIRE(msg.find("exchange::send") != std::string::npos);
    REQUIRE(msg.find("INT_MAX") != std::string::npos);
  }
}
