// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>

TEST_CASE("expect_mpi_io_count accepts zero and INT_MAX",
          "[mpi_io_helpers][mpi]") {
  REQUIRE(pfc::mpi::expect_mpi_io_count(0u, "test") == 0);
  REQUIRE(pfc::mpi::expect_mpi_io_count(static_cast<std::size_t>(INT_MAX),
                                        "test") == INT_MAX);
}

TEST_CASE("expect_mpi_io_count throws on INT_MAX+1", "[mpi_io_helpers][mpi]") {
  const std::size_t oversize = static_cast<std::size_t>(INT_MAX) + 1u;
  REQUIRE_THROWS_AS(
      pfc::mpi::expect_mpi_io_count(oversize, "BinaryWriter::write"),
      std::overflow_error);

  try {
    (void)pfc::mpi::expect_mpi_io_count(oversize, "BinaryWriter::write");
    FAIL("expected std::overflow_error");
  } catch (const std::overflow_error &ex) {
    const std::string msg = ex.what();
    REQUIRE(msg.find("BinaryWriter::write") != std::string::npos);
    REQUIRE(msg.find("INT_MAX") != std::string::npos);
  }
}
