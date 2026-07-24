// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <mpi.h>
#include <openpfc/frontend/utils/memory_reporter.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/utils/logging.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>

TEST_CASE("MemoryUsage total_bytes sums application and FFT allocations",
          "[memory_reporter]") {
  const pfc::utils::MemoryUsage usage{1024, 2048};

  REQUIRE(usage.total_bytes() == 3072);
}

#if defined(__linux__)
TEST_CASE("get_system_memory_bytes reads Linux MemTotal", "[memory_reporter]") {
  REQUIRE(pfc::utils::get_system_memory_bytes() > 0);
}
#endif

TEST_CASE("throw_on_mpi_error functional test", "[frontend_utils][memory_reporter][MPI]") {
  // Direct test of the error checking wrapper that is used in report_memory_usage.
  // This verifies the core error handling mechanism works correctly with simulated
  // error codes, avoiding MPI library configuration issues with fatal error handlers.
  
  // Verify successful case (no exception)
  REQUIRE_NOTHROW(pfc::mpi::throw_on_mpi_error(MPI_SUCCESS, "MPI_Reduce test"));
  
  // Verify error case (throws std::runtime_error)
  REQUIRE_THROWS_AS(
      pfc::mpi::throw_on_mpi_error(MPI_ERR_COMM, "MPI_Reduce test"),
      std::runtime_error);
  
  // Verify error message includes context
  try {
    pfc::mpi::throw_on_mpi_error(MPI_ERR_OTHER, "MPI_Reduce in report_memory_usage");
    FAIL("expected std::runtime_error to be thrown");
  } catch (const std::runtime_error &e) {
    REQUIRE_THAT(e.what(), 
                  Catch::Matchers::ContainsSubstring("MPI_Reduce in report_memory_usage"));
  }
}

TEST_CASE("report_memory_usage has error checking wrapper", "[frontend_utils][memory_reporter][MPI]") {
  // Code inspection test to verify that MPI_Reduce in report_memory_usage
  // is properly wrapped with throw_on_mpi_error with descriptive context.
  // This follows the same pattern as halo_persistent MPI error verification.
  
  const auto header = std::filesystem::path(__FILE__).parent_path() / 
                      "../../../../include/openpfc/frontend/utils/memory_reporter.hpp";
  REQUIRE(std::filesystem::exists(header));
  
  // Read the source file
  std::ifstream in(header);
  REQUIRE(in.good());
  std::ostringstream oss;
  oss << in.rdbuf();
  const std::string src = oss.str();
  
  // Verify that throw_on_mpi_error wrapper is present around MPI_Reduce
  REQUIRE(src.find("throw_on_mpi_error") != std::string::npos);
  REQUIRE(src.find("MPI_Reduce") != std::string::npos);
  
  // Verify that the error context message mentions memory statistics aggregation
  REQUIRE(src.find("memory statistics aggregation") != std::string::npos);
}
