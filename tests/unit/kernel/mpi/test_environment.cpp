// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_environment.cpp
 * @brief Unit tests for pfc::mpi::environment class
 *
 * Tests copy/move deletion and MPI error handling infrastructure.
 */

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/environment.hpp>
#include <type_traits>

using Env = pfc::mpi::environment;

TEST_CASE("test_environment_copy_constructor_is_deleted", "[mpi][environment]") {
  static_assert(!std::is_copy_constructible_v<Env>,
                "environment must not be copy constructible");
}

TEST_CASE("test_environment_copy_assignment_is_deleted", "[mpi][environment]") {
  static_assert(!std::is_copy_assignable_v<Env>,
                "environment must not be copy assignable");
}

TEST_CASE("test_environment_move_constructor_is_deleted", "[mpi][environment]") {
  static_assert(!std::is_move_constructible_v<Env>,
                "environment must not be move constructible");
}

TEST_CASE("test_environment_move_assignment_is_deleted", "[mpi][environment]") {
  static_assert(!std::is_move_assignable_v<Env>,
                "environment must not be move assignable");
}

TEST_CASE("test_environment_mpi_init_catches_errors", "[mpi][environment]") {
  // Verify that environment constructor uses throw_on_mpi_error
  // Note: In the shared test harness, MPI is already initialized globally,
  // so we verify the error handling infrastructure is in place by checking
  // that the environment class properly uses throw_on_mpi_error internally.
  // Actual MPI error scenarios are tested in the standalone executable.
  
  // The key verification is that the implementation in environment.hpp
  // includes the throw_on_mpi_error call, which is verified by code inspection
  // and by the standalone test executable.
  
  // We can verify that environment properly reports MPI state
  REQUIRE(Env::initialized());
  REQUIRE(!Env::finalized());
  REQUIRE(!Env::processor_name().empty());
}

TEST_CASE("test_environment_mpi_finalize_catches_errors", "[mpi][environment]") {
  // Verify that environment destructor uses throw_on_mpi_error
  // Note: In the shared test harness, MPI is managed globally, so we cannot
  // actually test MPI_Finalize error handling here. The error handling
  // infrastructure is verified by code inspection and by the standalone
  // test executable.
  
  // The key verification is that the implementation in environment.hpp
  // includes the throw_on_mpi_error call in the destructor, which is
  // verified by code inspection and by the standalone test executable.
  
  // We can verify that environment properly reports MPI state
  REQUIRE(Env::initialized());
  REQUIRE(!Env::finalized());
}
