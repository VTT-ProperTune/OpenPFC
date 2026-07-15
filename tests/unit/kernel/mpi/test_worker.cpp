// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/worker.hpp>
#include <type_traits>

TEST_CASE("MPI_Worker is not copy constructible", "[mpi][worker]") {
  STATIC_REQUIRE(!std::is_copy_constructible_v<pfc::MPI_Worker>);
}

TEST_CASE("MPI_Worker is not copy assignable", "[mpi][worker]") {
  STATIC_REQUIRE(!std::is_copy_assignable_v<pfc::MPI_Worker>);
}

TEST_CASE("MPI_Worker is not move constructible", "[mpi][worker]") {
  STATIC_REQUIRE(!std::is_move_constructible_v<pfc::MPI_Worker>);
}

TEST_CASE("MPI_Worker is not move assignable", "[mpi][worker]") {
  STATIC_REQUIRE(!std::is_move_assignable_v<pfc::MPI_Worker>);
}

