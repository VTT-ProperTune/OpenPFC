// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/world.hpp>
#include <openpfc/fft.hpp>

using namespace pfc;

TEST_CASE("Domain decomposition basic properties",
          "[integration][mpi][decomposition]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = world::uniform(32, 1.0);
  auto decomp = decomposition::create(world, size);
  auto fft = fft::create(decomp);

  // Validate local subdomain for this rank
  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  REQUIRE(local_size[0] > 0);
  REQUIRE(local_size[1] > 0);
  REQUIRE(local_size[2] > 0);

  // Inbox size matches local domain cell count
  const std::size_t local_cells = static_cast<std::size_t>(local_size[0]) *
                                  static_cast<std::size_t>(local_size[1]) *
                                  static_cast<std::size_t>(local_size[2]);
  REQUIRE(fft.size_inbox() == local_cells);
}
