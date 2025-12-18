// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/core/decomposition.hpp>
#include <openpfc/core/halo_pattern.hpp>
#include <openpfc/core/world.hpp>

using namespace pfc;

TEST_CASE("Halo send/recv sizes match expected face areas",
          "[integration][mpi][halo]") {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto world = world::uniform(24, 1.0);
  auto decomp = decomposition::create(world, size);

  const auto &local_world = decomposition::get_subworld(decomp, rank);
  auto local_size = world::get_size(local_world);
  auto local_lower = world::get_lower(local_world);
  auto local_upper = world::get_upper(local_world);

  const int halo_width = 1;
  auto patterns = halo::create_halo_patterns(decomp, rank, halo::Connectivity::Faces,
                                             halo_width);

  for (const auto &entry : patterns) {
    const auto &dir = entry.first;
    const auto &send_recv = entry.second;
    const auto &send = send_recv.first;
    const auto &recv = send_recv.second;

    // Expected indices count based on direction
    // Compute lengths directly from bounds used by halo implementation
    const long long nx = static_cast<long long>(local_upper[0] - local_lower[0]);
    const long long ny = static_cast<long long>(local_upper[1] - local_lower[1]);
    const long long nz = static_cast<long long>(local_upper[2] - local_lower[2]);

    long long expected = 0;
    if (dir[0] != 0) {
      expected = static_cast<long long>(halo_width) * ny * nz;
    } else if (dir[1] != 0) {
      expected = nx * static_cast<long long>(halo_width) * nz;
    } else if (dir[2] != 0) {
      expected = nx * ny * static_cast<long long>(halo_width);
    }

    REQUIRE(static_cast<long long>(send.size()) == expected);
    REQUIRE(static_cast<long long>(recv.size()) == expected);
  }
}
