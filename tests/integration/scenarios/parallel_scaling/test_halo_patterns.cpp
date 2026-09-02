// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/decomposition/halo_pattern.hpp>

using namespace pfc;

TEST_CASE("Halo send/recv sizes match expected face areas",
          "[integration][mpi][halo]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto domain = domain::create(GridSize({24, 24, 24}), PhysicalOrigin({0.0, 0.0, 0.0}), GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(domain, size);

  const auto local_world = decomposition::local_box(decomp, rank);
  auto local_size = local_world.size;

  const int halo_width = 1;
  auto patterns = halo::create_halo_patterns(decomp, rank, halo::Connectivity::Faces,
                                             halo_width);

  bool pattern_sizes_match = true;
  for (const auto &entry : patterns) {
    const auto &dir = entry.first;
    const auto &send_recv = entry.second;
    const auto &send = send_recv.first;
    const auto &recv = send_recv.second;

    // Expected indices count based on direction (match world::get_size: inclusive
    // upper bounds → nx = upper - lower + 1)
    const auto nx = static_cast<long long>(local_size[0]);
    const auto ny = static_cast<long long>(local_size[1]);
    const auto nz = static_cast<long long>(local_size[2]);

    long long expected = 0;
    if (dir[0] != 0) {
      expected = static_cast<long long>(halo_width) * ny * nz;
    } else if (dir[1] != 0) {
      expected = nx * static_cast<long long>(halo_width) * nz;
    } else if (dir[2] != 0) {
      expected = nx * ny * static_cast<long long>(halo_width);
    }

    pattern_sizes_match &= static_cast<long long>(send.size()) == expected &&
                           static_cast<long long>(recv.size()) == expected;
  }
  REQUIRE(pattern_sizes_match);
}
