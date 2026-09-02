// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/fft/fft_fftw.hpp>

using namespace pfc;

TEST_CASE("Domain decomposition basic properties",
          "[integration][mpi][decomposition]") {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto domain = domain::create(GridSize({32, 32, 32}), PhysicalOrigin({0.0, 0.0, 0.0}), GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(domain, size);
  auto fft = fft::create(decomp);

  // Validate local subdomain for this rank
  const auto local_world = decomposition::local_box(decomp, rank);
  auto local_size = local_world.size;
  REQUIRE(local_size[0] > 0);
  REQUIRE(local_size[1] > 0);
  REQUIRE(local_size[2] > 0);

  // Inbox size matches local domain cell count
  const std::size_t local_cells = static_cast<std::size_t>(local_size[0]) *
                                  static_cast<std::size_t>(local_size[1]) *
                                  static_cast<std::size_t>(local_size[2]);
  REQUIRE(fft.size_inbox() == local_cells);
}
