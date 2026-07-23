// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/frontend/io/png_writer.hpp>
#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/data/world_factory.hpp>
#include <openpfc/kernel/data/world_queries.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

#include <filesystem>
#include <string>
#include <vector>

using namespace pfc;
using namespace pfc::io;

namespace {

// Test fixture for PNG writer tests
struct PNGWriterTestFixture {
  MPI_Comm m_comm = MPI_COMM_WORLD;
  int m_rank = 0;
  int m_num_ranks = 1;

  PNGWriterTestFixture() {
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_num_ranks);
  }

  ~PNGWriterTestFixture() {
    // Clean up any test files on rank 0
    if (m_rank == 0) {
      std::filesystem::remove("test_collective_size_mismatch.png");
      std::filesystem::remove("test_valid_write.png");
      std::filesystem::remove("test_invalid_nz.png");
    }
  }

  // Create a simple World and decomposition for testing
  std::pair<world::World, decomposition::Decomposition>
  create_test_decomp(int nx_global, int ny_global) {
    auto world =
        world::create(GridSize({nx_global, ny_global, 1}),
                      PhysicalOrigin({0.0, 0.0, 0.0}), GridSpacing({1.0, 1.0, 1.0}));
    auto decomp = make_decomposition(world, m_comm);
    return {world, decomp};
  }
};

} // namespace

// Custom main to initialize MPI before tests run
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  Catch::Session session;
  // Keep Catch2's SECTION ordering and shuffling identical on all MPI ranks
  // (otherwise different --rng-seed defaults can reorder nested SECTIONs and
  // deadlock MPI collectives inside tests).
  session.configData().rngSeed = 1u;
  session.configData().runOrder = Catch::TestRunOrder::Declared;

  const int cli = session.applyCommandLine(argc, argv);
  if (cli != 0) {
    MPI_Finalize();
    return cli;
  }
  const int result = session.run();
  MPI_Finalize();
  return result;
}

TEST_CASE("PNGWriter - Valid write with correct sizes", "[png_writer][io]") {
  PNGWriterTestFixture fixture;

  const int nx_global = 8;
  const int ny_global = 8;

  auto [world, decomp] = fixture.create_test_decomp(nx_global, ny_global);

  // Get the expected local size for this rank
  const auto &local_world = decomposition::get_subworld(decomp, fixture.m_rank);
  auto local_size = world::get_size(local_world);
  const int expected_pts = local_size[0] * local_size[1] * local_size[2];

  // Create test data with correct size
  std::vector<double> data(expected_pts);
  for (int i = 0; i < expected_pts; ++i) {
    data[static_cast<std::size_t>(i)] = static_cast<double>(i) * 0.1;
  }

  // Should succeed without throwing
  REQUIRE_NOTHROW(write_mpi_scalar_field_png_xy(
      fixture.m_comm, decomp, fixture.m_rank, data, "test_valid_write.png"));
}

// MPI tests must not use nested Catch2 SECTIONs: each rank advances through the
// SECTION tree independently, which can mis-match MPI collectives in VTKWriter.

TEST_CASE("PNGWriter - Collective size mismatch (all ranks throw)",
          "[png][mpi][error]") {
  PNGWriterTestFixture fixture;

  // Only run this test with multiple MPI ranks
  if (fixture.m_num_ranks < 2) {
    SKIP("Test requires multiple MPI ranks to verify collective error agreement");
  }

  const int nx_global = 4;
  const int ny_global = 4;

  auto [world, decomp] = fixture.create_test_decomp(nx_global, ny_global);

  // Get the expected local size for this rank
  const auto &local_world = decomposition::get_subworld(decomp, fixture.m_rank);
  auto local_size = world::get_size(local_world);
  const int expected_pts = local_size[0] * local_size[1] * local_size[2];

  // Create test data with WRONG size on rank 0 only
  std::vector<double> data;

  if (fixture.m_rank == 0) {
    // Rank 0: create data with wrong size (one less than expected)
    data = std::vector<double>(expected_pts - 1);
    for (int i = 0; i < expected_pts - 1; ++i) {
      data[static_cast<std::size_t>(i)] = static_cast<double>(i) * 0.1;
    }
  } else {
    // Other ranks: create data with correct size
    data = std::vector<double>(expected_pts);
    for (int i = 0; i < expected_pts; ++i) {
      data[static_cast<std::size_t>(i)] = static_cast<double>(i) * 0.1;
    }
  }

  // ALL ranks should throw std::runtime_error due to collective error agreement
  // This verifies that the collective error agreement prevents deadlock
  // in MPI_Allgather/MPI_Gatherv
  REQUIRE_THROWS_AS(
      write_mpi_scalar_field_png_xy(fixture.m_comm, decomp, fixture.m_rank, data,
                                    "test_collective_size_mismatch.png"),
      std::runtime_error);
}

TEST_CASE("PNGWriter - Global nz validation (single rank)", "[png_writer][io]") {
  PNGWriterTestFixture fixture;

  // Create a 3D world (nz=8) which should fail validation
  auto world = world::create(GridSize({8, 8, 8}), PhysicalOrigin({0.0, 0.0, 0.0}),
                             GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = make_decomposition(world, fixture.m_comm);

  const auto &local_world = decomposition::get_subworld(decomp, fixture.m_rank);
  auto local_size = world::get_size(local_world);
  const int expected_pts = local_size[0] * local_size[1] * local_size[2];

  std::vector<double> data(expected_pts);
  for (int i = 0; i < expected_pts; ++i) {
    data[static_cast<std::size_t>(i)] = static_cast<double>(i) * 0.1;
  }

  // Should throw because global nz != 1
  REQUIRE_THROWS_AS(write_mpi_scalar_field_png_xy(fixture.m_comm, decomp,
                                                  fixture.m_rank, data,
                                                  "test_invalid_nz.png"),
                    std::invalid_argument);
}
