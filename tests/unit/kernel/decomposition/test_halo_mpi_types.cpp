// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/decomposition/exchange.hpp>
#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>

TEST_CASE("create_face_types_6: thin local brick throws before MPI",
          "[halo][mpi_types]") {
  // nx=1 < hw=2 — send/recv slab of thickness hw cannot fit.
  REQUIRE_THROWS_AS(pfc::halo::create_face_types_6(1, 8, 8, 2, MPI_DOUBLE),
                    std::invalid_argument);
  try {
    (void)pfc::halo::create_face_types_6(1, 8, 8, 2, MPI_DOUBLE);
    FAIL("expected throw");
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("1x8x8") != std::string::npos);
    REQUIRE(msg.find("halo_width=2") != std::string::npos);
  }
}

TEST_CASE("create_face_types_6: borderline extent == hw constructs",
          "[halo][mpi_types]") {
  const int nx = 2, ny = 2, nz = 2;
  const int hw = 2;
  auto faces = pfc::halo::create_face_types_6(nx, ny, nz, hw, MPI_DOUBLE);

  int size = 0;
  MPI_Type_size(faces[0].send_type.get(), &size);
  REQUIRE(static_cast<std::size_t>(size) / sizeof(double) ==
          static_cast<std::size_t>(hw * ny * nz));
}

TEST_CASE("create_face_types_6: flat nz==1 Axes2D-style constructs",
          "[halo][mpi_types]") {
  // Orthogonal thin axis must remain valid (e.g. 8x8x1 + hw=1).
  auto faces = pfc::halo::create_face_types_6(8, 8, 1, 1, MPI_DOUBLE);
  int size = 0;
  MPI_Type_size(faces[0].send_type.get(), &size);
  REQUIRE(static_cast<std::size_t>(size) / sizeof(double) ==
          static_cast<std::size_t>(1 * 8 * 1));
}

TEST_CASE("uint64 exchange preserves full-width identities", "[halo][mpi_types]") {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const std::array<std::uint64_t, 4> values{
      (std::uint64_t{1} << 53) + rank,
      std::numeric_limits<std::uint64_t>::max() - rank, 0, std::uint64_t{1} << 63};
  std::array<std::uint64_t, 4> received{};
  const auto type = pfc::exchange::detail::get_mpi_type<std::uint64_t>();
  int bytes = 0;
  REQUIRE(MPI_Type_size(type, &bytes) == MPI_SUCCESS);
  REQUIRE(bytes == sizeof(std::uint64_t));
  const int source = (rank + size - 1) % size;
  REQUIRE(MPI_Sendrecv(values.data(), 4, type, (rank + 1) % size, 777,
                       received.data(), 4, type, source, 777, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE) == MPI_SUCCESS);
  auto expected = values;
  expected[0] = (std::uint64_t{1} << 53) + source;
  expected[1] = std::numeric_limits<std::uint64_t>::max() - source;
  REQUIRE(received == expected);
}
