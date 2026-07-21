// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

#include <openpfc/kernel/decomposition/halo_mpi_types.hpp>

TEST_CASE("create_face_types_6: thin local brick throws before MPI",
          "[halo][mpi_types]") {
  // nx=3 <= 2*hw=4 — overlapping / negative-start face slabs.
  REQUIRE_THROWS_AS(pfc::halo::create_face_types_6(3, 8, 8, 2, MPI_DOUBLE),
                    std::invalid_argument);
  try {
    (void)pfc::halo::create_face_types_6(3, 8, 8, 2, MPI_DOUBLE);
    FAIL("expected throw");
  } catch (const std::invalid_argument &e) {
    const std::string msg = e.what();
    REQUIRE(msg.find("3x8x8") != std::string::npos);
    REQUIRE(msg.find("halo_width=2") != std::string::npos);
  }
}

TEST_CASE("create_face_types_6: borderline nx > 2*hw constructs",
          "[halo][mpi_types]") {
  const int nx = 5, ny = 5, nz = 5;
  const int hw = 2;
  auto faces = pfc::halo::create_face_types_6(nx, ny, nz, hw, MPI_DOUBLE);

  int size = 0;
  MPI_Type_size(faces[0].send_type.get(), &size);
  REQUIRE(static_cast<std::size_t>(size) / sizeof(double) ==
          static_cast<std::size_t>(hw * ny * nz));
}
