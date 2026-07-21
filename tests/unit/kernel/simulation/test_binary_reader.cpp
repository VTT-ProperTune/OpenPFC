// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/model_types.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>

#include "../../../fixtures/mpi_file_guard_test_utils.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

// MPI is already initialized once for the whole openpfc-tests executable by
// runtests.cpp's MPI_Worker singleton -- this file must not call MPI_Init
// itself or define its own main().

using namespace pfc;

TEST_CASE("BinaryReader does not leak an MPI_File handle when MPI_File_open fails",
          "[binary_reader][mpi][leak]") {
  BinaryReader reader;
  const std::array<int, 3> global{2, 2, 1};
  const std::array<int, 3> local{2, 2, 1};
  const std::array<int, 3> offset{0, 0, 0};
  reader.set_domain(global, local, offset);

  RealField data(4, 0.0);
  // MPI_File_open itself fails here (nonexistent file/directory), so no
  // MPI_File handle is ever produced to leak.
  REQUIRE_THROWS_AS(
      reader.read("/nonexistent_dir_for_binary_reader_test/in.bin", data),
      std::runtime_error);
}

TEST_CASE(
    "MPI_File_guard closes the handle when MPI_File_set_view fails on read after "
    "open",
    "[binary_reader][mpi][leak]") {
  // Same rationale as the writer-side set_view test: m_filetype is always a
  // validly-committed subarray type through BinaryReader's public API once
  // set_domain() succeeds, so this exercises the RAII guard directly against
  // a genuine MPI_File_set_view failure (MPI_DATATYPE_NULL).
  const std::string filename = "/tmp/test_mpi_file_guard_reader_set_view_fail.bin";
  {
    std::ofstream touch(filename, std::ios::binary);
  }
  MPI_File fh{};
  pfc::mpi::throw_on_mpi_error(MPI_File_open(MPI_COMM_WORLD,
                                             const_cast<char *>(filename.c_str()),
                                             MPI_MODE_RDONLY, MPI_INFO_NULL, &fh),
                               "MPI_File_open");
  REQUIRE(pfc::test::is_path_open(filename));

  bool threw = false;
  {
    pfc::mpi::MPI_File_guard guard(fh);
    try {
      pfc::mpi::throw_on_mpi_error(MPI_File_set_view(fh, 0, MPI_DATATYPE_NULL,
                                                     MPI_DATATYPE_NULL, "native",
                                                     MPI_INFO_NULL),
                                   "MPI_File_set_view");
    } catch (const std::runtime_error &) {
      threw = true;
    }
  }

  REQUIRE(threw);
  REQUIRE_FALSE(pfc::test::is_path_open(filename));
  std::filesystem::remove(filename);
}

TEST_CASE("MPI_File_guard closes the handle when MPI_File_read_all fails after open",
          "[binary_reader][mpi][leak]") {
  const std::string filename = "/tmp/test_mpi_file_guard_read_all_fail.bin";
  {
    std::ofstream touch(filename, std::ios::binary);
    const double zero = 0.0;
    touch.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
  }
  MPI_File fh{};
  pfc::mpi::throw_on_mpi_error(MPI_File_open(MPI_COMM_WORLD,
                                             const_cast<char *>(filename.c_str()),
                                             MPI_MODE_RDONLY, MPI_INFO_NULL, &fh),
                               "MPI_File_open");
  REQUIRE(pfc::test::is_path_open(filename));

  bool threw = false;
  {
    pfc::mpi::MPI_File_guard guard(fh);
    std::vector<double> data(1, 0.0);
    MPI_Status status{};
    try {
      pfc::mpi::throw_on_mpi_error(MPI_File_read_all(fh, data.data(),
                                                     static_cast<int>(data.size()),
                                                     MPI_DATATYPE_NULL, &status),
                                   "MPI_File_read_all");
    } catch (const std::runtime_error &) {
      threw = true;
    }
  }

  REQUIRE(threw);
  REQUIRE_FALSE(pfc::test::is_path_open(filename));
  std::filesystem::remove(filename);
}
