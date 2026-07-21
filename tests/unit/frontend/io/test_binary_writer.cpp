// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/mpi/mpi_io_helpers.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>

#include "../../../fixtures/mpi_file_guard_test_utils.hpp"

#include <array>
#include <climits>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pfc;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kTol = 1e-12;

struct DomainBrick {
  std::array<int, 3> global{};
  std::array<int, 3> local{};
  std::array<int, 3> offset{};
};

DomainBrick make_domain(int rank, int size) {
  // Global {4,2,2}. With 1 rank: whole brick. With 2+ ranks: split on x.
  DomainBrick d;
  d.global = {4, 2, 2};
  if (size == 1) {
    d.local = {4, 2, 2};
    d.offset = {0, 0, 0};
    return d;
  }
  REQUIRE(size >= 2);
  // Two non-overlapping bricks covering the global grid (ranks >1 idle-local empty
  // is not used; CTest runs n=1 and n=2 only).
  if (rank == 0) {
    d.local = {2, 2, 2};
    d.offset = {0, 0, 0};
  } else {
    d.local = {2, 2, 2};
    d.offset = {2, 0, 0};
  }
  return d;
}

std::size_t local_count(const DomainBrick &d) {
  return static_cast<std::size_t>(d.local[0]) * static_cast<std::size_t>(d.local[1]) *
         static_cast<std::size_t>(d.local[2]);
}

std::filesystem::path test_dir() {
  auto dir = std::filesystem::path(".temp") / "tests" / "unit" / "binary_writer";
  std::filesystem::create_directories(dir);
  return dir;
}

RealField make_real(const DomainBrick &d, int rank) {
  const auto n = local_count(d);
  RealField data(n);
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = static_cast<double>(i) + 1000.0 * static_cast<double>(rank) +
              0.25 * static_cast<double>(d.offset[0]);
  }
  return data;
}

ComplexField make_complex(const DomainBrick &d, int rank) {
  const auto n = local_count(d);
  ComplexField data(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double re = static_cast<double>(i) + 10.0 * static_cast<double>(rank);
    const double im = static_cast<double>(i) * 2.0 + static_cast<double>(d.offset[0]);
    data[i] = std::complex<double>(re, im);
  }
  return data;
}

} // namespace

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  Catch::Session session;
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

TEST_CASE("BinaryWriter RealField round-trip via BinaryReader",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size > 2) {
    SKIP("binary_writer tests support 1 or 2 ranks only");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto out_dir = test_dir();
  const std::string template_path =
      (out_dir / "real_%d.bin").string();
  const std::string path = (out_dir / "real_0.bin").string();

  const RealField written = make_real(domain, rank);

  {
    BinaryWriter writer(template_path);
    writer.set_domain(domain.global, domain.local, domain.offset);
    REQUIRE_NOTHROW(writer.write(0, written));
  }

  RealField read_back(local_count(domain));
  {
    BinaryReader reader;
    reader.set_domain(domain.global, domain.local, domain.offset);
    REQUIRE_NOTHROW(reader.read(path, read_back));
  }

  REQUIRE(read_back.size() == written.size());
  for (std::size_t i = 0; i < written.size(); ++i) {
    REQUIRE_THAT(read_back[i], WithinAbs(written[i], kTol));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("BinaryWriter ComplexField round-trip via BinaryReader",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size > 2) {
    SKIP("binary_writer tests support 1 or 2 ranks only");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto out_dir = test_dir();
  const std::string template_path =
      (out_dir / "complex_%d.bin").string();
  const std::string path = (out_dir / "complex_0.bin").string();

  const ComplexField written = make_complex(domain, rank);

  {
    BinaryWriter writer(template_path);
    writer.set_domain(domain.global, domain.local, domain.offset);
    REQUIRE_NOTHROW(writer.write(0, written));
  }

  ComplexField read_back(local_count(domain));
  {
    BinaryReader reader;
    reader.set_domain(domain.global, domain.local, domain.offset);
    REQUIRE_NOTHROW(reader.read(path, read_back));
  }

  REQUIRE(read_back.size() == written.size());
  for (std::size_t i = 0; i < written.size(); ++i) {
    REQUIRE_THAT(read_back[i].real(), WithinAbs(written[i].real(), kTol));
    REQUIRE_THAT(read_back[i].imag(), WithinAbs(written[i].imag(), kTol));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("BinaryWriter rebuilds filetype when switching real then complex",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size > 2) {
    SKIP("binary_writer tests support 1 or 2 ranks only");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto out_dir = test_dir();
  const std::string shared_tmpl = (out_dir / "shared_%d.bin").string();
  const std::string real_path = (out_dir / "shared_0.bin").string();
  const std::string cplx_path = (out_dir / "shared_1.bin").string();

  const RealField real_data = make_real(domain, rank);
  const ComplexField cplx_data = make_complex(domain, rank);

  // Same writer + domain; ensure_filetype rebuilds when etype changes.
  BinaryWriter writer(shared_tmpl);
  writer.set_domain(domain.global, domain.local, domain.offset);
  REQUIRE_NOTHROW(writer.write(0, real_data));
  REQUIRE_NOTHROW(writer.write(1, cplx_data));

  RealField real_back(local_count(domain));
  ComplexField cplx_back(local_count(domain));
  BinaryReader reader;
  reader.set_domain(domain.global, domain.local, domain.offset);
  REQUIRE_NOTHROW(reader.read(real_path, real_back));
  REQUIRE_NOTHROW(reader.read(cplx_path, cplx_back));

  for (std::size_t i = 0; i < real_data.size(); ++i) {
    REQUIRE_THAT(real_back[i], WithinAbs(real_data[i], kTol));
  }
  for (std::size_t i = 0; i < cplx_data.size(); ++i) {
    REQUIRE_THAT(cplx_back[i].real(), WithinAbs(cplx_data[i].real(), kTol));
    REQUIRE_THAT(cplx_back[i].imag(), WithinAbs(cplx_data[i].imag(), kTol));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(real_path);
    std::filesystem::remove(cplx_path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("BinaryWriter and BinaryReader reject invalid domain geometry",
          "[binary_writer][io][validation][unit]") {
  if (mpi::get_size() > 1) {
    SKIP("binary domain validation tests require single MPI rank");
  }

  auto require_invalid_domain = [](auto &io) {
    REQUIRE_THROWS_AS(io.set_domain({0, 8, 8}, {0, 8, 8}, {0, 0, 0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(io.set_domain({8, 8, 8}, {0, 8, 8}, {0, 0, 0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(io.set_domain({8, 8, 8}, {8, 8, 8}, {-1, 0, 0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(io.set_domain({8, 8, 8}, {8, 8, 8}, {4, 0, 0}),
                      std::invalid_argument);
  };

  SECTION("BinaryWriter") {
    BinaryWriter writer("unused.bin");
    require_invalid_domain(writer);
    REQUIRE_NOTHROW(writer.set_domain({8, 8, 8}, {8, 8, 8}, {0, 0, 0}));
  }

  SECTION("BinaryReader") {
    BinaryReader reader;
    require_invalid_domain(reader);
    REQUIRE_NOTHROW(reader.set_domain({8, 8, 8}, {8, 8, 8}, {0, 0, 0}));
  }

  SECTION("invalid attempts leave BinaryWriter domain unconfigured") {
    BinaryWriter writer("unused.bin");
    REQUIRE_THROWS_AS(writer.set_domain({8, 8, 8}, {8, 8, 8}, {4, 0, 0}),
                      std::invalid_argument);
    // Domain members stay default (non-positive); write fails closed before
    // MPI-IO (checked_local_extent_product), not via a successful configure.
    REQUIRE_THROWS_AS(writer.write(0, RealField(1)), std::invalid_argument);
  }
}

TEST_CASE("BinaryWriter does not leak an MPI_File handle when MPI_File_open fails",
          "[binary_writer][mpi][leak]") {
  BinaryWriter writer("/nonexistent_dir_for_binary_writer_test/out_%04d.bin");
  const std::array<int, 3> global{2, 2, 1};
  const std::array<int, 3> local{2, 2, 1};
  const std::array<int, 3> offset{0, 0, 0};
  writer.set_domain(global, local, offset);

  // MPI_File_open itself fails here (nonexistent directory), so no MPI_File
  // handle is ever produced to leak -- the property under test is simply
  // that the failure propagates as a clean exception rather than hanging.
  REQUIRE_THROWS_AS(writer.write(1, RealField(4, 1.0)), std::runtime_error);
}

TEST_CASE("checked_local_extent_product rejects overflow and non-positive extents",
          "[binary_writer][io]") {
  REQUIRE_THROWS_AS(
      pfc::mpi::checked_local_extent_product({INT_MAX, INT_MAX, INT_MAX},
                                             "test"),
      std::overflow_error);
  REQUIRE_THROWS_AS(pfc::mpi::checked_local_extent_product({0, 2, 2}, "test"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pfc::mpi::checked_local_extent_product({2, -1, 2}, "test"),
                    std::invalid_argument);
  REQUIRE(pfc::mpi::checked_local_extent_product({4, 2, 2}, "test") == 16u);
}

TEST_CASE("BinaryWriter::write fails closed on buffer size mismatch",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size > 2) {
    SKIP("binary_writer tests support 1 or 2 ranks only");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto n = local_count(domain);
  const auto out_dir = test_dir();
  const std::string template_path = (out_dir / "mismatch_write_%d.bin").string();

  BinaryWriter writer(template_path);
  writer.set_domain(domain.global, domain.local, domain.offset);

  REQUIRE_THROWS_AS(writer.write(0, RealField(n - 1, 0.0)), std::runtime_error);
  REQUIRE_THROWS_AS(writer.write(0, RealField(n + 1, 0.0)), std::runtime_error);
}

TEST_CASE("BinaryReader::read fails closed on buffer size mismatch",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size > 2) {
    SKIP("binary_writer tests support 1 or 2 ranks only");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto n = local_count(domain);
  const auto out_dir = test_dir();
  const std::string template_path = (out_dir / "mismatch_read_%d.bin").string();
  const std::string path = (out_dir / "mismatch_read_0.bin").string();

  const RealField written = make_real(domain, rank);
  {
    BinaryWriter writer(template_path);
    writer.set_domain(domain.global, domain.local, domain.offset);
    REQUIRE_NOTHROW(writer.write(0, written));
  }

  BinaryReader reader;
  reader.set_domain(domain.global, domain.local, domain.offset);

  RealField too_small(n - 1);
  RealField too_large(n + 1);
  REQUIRE_THROWS_AS(reader.read(path, too_small), std::runtime_error);
  REQUIRE_THROWS_AS(reader.read(path, too_large), std::runtime_error);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("BinaryWriter buffer mismatch on one rank fails closed for all ranks",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size != 2) {
    SKIP("single-rank mismatch Allreduce case requires exactly 2 ranks");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto n = local_count(domain);
  const auto out_dir = test_dir();
  const std::string template_path =
      (out_dir / "mismatch_peer_write_%d.bin").string();

  BinaryWriter writer(template_path);
  writer.set_domain(domain.global, domain.local, domain.offset);

  // Rank 0 posts a wrong-sized buffer; rank 1 is correctly sized. Allreduce
  // must make every rank throw before MPI_File_open.
  RealField data = (rank == 0) ? RealField(n + 1, 0.0) : make_real(domain, rank);
  REQUIRE_THROWS_AS(writer.write(0, data), std::runtime_error);
}

TEST_CASE("BinaryReader buffer mismatch on one rank fails closed for all ranks",
          "[binary_writer][io]") {
  const int rank = mpi::get_rank();
  const int size = mpi::get_size();
  if (size != 2) {
    SKIP("single-rank mismatch Allreduce case requires exactly 2 ranks");
  }

  const DomainBrick domain = make_domain(rank, size);
  const auto n = local_count(domain);
  const auto out_dir = test_dir();
  const std::string template_path =
      (out_dir / "mismatch_peer_read_%d.bin").string();
  const std::string path = (out_dir / "mismatch_peer_read_0.bin").string();

  {
    BinaryWriter writer(template_path);
    writer.set_domain(domain.global, domain.local, domain.offset);
    REQUIRE_NOTHROW(writer.write(0, make_real(domain, rank)));
  }

  BinaryReader reader;
  reader.set_domain(domain.global, domain.local, domain.offset);

  RealField data = (rank == 0) ? RealField(n - 1) : RealField(n);
  REQUIRE_THROWS_AS(reader.read(path, data), std::runtime_error);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::filesystem::remove(path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_CASE("MPI_File_guard closes the handle when MPI_File_set_view fails after open",
          "[binary_writer][mpi][leak]") {
  const std::string filename = "/tmp/test_mpi_file_guard_set_view_fail.bin";
  MPI_File fh{};
  pfc::mpi::throw_on_mpi_error(
      MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filename.c_str()),
                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh),
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

TEST_CASE(
    "MPI_File_guard closes the handle when MPI_File_write_all fails after open",
    "[binary_writer][mpi][leak]") {
  const std::string filename = "/tmp/test_mpi_file_guard_write_all_fail.bin";
  MPI_File fh{};
  pfc::mpi::throw_on_mpi_error(
      MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filename.c_str()),
                    MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh),
      "MPI_File_open");
  REQUIRE(pfc::test::is_path_open(filename));

  bool threw = false;
  {
    pfc::mpi::MPI_File_guard guard(fh);
    std::vector<double> data(4, 1.0);
    MPI_Status status{};
    try {
      pfc::mpi::throw_on_mpi_error(MPI_File_write_all(fh, data.data(),
                                                      static_cast<int>(data.size()),
                                                      MPI_DATATYPE_NULL, &status),
                                   "MPI_File_write_all");
    } catch (const std::runtime_error &) {
      threw = true;
    }
  }

  REQUIRE(threw);
  REQUIRE_FALSE(pfc::test::is_path_open(filename));
  std::filesystem::remove(filename);
}
