// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/frontend/io/binary_writer.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>
#include <openpfc/kernel/simulation/binary_reader.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <filesystem>
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
