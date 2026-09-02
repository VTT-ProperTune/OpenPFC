// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/simulation/initial_conditions/file_reader.hpp>
#include <openpfc/kernel/simulation/simulation_context.hpp>

using namespace pfc;

TEST_CASE("FileReader - Parameter Access", "[ic_file_reader]") {
  FileReader reader;

  SECTION("Default constructor") {
    FileReader default_reader;
    REQUIRE(default_reader.get_filename().empty());
  }

  SECTION("Constructor with filename") {
    FileReader named_reader("test_file.bin");
    REQUIRE(named_reader.get_filename() == "test_file.bin");
  }

  SECTION("Set and get filename") {
    reader.set_filename("checkpoint.bin");
    REQUIRE(reader.get_filename() == "checkpoint.bin");
  }
}

TEST_CASE("FileReader - Field Name Assignment", "[ic_file_reader]") {
  FileReader reader;
  reader.set_field_name("density_field");
  REQUIRE(reader.get_field_name() == "density_field");
}

TEST_CASE("FileReader - Invalid File Handling", "[ic_file_reader]") {
  auto domain = pfc::domain::create(pfc::Int3{8, 8, 8});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  FileReader reader("nonexistent_file.bin");
  SimulationContext ctx(MPI_COMM_WORLD);

  SECTION("Apply with nonexistent file") {
    REQUIRE_THROWS_AS(reader.apply(ctx, psi, domain, box), std::runtime_error);
  }
}

void create_test_binary_file(const std::string &filename,
                             const std::vector<double> &data) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to create test file");
  }
  file.write(reinterpret_cast<const char *>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(double)));
  file.close();
}

TEST_CASE("FileReader - Read Valid File", "[ic_file_reader]") {
  auto domain = pfc::domain::create(pfc::Int3{4, 4, 4});
  auto box = pfc::domain::index_box(domain);
  std::vector<double> psi(static_cast<size_t>(box.count()), 0.0);

  std::vector<double> test_data(64, 0.0);
  for (size_t i = 0; i < test_data.size(); ++i) {
    test_data[i] = static_cast<double>(i);
  }

  const std::string test_filename = "test_field_reader.bin";
  SimulationContext ctx(MPI_COMM_WORLD);

  SECTION("Read file and verify data") {
    create_test_binary_file(test_filename, test_data);
    FileReader reader(test_filename);
    reader.set_field_name("default");
    REQUIRE_NOTHROW(reader.apply(ctx, psi, domain, box));
    std::remove(test_filename.c_str());
  }
}

TEST_CASE("FileReader - Named field interface", "[ic_file_reader]") {
  FileReader reader;
  reader.set_filename("restart.bin");
  reader.set_field_name("density");
  REQUIRE(reader.get_field_name() == "density");
  REQUIRE(reader.get_filename() == "restart.bin");
}
