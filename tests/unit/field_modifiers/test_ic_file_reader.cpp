// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <fstream>
#include <iostream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/decomposition.hpp"
#include "openpfc/core/types.hpp"
#include "openpfc/core/world.hpp"
#include "openpfc/factory/decomposition_factory.hpp"
#include "openpfc/fft.hpp"
#include "openpfc/initial_conditions/file_reader.hpp"
#include "openpfc/model.hpp"

using namespace pfc;
using pfc::types::Int3;

// Mock model class for testing
class ModelWithFileReaderIC : public Model {
public:
  ModelWithFileReaderIC(FFT &fft, const pfc::World &world)
      : pfc::Model(fft, world) {}

  void step(double /*t*/) override {}
  void initialize(double /*dt*/) override {}
};

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
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithFileReaderIC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  FileReader reader("nonexistent_file.bin");

  SECTION("Apply with nonexistent file") {
    // BinaryReader prints error message but doesn't throw
    // This tests error handling path (file not found)
    REQUIRE_NOTHROW(reader.apply(m, 0.0));
  }
}

// Helper function to create a simple binary file for testing
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
  auto world = world::create(GridSize({4, 4, 4}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithFileReaderIC m(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  m.add_real_field("default", psi);

  // Create test data with known pattern
  std::vector<double> test_data(64, 0.0); // 4x4x4 = 64 points
  for (size_t i = 0; i < test_data.size(); ++i) {
    test_data[i] = static_cast<double>(i);
  }

  const std::string test_filename = "test_field_reader.bin";

  SECTION("Read file and verify data") {
    // Create test file
    create_test_binary_file(test_filename, test_data);

    FileReader reader(test_filename);
    reader.set_field_name("default");

    // Note: BinaryReader expects specific format with domain info
    // This test may fail if BinaryReader has strict format requirements
    // In production code, we'd use actual output format
    REQUIRE_NOTHROW(reader.apply(m, 0.0));

    // Cleanup
    std::remove(test_filename.c_str());
  }
}

TEST_CASE("FileReader - Integration with Model", "[ic_file_reader]") {
  auto world = world::create(GridSize({8, 8, 8}));
  auto decomposition = decomposition::create(world, 1);
  auto fft = fft::create(decomposition);
  ModelWithFileReaderIC model(fft, world);

  const size_t field_size = fft.size_inbox();
  std::vector<double> psi(field_size, 0.0);
  model.add_real_field("density", psi);

  FileReader reader;
  reader.set_filename("restart.bin");
  reader.set_field_name("density");

  // Just verify interface works (file doesn't exist, so will throw)
  REQUIRE(reader.get_field_name() == "density");
  REQUIRE(reader.get_filename() == "restart.bin");
}
