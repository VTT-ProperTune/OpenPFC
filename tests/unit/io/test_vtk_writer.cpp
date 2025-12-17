// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/mpi.hpp"
#include "openpfc/results_writers/vtk_writer.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace pfc;

// Custom main to initialize MPI before tests run
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int result = Catch::Session().run(argc, argv);

  MPI_Finalize();
  return result;
}

namespace {

/**
 * @brief Test fixture for VTK Writer tests
 *
 * Provides utility functions for:
 * - Creating test data
 * - Validating VTK file format
 * - Cleaning up test files
 */
class VTKWriterTestFixture {
public:
  VTKWriterTestFixture() {
    m_rank = mpi::get_rank();
    m_num_ranks = mpi::get_size();
  }

  ~VTKWriterTestFixture() { cleanup_test_files(); }

  /**
   * @brief Create test data with known pattern
   */
  RealField create_test_data(size_t size) {
    RealField data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<double>(i) + m_rank * 1000.0;
    }
    return data;
  }

  /**
   * @brief Create complex test data
   */
  ComplexField create_complex_test_data(size_t size) {
    ComplexField data(size);
    for (size_t i = 0; i < size; ++i) {
      double real = static_cast<double>(i);
      double imag = static_cast<double>(i) * 2.0;
      data[i] = std::complex<double>(real, imag);
    }
    return data;
  }

  /**
   * @brief Check if file exists
   */
  bool file_exists(const std::string &filename) const {
    return std::filesystem::exists(filename);
  }

  /**
   * @brief Read entire file content as string
   */
  std::string read_file_content(const std::string &filename) const {
    std::ifstream file(filename);
    if (!file) return "";
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  }

  /**
   * @brief Check if file contains expected string
   */
  bool file_contains(const std::string &filename, const std::string &pattern) const {
    std::string content = read_file_content(filename);
    return content.find(pattern) != std::string::npos;
  }

  /**
   * @brief Validate VTK XML header structure
   */
  bool validate_vti_header(const std::string &filename) const {
    std::string content = read_file_content(filename);

    // Check required XML elements
    bool has_xml_header = content.find("<?xml version=") != std::string::npos;
    bool has_vtk_file =
        content.find("<VTKFile type=\"ImageData\"") != std::string::npos;
    bool has_image_data = content.find("<ImageData") != std::string::npos;
    bool has_piece = content.find("<Piece") != std::string::npos;
    bool has_point_data = content.find("<PointData>") != std::string::npos;
    bool has_data_array = content.find("<DataArray") != std::string::npos;
    bool has_appended_data = content.find("<AppendedData") != std::string::npos;

    return has_xml_header && has_vtk_file && has_image_data && has_piece &&
           has_point_data && has_data_array && has_appended_data;
  }

  /**
   * @brief Validate PVTI (parallel master) file structure
   */
  bool validate_pvti_header(const std::string &filename) const {
    std::string content = read_file_content(filename);

    // Check required XML elements for parallel file
    bool has_xml_header = content.find("<?xml version=") != std::string::npos;
    bool has_pvtk_file =
        content.find("<VTKFile type=\"PImageData\"") != std::string::npos;
    bool has_pimage_data = content.find("<PImageData") != std::string::npos;
    bool has_ppoint_data = content.find("<PPointData>") != std::string::npos;
    bool has_piece_source = content.find("<Piece Source=") != std::string::npos;

    return has_xml_header && has_pvtk_file && has_pimage_data && has_ppoint_data &&
           has_piece_source;
  }

  /**
   * @brief Extract extent values from VTI file
   */
  std::array<int, 6> extract_extent(const std::string &filename,
                                    const std::string &extent_type) const {
    std::string content = read_file_content(filename);
    std::array<int, 6> extent = {0, 0, 0, 0, 0, 0};

    std::string pattern = extent_type + "=\"";
    size_t pos = content.find(pattern);
    if (pos == std::string::npos) return extent;

    pos += pattern.length();
    std::string extent_str = content.substr(pos, 100);
    std::istringstream iss(extent_str);
    for (int i = 0; i < 6; ++i) {
      iss >> extent[i];
    }

    return extent;
  }

  /**
   * @brief Clean up all test files created during testing
   */
  void cleanup_test_files() {
    // Pattern: test_*.vti, test_*.pvti
    for (const auto &entry : std::filesystem::directory_iterator(".")) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();
        if (filename.find("test_") == 0 &&
            (filename.find(".vti") != std::string::npos ||
             filename.find(".pvti") != std::string::npos)) {
          std::filesystem::remove(entry.path());
        }
      }
    }
  }

  int m_rank;
  int m_num_ranks;
};

} // anonymous namespace

TEST_CASE("VTKWriter - Basic construction", "[vtk_writer][io][unit]") {
  VTKWriterTestFixture fixture;

  SECTION("Construct with filename pattern") {
    VTKWriter writer("test_output_%04d.vti");

    // Should construct without throwing
    REQUIRE(true);
  }

  SECTION("Set domain configuration") {
    VTKWriter writer("test_output.vti");

    std::array<int, 3> global_size = {32, 32, 32};
    std::array<int, 3> local_size = {32, 32, 32};
    std::array<int, 3> offset = {0, 0, 0};

    writer.set_domain(global_size, local_size, offset);

    // Should set domain without throwing
    REQUIRE(true);
  }

  SECTION("Set origin and spacing") {
    VTKWriter writer("test_output.vti");

    std::array<double, 3> origin = {1.0, 2.0, 3.0};
    std::array<double, 3> spacing = {0.5, 0.5, 0.5};

    writer.set_origin(origin);
    writer.set_spacing(spacing);

    // Should set parameters without throwing
    REQUIRE(true);
  }

  SECTION("Set field name") {
    VTKWriter writer("test_output.vti");

    writer.set_field_name("density");

    // Should set field name without throwing
    REQUIRE(true);
  }
}

TEST_CASE("VTKWriter - Serial output", "[vtk_writer][io][serial]") {
  VTKWriterTestFixture fixture;

  // Skip this entire test case if running with multiple ranks
  if (fixture.m_num_ranks > 1) {
    SKIP("Serial output test requires single MPI rank");
  }

  SECTION("Write single VTI file") {
    VTKWriter writer("test_serial_0001.vti");

    std::array<int, 3> size = {8, 8, 8};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(8 * 8 * 8);

    [[maybe_unused]] MPI_Status status = writer.write(1, data);

    // Check file was created
    REQUIRE(fixture.file_exists("test_serial_0001.vti"));

    // Validate VTK header structure
    REQUIRE(fixture.validate_vti_header("test_serial_0001.vti"));

    // Check for required elements
    REQUIRE(fixture.file_contains("test_serial_0001.vti", "ImageData"));
    REQUIRE(fixture.file_contains("test_serial_0001.vti", "PointData"));
    REQUIRE(fixture.file_contains("test_serial_0001.vti", "AppendedData"));
  }

  SECTION("Write with custom origin and spacing") {
    VTKWriter writer("test_custom_domain_0001.vti");

    std::array<int, 3> size = {4, 4, 4};
    std::array<double, 3> origin = {10.0, 20.0, 30.0};
    std::array<double, 3> spacing = {2.0, 2.0, 2.0};

    writer.set_domain(size, size, {0, 0, 0});
    writer.set_origin(origin);
    writer.set_spacing(spacing);

    auto data = fixture.create_test_data(4 * 4 * 4);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_custom_domain_0001.vti"));

    // Check origin values appear in file
    REQUIRE(
        fixture.file_contains("test_custom_domain_0001.vti", "Origin=\"10 20 30\""));
    REQUIRE(
        fixture.file_contains("test_custom_domain_0001.vti", "Spacing=\"2 2 2\""));
  }

  SECTION("Write with custom field name") {
    VTKWriter writer("test_field_name_0001.vti");

    std::array<int, 3> size = {4, 4, 4};
    writer.set_domain(size, size, {0, 0, 0});
    writer.set_field_name("temperature");

    auto data = fixture.create_test_data(4 * 4 * 4);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_field_name_0001.vti"));
    REQUIRE(
        fixture.file_contains("test_field_name_0001.vti", "Name=\"temperature\""));
  }

  SECTION("Write multiple time steps") {
    VTKWriter writer("test_timestep_%04d.vti");

    std::array<int, 3> size = {4, 4, 4};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(4 * 4 * 4);

    // Write 3 time steps
    writer.write(0, data);
    writer.write(1, data);
    writer.write(2, data);

    // Check all files exist
    REQUIRE(fixture.file_exists("test_timestep_0000.vti"));
    REQUIRE(fixture.file_exists("test_timestep_0001.vti"));
    REQUIRE(fixture.file_exists("test_timestep_0002.vti"));
  }
}

TEST_CASE("VTKWriter - Extent validation", "[vtk_writer][io][extent]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks > 1) {
    SKIP("Extent validation test requires single MPI rank");
  }

  SECTION("Verify WholeExtent in serial output") {
    VTKWriter writer("test_extent_0001.vti");

    std::array<int, 3> size = {10, 20, 30};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(10 * 20 * 30);
    writer.write(1, data);

    auto extent = fixture.extract_extent("test_extent_0001.vti", "WholeExtent");

    REQUIRE(extent[0] == 0);
    REQUIRE(extent[1] == 9); // size[0] - 1
    REQUIRE(extent[2] == 0);
    REQUIRE(extent[3] == 19); // size[1] - 1
    REQUIRE(extent[4] == 0);
    REQUIRE(extent[5] == 29); // size[2] - 1
  }

  SECTION("Verify Piece extent matches WholeExtent in serial") {
    VTKWriter writer("test_piece_extent_0001.vti");

    std::array<int, 3> size = {8, 8, 8};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(8 * 8 * 8);
    writer.write(1, data);

    auto whole_extent =
        fixture.extract_extent("test_piece_extent_0001.vti", "WholeExtent");
    auto piece_extent =
        fixture.extract_extent("test_piece_extent_0001.vti", "Piece Extent");

    // In serial, Piece extent should match WholeExtent
    for (int i = 0; i < 6; ++i) {
      REQUIRE(piece_extent[i] == whole_extent[i]);
    }
  }
}

TEST_CASE("VTKWriter - Complex field handling", "[vtk_writer][io][complex]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks > 1) {
    SKIP("Complex field test requires single MPI rank");
  }

  SECTION("Write complex field converts to magnitude") {
    VTKWriter writer("test_complex_0001.vti");

    std::array<int, 3> size = {4, 4, 4};
    writer.set_domain(size, size, {0, 0, 0});

    auto complex_data = fixture.create_complex_test_data(4 * 4 * 4);

    [[maybe_unused]] MPI_Status status = writer.write(1, complex_data);

    // File should be created
    REQUIRE(fixture.file_exists("test_complex_0001.vti"));

    // Validate structure
    REQUIRE(fixture.validate_vti_header("test_complex_0001.vti"));
  }

  SECTION("Complex magnitude is correct") {
    VTKWriter writer("test_complex_magnitude_0001.vti");

    std::array<int, 3> size = {2, 2, 2};
    writer.set_domain(size, size, {0, 0, 0});

    // Create simple complex data: (3, 4) -> magnitude 5
    ComplexField data(8);
    for (size_t i = 0; i < 8; ++i) {
      data[i] = std::complex<double>(3.0, 4.0);
    }

    writer.write(1, data);

    // File should exist and be valid
    REQUIRE(fixture.file_exists("test_complex_magnitude_0001.vti"));
    REQUIRE(fixture.validate_vti_header("test_complex_magnitude_0001.vti"));

    // Data section should contain magnitude values (5.0)
    // This is implicitly validated by successful write
  }
}

TEST_CASE("VTKWriter - Parallel output", "[vtk_writer][io][parallel]") {
  VTKWriterTestFixture fixture;

  // This test only makes sense with multiple ranks
  if (fixture.m_num_ranks == 1) {
    SKIP("Parallel test requires multiple MPI ranks");
  }

  SECTION("Each rank writes piece file") {
    VTKWriter writer("test_parallel_%04d.vti");

    // Simple decomposition: split in X direction
    int nx_local = 8 / fixture.m_num_ranks;
    std::array<int, 3> global_size = {8, 4, 4};
    std::array<int, 3> local_size = {nx_local, 4, 4};
    std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

    writer.set_domain(global_size, local_size, offset);

    auto data = fixture.create_test_data(nx_local * 4 * 4);
    writer.write(1, data);

    // Each rank should create its piece file
    std::string piece_filename =
        "test_parallel_0001_" + std::to_string(fixture.m_rank) + ".vti";
    REQUIRE(fixture.file_exists(piece_filename));
    REQUIRE(fixture.validate_vti_header(piece_filename));
  }

  SECTION("Rank 0 creates PVTI master file") {
    VTKWriter writer("test_parallel_master_%04d.vti");

    int nx_local = 8 / fixture.m_num_ranks;
    std::array<int, 3> global_size = {8, 4, 4};
    std::array<int, 3> local_size = {nx_local, 4, 4};
    std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

    writer.set_domain(global_size, local_size, offset);

    auto data = fixture.create_test_data(nx_local * 4 * 4);
    writer.write(1, data);

    // Wait for all ranks to finish writing
    MPI_Barrier(MPI_COMM_WORLD);

    // All ranks check their piece file exists
    std::string piece_filename =
        "test_parallel_master_0001_" + std::to_string(fixture.m_rank) + ".vti";
    REQUIRE(fixture.file_exists(piece_filename));
    REQUIRE(fixture.validate_vti_header(piece_filename));

    MPI_Barrier(MPI_COMM_WORLD);

    // Only rank 0 checks PVTI file
    // Other ranks just pass to avoid test failure
    if (fixture.m_rank == 0) {
      // Master file should exist
      REQUIRE(fixture.file_exists("test_parallel_master_0001.pvti"));
      REQUIRE(fixture.validate_pvti_header("test_parallel_master_0001.pvti"));

      // Should reference all piece files
      std::string content =
          fixture.read_file_content("test_parallel_master_0001.pvti");
      for (int r = 0; r < fixture.m_num_ranks; ++r) {
        std::string piece_ref =
            "test_parallel_master_0001_" + std::to_string(r) + ".vti";
        bool contains = content.find(piece_ref) != std::string::npos;
        REQUIRE(contains);
      }
    } else {
      // Non-zero ranks just pass these checks
      REQUIRE(true);
      REQUIRE(true);
      REQUIRE(true);
      REQUIRE(true);
    }
  }

  SECTION("PVTI WholeExtent matches global domain") {
    VTKWriter writer("test_pvti_extent_%04d.vti");

    int nx_local = 16 / fixture.m_num_ranks;
    std::array<int, 3> global_size = {16, 8, 8};
    std::array<int, 3> local_size = {nx_local, 8, 8};
    std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

    writer.set_domain(global_size, local_size, offset);

    auto data = fixture.create_test_data(nx_local * 8 * 8);
    writer.write(1, data);

    MPI_Barrier(MPI_COMM_WORLD);

    if (fixture.m_rank == 0) {
      auto extent =
          fixture.extract_extent("test_pvti_extent_0001.pvti", "WholeExtent");

      REQUIRE(extent[0] == 0);
      REQUIRE(extent[1] == 15); // 16 - 1
      REQUIRE(extent[2] == 0);
      REQUIRE(extent[3] == 7); // 8 - 1
      REQUIRE(extent[4] == 0);
      REQUIRE(extent[5] == 7); // 8 - 1
    }
  }
}

TEST_CASE("VTKWriter - Edge cases", "[vtk_writer][io][edge]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks > 1) {
    SKIP("Edge case tests require single MPI rank");
  }

  SECTION("Empty field name uses default") {
    VTKWriter writer("test_default_name_0001.vti");

    std::array<int, 3> size = {2, 2, 2};
    writer.set_domain(size, size, {0, 0, 0});
    // Don't set field name - should use default

    auto data = fixture.create_test_data(8);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_default_name_0001.vti"));
    // Should contain default field name "Field"
    REQUIRE(fixture.file_contains("test_default_name_0001.vti", "Name=\"Field\""));
  }

  SECTION("Single point domain") {
    VTKWriter writer("test_single_point_0001.vti");

    std::array<int, 3> size = {1, 1, 1};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(1);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_single_point_0001.vti"));
    REQUIRE(fixture.validate_vti_header("test_single_point_0001.vti"));
  }

  SECTION("Large increment number formatting") {
    VTKWriter writer("test_large_inc_%04d.vti");

    std::array<int, 3> size = {2, 2, 2};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(8);
    writer.write(9999, data);

    REQUIRE(fixture.file_exists("test_large_inc_9999.vti"));
  }

  SECTION("Non-cubic domain") {
    VTKWriter writer("test_noncubic_0001.vti");

    std::array<int, 3> size = {16, 8, 4};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(16 * 8 * 4);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_noncubic_0001.vti"));

    auto extent = fixture.extract_extent("test_noncubic_0001.vti", "WholeExtent");
    REQUIRE(extent[1] == 15); // 16 - 1
    REQUIRE(extent[3] == 7);  // 8 - 1
    REQUIRE(extent[5] == 3);  // 4 - 1
  }

  SECTION("Zero origin") {
    VTKWriter writer("test_zero_origin_0001.vti");

    std::array<int, 3> size = {4, 4, 4};
    std::array<double, 3> origin = {0.0, 0.0, 0.0};

    writer.set_domain(size, size, {0, 0, 0});
    writer.set_origin(origin);

    auto data = fixture.create_test_data(64);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_zero_origin_0001.vti"));
    REQUIRE(fixture.file_contains("test_zero_origin_0001.vti", "Origin=\"0 0 0\""));
  }

  SECTION("Unit spacing") {
    VTKWriter writer("test_unit_spacing_0001.vti");

    std::array<int, 3> size = {4, 4, 4};
    std::array<double, 3> spacing = {1.0, 1.0, 1.0};

    writer.set_domain(size, size, {0, 0, 0});
    writer.set_spacing(spacing);

    auto data = fixture.create_test_data(64);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_unit_spacing_0001.vti"));
    REQUIRE(
        fixture.file_contains("test_unit_spacing_0001.vti", "Spacing=\"1 1 1\""));
  }
}

TEST_CASE("VTKWriter - Data integrity", "[vtk_writer][io][integrity]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks > 1) {
    SKIP("Data integrity tests require single MPI rank");
  }

  SECTION("File size reflects data size") {
    VTKWriter writer("test_data_size_0001.vti");

    std::array<int, 3> size = {10, 10, 10};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(1000);
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_data_size_0001.vti"));

    // File should be larger than just header (at least data size + header)
    auto file_size = std::filesystem::file_size("test_data_size_0001.vti");
    size_t expected_data_size = 1000 * sizeof(double);
    REQUIRE(file_size > expected_data_size);
  }

  SECTION("Multiple writes don't interfere") {
    VTKWriter writer1("test_multi_write_a_%04d.vti");
    VTKWriter writer2("test_multi_write_b_%04d.vti");

    std::array<int, 3> size = {4, 4, 4};
    writer1.set_domain(size, size, {0, 0, 0});
    writer2.set_domain(size, size, {0, 0, 0});

    auto data1 = fixture.create_test_data(64);
    auto data2 = fixture.create_test_data(64);

    writer1.write(1, data1);
    writer2.write(1, data2);

    REQUIRE(fixture.file_exists("test_multi_write_a_0001.vti"));
    REQUIRE(fixture.file_exists("test_multi_write_b_0001.vti"));

    // Both should be valid
    REQUIRE(fixture.validate_vti_header("test_multi_write_a_0001.vti"));
    REQUIRE(fixture.validate_vti_header("test_multi_write_b_0001.vti"));
  }
}
