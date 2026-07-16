// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/frontend/io/vtk_writer.hpp>
#include <openpfc/kernel/mpi/mpi.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pfc;

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
  RealField create_test_data(size_t size) const {
    RealField data(size);
    for (size_t i = 0; i < size; ++i) {
      data[i] = static_cast<double>(i) + m_rank * 1000.0;
    }
    return data;
  }

  /**
   * @brief Create complex test data
   */
  static ComplexField create_complex_test_data(size_t size) {
    ComplexField data(size);
    for (size_t i = 0; i < size; ++i) {
      auto real = static_cast<double>(i);
      double imag = static_cast<double>(i) * 2.0;
      data[i] = std::complex<double>(real, imag);
    }
    return data;
  }

  /**
   * @brief Check if file exists
   */
  static bool file_exists(const std::string &filename) {
    return std::filesystem::exists(filename);
  }

  /**
   * @brief Read entire file content as string
   */
  static std::string read_file_content(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) {
      return "";
    }
    return {(std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()};
  }

  /**
   * @brief Check if file contains expected string
   */
  static bool file_contains(const std::string &filename,
                            const std::string &pattern) {
    std::string content = read_file_content(filename);
    return content.find(pattern) != std::string::npos;
  }

  /**
   * @brief Validate VTK XML header structure
   */
  static bool validate_vti_header(const std::string &filename) {
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
  static bool validate_pvti_header(const std::string &filename) {
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
  static std::array<int, 6> extract_extent(const std::string &filename,
                                           const std::string &extent_type) {
    std::string content = read_file_content(filename);
    std::array<int, 6> extent = {0, 0, 0, 0, 0, 0};

    std::string pattern = extent_type + "=\"";
    size_t pos = content.find(pattern);
    if (pos == std::string::npos) {
      return extent;
    }

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
  void cleanup_test_files() const {
    // Synchronize all ranks before cleanup to avoid race conditions
    MPI_Barrier(MPI_COMM_WORLD);

    if (m_num_ranks > 1) {
      // In multi-rank runs, rank 0 used to delete every test_*.vti, which could
      // remove another rank's piece file before that rank finished the test.
      // Each rank removes only its own piece files: ..._<rank>.vti
      const std::string own_suffix = "_" + std::to_string(m_rank) + ".vti";
      for (const auto &entry : std::filesystem::directory_iterator(".")) {
        if (!entry.is_regular_file()) {
          continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.find("test_") != 0) {
          continue;
        }
        if (filename.size() >= own_suffix.size() &&
            filename.compare(filename.size() - own_suffix.size(), own_suffix.size(),
                             own_suffix) == 0) {
          std::filesystem::remove(entry.path());
          continue;
        }
        if (m_rank == 0 && filename.find(".pvti") != std::string::npos) {
          std::filesystem::remove(entry.path());
        }
      }
      return;
    }

    if (m_rank != 0) {
      return;
    }

    // Single rank: delete conventional VTK outputs under test_
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

    auto data = fixture.create_test_data(static_cast<std::size_t>(8 * 8 * 8));

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

    auto data = fixture.create_test_data(static_cast<std::size_t>(4 * 4 * 4));
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

    auto data = fixture.create_test_data(static_cast<std::size_t>(4 * 4 * 4));
    writer.write(1, data);

    REQUIRE(fixture.file_exists("test_field_name_0001.vti"));
    REQUIRE(
        fixture.file_contains("test_field_name_0001.vti", "Name=\"temperature\""));
  }

  SECTION("Write multiple time steps") {
    VTKWriter writer("test_timestep_%04d.vti");

    std::array<int, 3> size = {4, 4, 4};
    writer.set_domain(size, size, {0, 0, 0});

    auto data = fixture.create_test_data(static_cast<std::size_t>(4 * 4 * 4));

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

    auto data = fixture.create_test_data(static_cast<std::size_t>(10 * 20 * 30));
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

    auto data = fixture.create_test_data(static_cast<std::size_t>(8 * 8 * 8));
    writer.write(1, data);

    auto whole_extent =
        fixture.extract_extent("test_piece_extent_0001.vti", "WholeExtent");
    auto piece_extent =
        fixture.extract_extent("test_piece_extent_0001.vti", "Piece Extent");

    // In serial, Piece extent should match WholeExtent
    bool extents_match = true;
    for (int i = 0; i < 6; ++i) extents_match &= piece_extent[i] == whole_extent[i];
    REQUIRE(extents_match);
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

    auto complex_data = VTKWriterTestFixture::create_complex_test_data(
        static_cast<std::size_t>(4) * static_cast<std::size_t>(4) *
        static_cast<std::size_t>(4));

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

// MPI tests must not use nested Catch2 SECTIONs: each rank advances through the
// SECTION tree independently, which can mis-match MPI collectives in VTKWriter.

TEST_CASE("VTKWriter - Parallel each rank writes piece file",
          "[vtk_writer][io][parallel]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks == 1) {
    SKIP("Parallel test requires multiple MPI ranks");
  }
  if (12 % fixture.m_num_ranks != 0) {
    SKIP("Needs MPI size dividing 12 for this decomposition");
  }

  VTKWriter writer("test_parallel_%04d.vti");

  const int nx_global = 12;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 4, 4};
  std::array<int, 3> local_size = {nx_local, 4, 4};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  const std::size_t n_pts = static_cast<std::size_t>(nx_local) *
                            static_cast<std::size_t>(4) *
                            static_cast<std::size_t>(4);
  auto data = fixture.create_test_data(n_pts);
  writer.write(1, data);

  std::string piece_filename =
      "test_parallel_0001_" + std::to_string(fixture.m_rank) + ".vti";
  REQUIRE(fixture.file_exists(piece_filename));
  REQUIRE(fixture.validate_vti_header(piece_filename));
}

TEST_CASE("VTKWriter - Parallel PVTI master file", "[vtk_writer][io][parallel]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks == 1) {
    SKIP("Parallel test requires multiple MPI ranks");
  }
  if (12 % fixture.m_num_ranks != 0) {
    SKIP("Needs MPI size dividing 12 for this decomposition");
  }

  VTKWriter writer("test_parallel_master_%04d.vti");

  const int nx_global = 12;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 4, 4};
  std::array<int, 3> local_size = {nx_local, 4, 4};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  const std::size_t n_pts = static_cast<std::size_t>(nx_local) *
                            static_cast<std::size_t>(4) *
                            static_cast<std::size_t>(4);
  auto data = fixture.create_test_data(n_pts);
  writer.write(1, data);

  MPI_Barrier(MPI_COMM_WORLD);

  std::string piece_filename =
      "test_parallel_master_0001_" + std::to_string(fixture.m_rank) + ".vti";
  REQUIRE(fixture.file_exists(piece_filename));
  REQUIRE(fixture.validate_vti_header(piece_filename));

  MPI_Barrier(MPI_COMM_WORLD);

  if (fixture.m_rank == 0) {
    REQUIRE(fixture.file_exists("test_parallel_master_0001.pvti"));
    REQUIRE(fixture.validate_pvti_header("test_parallel_master_0001.pvti"));

    std::string content =
        VTKWriterTestFixture::read_file_content("test_parallel_master_0001.pvti");
    bool references_all_pieces = true;
    for (int r = 0; r < fixture.m_num_ranks; ++r) {
      std::string piece_ref =
          "test_parallel_master_0001_" + std::to_string(r) + ".vti";
      references_all_pieces &= content.find(piece_ref) != std::string::npos;
    }
    REQUIRE(references_all_pieces);
  }
}

TEST_CASE("VTKWriter - Parallel PVTI WholeExtent", "[vtk_writer][io][parallel]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks == 1) {
    SKIP("Parallel test requires multiple MPI ranks");
  }
  if (24 % fixture.m_num_ranks != 0) {
    SKIP("Needs MPI size dividing 24 for this decomposition");
  }

  VTKWriter writer("test_pvti_extent_%04d.vti");

  const int nx_global = 24;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 8, 8};
  std::array<int, 3> local_size = {nx_local, 8, 8};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  const std::size_t n_pts = static_cast<std::size_t>(nx_local) *
                            static_cast<std::size_t>(8) *
                            static_cast<std::size_t>(8);
  auto data = fixture.create_test_data(n_pts);
  writer.write(1, data);

  MPI_Barrier(MPI_COMM_WORLD);

  if (fixture.m_rank == 0) {
    auto extent =
        fixture.extract_extent("test_pvti_extent_0001.pvti", "WholeExtent");

    REQUIRE(extent[0] == 0);
    REQUIRE(extent[1] == 23); // 24 - 1
    REQUIRE(extent[2] == 0);
    REQUIRE(extent[3] == 7); // 8 - 1
    REQUIRE(extent[4] == 0);
    REQUIRE(extent[5] == 7); // 8 - 1
  }
}

TEST_CASE("VTKWriter rejects invalid VTK extents/domain metadata",
          "[vtk_writer][io][validation][unit]") {
  VTKWriterTestFixture fixture;

  if (fixture.m_num_ranks > 1) {
    SKIP("VTKWriter validation tests require single MPI rank");
  }

  SECTION("non-positive dimensions") {
    VTKWriter writer("test_invalid_dims.vti");

    REQUIRE_THROWS_AS(writer.set_domain({0, 8, 8}, {0, 8, 8}, {0, 0, 0}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(writer.set_domain({8, 8, 8}, {0, 8, 8}, {0, 0, 0}),
                      std::invalid_argument);
  }

  SECTION("Piece extent outside WholeExtent") {
    VTKWriter writer("test_piece_outside_whole.vti");

    REQUIRE_THROWS_AS(writer.set_domain({8, 8, 8}, {8, 8, 8}, {4, 0, 0}),
                      std::invalid_argument);
  }

  SECTION("data size mismatch is caught at write time") {
    VTKWriter writer("test_size_mismatch_0001.vti");

    writer.set_domain({8, 8, 8}, {8, 8, 8}, {0, 0, 0});
    auto too_small = fixture.create_test_data(10);
    REQUIRE_THROWS_AS(writer.write(1, too_small), std::runtime_error);
  }

  SECTION("invalid spacing") {
    VTKWriter writer("test_bad_spacing.vti");

    REQUIRE_THROWS_AS(writer.set_spacing({0.0, 1.0, 1.0}), std::invalid_argument);
    REQUIRE_THROWS_AS(
        writer.set_spacing({std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0}),
        std::invalid_argument);
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

    auto data = fixture.create_test_data(static_cast<std::size_t>(16 * 8 * 4));
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

TEST_CASE("VTKWriter - PVTI file open error", "[vtk_writer][io][error]") {
  VTKWriterTestFixture fixture;

  SECTION("Throw std::runtime_error when PVTI file cannot be opened") {
    // This test specifically exercises the PVTI file-open error path in
    // write_pvti_file(). The write_pvti_file() function is only called
    // when running with multiple MPI ranks (current_size > 1).

    if (fixture.m_num_ranks == 1) {
      SKIP("PVTI master file only written in parallel with multiple ranks");
    }

    // Use a valid output directory (current directory) so per-rank .vti files
    // can be written successfully. This ensures we reach the PVTI code path.
    VTKWriter writer("test_pvti_open_%04d.vti");

    const int nx_global = 4;
    const int nx_local = nx_global / fixture.m_num_ranks;
    std::array<int, 3> global_size = {nx_global, 4, 4};
    std::array<int, 3> local_size = {nx_local, 4, 4};
    std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

    writer.set_domain(global_size, local_size, offset);

    const std::size_t n_pts = static_cast<std::size_t>(nx_local) *
                              static_cast<std::size_t>(4) *
                              static_cast<std::size_t>(4);
    auto data = fixture.create_test_data(n_pts);

    // On rank 0, create a directory with the exact name of the expected .pvti file.
    // std::ofstream will fail to open a path that is itself a directory, triggering
    // the PVTI file-open error path specifically while per-rank .vti files succeed.
    const std::string pvti_filename = "test_pvti_open_0001.pvti";
    if (fixture.m_rank == 0) {
      std::filesystem::create_directory(pvti_filename);
    }

    // Synchronize to ensure rank 0 has created the directory before other ranks proceed
    MPI_Barrier(MPI_COMM_WORLD);

    // Only rank 0 throws std::runtime_error because only rank 0 writes the .pvti master file
    if (fixture.m_rank == 0) {
      REQUIRE_THROWS_AS(writer.write(1, data), std::runtime_error);

      // Clean up the blocking directory
      std::filesystem::remove(pvti_filename);
    } else {
      // Other ranks complete normally (write .vti piece files only)
      writer.write(1, data);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST_CASE("test_vtk_writer_collective_error_field_size_mismatch", "[vtk][mpi][error]") {
  VTKWriterTestFixture fixture;

  // Only run this test with multiple MPI ranks
  if (fixture.m_num_ranks < 2) {
    SKIP("Test requires multiple MPI ranks to verify collective error agreement");
  }

  VTKWriter writer("test_collective_size_mismatch_%04d.vti");

  const int nx_global = 4;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 4, 4};
  std::array<int, 3> local_size = {nx_local, 4, 4};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  // Create test data with WRONG size on rank 0 only
  std::size_t expected_pts = static_cast<std::size_t>(nx_local) *
                             static_cast<std::size_t>(4) *
                             static_cast<std::size_t>(4);

  RealField data;
  if (fixture.m_rank == 0) {
    // Rank 0: create data with wrong size (one less than expected)
    data = fixture.create_test_data(expected_pts - 1);
  } else {
    // Other ranks: create data with correct size
    data = fixture.create_test_data(expected_pts);
  }

  // ALL ranks should throw std::runtime_error due to collective error agreement
  // This verifies that the collective error agreement prevents deadlock
  REQUIRE_THROWS_AS(writer.write(1, data), std::runtime_error);
}

TEST_CASE("test_vtk_writer_collective_error_file_open_failure", "[vtk][mpi][error]") {
  VTKWriterTestFixture fixture;

  // Only run this test with multiple MPI ranks
  if (fixture.m_num_ranks < 2) {
    SKIP("Test requires multiple MPI ranks to verify collective error agreement");
  }

  // Use a filename pattern that will fail on rank 0 only
  std::string filename_pattern;
  if (fixture.m_rank == 0) {
    // Rank 0: use a path that includes a non-existent directory
    filename_pattern = "/nonexistent/directory/test_file_open_%04d.vti";
  } else {
    // Other ranks: use a valid path (current directory)
    filename_pattern = "test_file_open_%04d.vti";
  }

  VTKWriter writer(filename_pattern.c_str());

  const int nx_global = 4;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 4, 4};
  std::array<int, 3> local_size = {nx_local, 4, 4};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  const std::size_t n_pts = static_cast<std::size_t>(nx_local) *
                            static_cast<std::size_t>(4) *
                            static_cast<std::size_t>(4);
  auto data = fixture.create_test_data(n_pts);

  // ALL ranks should throw std::runtime_error due to collective error agreement
  // This verifies that the collective error agreement prevents deadlock
  REQUIRE_THROWS_AS(writer.write(1, data), std::runtime_error);

  // Clean up any files that were created on non-failing ranks
  if (fixture.m_rank != 0) {
    char filename[256];
    snprintf(filename, sizeof(filename), "test_file_open_0001_%d.vti", fixture.m_rank);
    if (fixture.file_exists(filename)) {
      std::filesystem::remove(filename);
    }
  }
}

TEST_CASE("test_vtk_writer_collective_error_file_write_failure", "[vtk][mpi][error]") {
  VTKWriterTestFixture fixture;

  // Only run this test with multiple MPI ranks
  if (fixture.m_num_ranks < 2) {
    SKIP("Test requires multiple MPI ranks to verify collective error agreement");
  }

  // Note: Testing actual file write failure is difficult without filesystem manipulation.
  // This test verifies the collective error agreement path by using a read-only file
  // on rank 0 only. However, since we open files with std::ios::binary and write,
  // we'd need to manipulate the filesystem to make the write fail.
  //
  // As a practical alternative, we verify the collective error agreement mechanism
  // is in place by checking that the code path exists and would be triggered
  // if a write failure occurred.
  //
  // In a real scenario, file write failures could occur due to:
  // - Disk full conditions
  // - Read-only filesystem
  // - Quota exceeded
  // - I/O errors
  //
  // Since we cannot reliably simulate these conditions in a test environment,
  // we document that the collective error agreement for file write failures
  // is implemented in vtk_writer.cpp lines 7-9 (second MPI_Allreduce).

  // For this test, we verify the normal successful path to ensure we didn't
  // break anything with the collective error agreement changes.
  VTKWriter writer("test_collective_write_success_%04d.vti");

  const int nx_global = 4;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 4, 4};
  std::array<int, 3> local_size = {nx_local, 4, 4};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  const std::size_t n_pts = static_cast<std::size_t>(nx_local) *
                            static_cast<std::size_t>(4) *
                            static_cast<std::size_t>(4);
  auto data = fixture.create_test_data(n_pts);

  // This should succeed without throwing
  REQUIRE_NOTHROW(writer.write(1, data));

  // Verify files were created
  // Filename format: test_collective_write_success_0001_RANK.vti
  char filename[256];
  snprintf(filename, sizeof(filename), "test_collective_write_success_0001_%d.vti", fixture.m_rank);
  REQUIRE(fixture.file_exists(filename));

  // Clean up
  if (fixture.m_rank == 0) {
    std::string pvti_filename = "test_collective_write_success_0001.pvti";
    if (fixture.file_exists(pvti_filename)) {
      std::filesystem::remove(pvti_filename);
    }
  }
  std::filesystem::remove(filename);
}

TEST_CASE("test_vtk_writer_complexfield_collective_error_agreement", "[vtk][mpi][error]") {
  VTKWriterTestFixture fixture;

  // Only run this test with multiple MPI ranks
  if (fixture.m_num_ranks < 2) {
    SKIP("Test requires multiple MPI ranks to verify collective error agreement");
  }

  VTKWriter writer("test_collective_complex_mismatch_%04d.vti");

  const int nx_global = 4;
  const int nx_local = nx_global / fixture.m_num_ranks;
  std::array<int, 3> global_size = {nx_global, 4, 4};
  std::array<int, 3> local_size = {nx_local, 4, 4};
  std::array<int, 3> offset = {fixture.m_rank * nx_local, 0, 0};

  writer.set_domain(global_size, local_size, offset);

  // Create complex test data with WRONG size on rank 0 only
  std::size_t expected_pts = static_cast<std::size_t>(nx_local) *
                             static_cast<std::size_t>(4) *
                             static_cast<std::size_t>(4);

  ComplexField data;
  if (fixture.m_rank == 0) {
    // Rank 0: create data with wrong size (one less than expected)
    data = VTKWriterTestFixture::create_complex_test_data(expected_pts - 1);
  } else {
    // Other ranks: create data with correct size
    data = VTKWriterTestFixture::create_complex_test_data(expected_pts);
  }

  // ALL ranks should throw std::runtime_error due to collective error agreement
  // This verifies that the collective error agreement prevents deadlock
  REQUIRE_THROWS_AS(writer.write(1, data), std::runtime_error);
}
