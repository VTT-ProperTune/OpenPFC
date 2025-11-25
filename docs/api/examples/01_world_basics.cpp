// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @example 01_world_basics.cpp
 * @brief Demonstrates basic World API usage
 *
 * This example shows:
 * - Creating World objects with different construction patterns
 * - Querying world properties (size, spacing, origin)
 * - Converting between grid indices and physical coordinates
 * - Working with 1D, 2D, and 3D domains
 * - Common usage patterns for simulation setup
 *
 * Expected output:
 * - Grid dimensions and properties
 * - Coordinate transformation examples
 * - Memory footprint calculations
 *
 * Time to run: < 1 second
 */

#include <iomanip>
#include <iostream>
#include <openpfc/core/world.hpp>
#include <vector>

using namespace pfc;

void print_section(const std::string &title) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "  " << title << "\n";
  std::cout << std::string(60, '=') << "\n";
}

void example_basic_creation() {
  print_section("Example 1: Basic World Creation");

  // Most common pattern: uniform grid with default spacing
  // Note: world::create() makes it clear we're creating a World object
  auto world1 = world::create({100, 100, 100});

  std::cout << "Created 100³ grid with default spacing\n";
  auto size = world::get_size(world1);
  std::cout << "  Size: " << size[0] << " × " << size[1] << " × " << size[2] << "\n";
  std::cout << "  Total grid points: " << world::get_total_size(world1) << "\n";

  // Memory footprint calculation
  size_t bytes = world::get_total_size(world1) * sizeof(double);
  std::cout << "  Memory for double field: " << bytes / (1024 * 1024) << " MB\n";
}

void example_custom_spacing() {
  print_section("Example 2: Custom Spacing and Origin");

  // Create domain with specific spacing and offset
  Real3 origin = {0.0, 0.0, 0.0};
  Real3 spacing = {0.1, 0.1, 0.1};
  auto world = world::create({256, 256, 256}, origin, spacing);

  std::cout << "Created 256³ grid:\n";
  std::cout << "  Spacing: " << std::fixed << std::setprecision(3);
  auto dx = world::get_spacing(world);
  std::cout << dx[0] << " × " << dx[1] << " × " << dx[2] << "\n";

  std::cout << "  Origin: ";
  auto orig = world::get_origin(world);
  std::cout << orig[0] << ", " << orig[1] << ", " << orig[2] << "\n";

  // Physical domain size
  double vol = world::physical_volume(world);
  std::cout << "  Physical volume: " << vol << " cubic units\n";

  // Domain bounds
  auto lower = get_lower_bounds(world);
  auto upper = get_upper_bounds(world);
  std::cout << "  Physical bounds: [" << lower[0] << ", " << upper[0] << "] × ";
  std::cout << "[" << lower[1] << ", " << upper[1] << "] × ";
  std::cout << "[" << lower[2] << ", " << upper[2] << "]\n";
}

void example_coordinate_transforms() {
  print_section("Example 3: Coordinate Transformations");

  auto world = world::create({100, 100, 100}, {0, 0, 0}, {0.1, 0.1, 0.1});

  std::cout << "Grid: 100³ with spacing 0.1\n\n";

  // Grid indices to physical coordinates
  std::cout << "Grid → Physical transformations:\n";
  Int3 idx1 = {0, 0, 0};
  Real3 pos1 = world::to_coords(world, idx1);
  std::cout << "  Index " << idx1[0] << "," << idx1[1] << "," << idx1[2];
  std::cout << " → Position " << pos1[0] << "," << pos1[1] << "," << pos1[2] << "\n";

  Int3 idx2 = {50, 50, 50};
  Real3 pos2 = world::to_coords(world, idx2);
  std::cout << "  Index " << idx2[0] << "," << idx2[1] << "," << idx2[2];
  std::cout << " → Position " << pos2[0] << "," << pos2[1] << "," << pos2[2] << "\n";

  Int3 idx3 = {99, 99, 99};
  Real3 pos3 = world::to_coords(world, idx3);
  std::cout << "  Index " << idx3[0] << "," << idx3[1] << "," << idx3[2];
  std::cout << " → Position " << pos3[0] << "," << pos3[1] << "," << pos3[2] << "\n";

  // Physical coordinates to grid indices
  std::cout << "\nPhysical → Grid transformations (nearest neighbor):\n";
  Real3 pos4 = {2.53, 4.78, 7.21};
  Int3 idx4 = world::to_indices(world, pos4);
  std::cout << "  Position " << pos4[0] << "," << pos4[1] << "," << pos4[2];
  std::cout << " → Index " << idx4[0] << "," << idx4[1] << "," << idx4[2];
  std::cout << " (rounded)\n";

  // Verify round-trip transformation
  Real3 pos5 = world::to_coords(world, idx4);
  std::cout << "  Snapped to grid: " << pos5[0] << "," << pos5[1] << "," << pos5[2]
            << "\n";
}

void example_dimensionality() {
  print_section("Example 4: 1D, 2D, and 3D Domains");

  // 1D domain (only x-direction)
  auto world1d = world::create({1000, 1, 1});
  std::cout << "1D domain (1000 × 1 × 1):\n";
  std::cout << "  is_1d: " << (world::is_1d(world1d) ? "true" : "false") << "\n";
  std::cout << "  dimensionality: " << world::dimensionality(world1d) << "\n";
  std::cout << "  Total points: " << world::get_total_size(world1d) << "\n";

  // 2D domain (x and y directions)
  auto world2d = world::create({256, 256, 1});
  std::cout << "\n2D domain (256 × 256 × 1):\n";
  std::cout << "  is_2d: " << (world::is_2d(world2d) ? "true" : "false") << "\n";
  std::cout << "  dimensionality: " << world::dimensionality(world2d) << "\n";
  std::cout << "  Total points: " << world::get_total_size(world2d) << "\n";

  // 3D domain (all directions)
  auto world3d = world::create({64, 64, 64});
  std::cout << "\n3D domain (64 × 64 × 64):\n";
  std::cout << "  is_3d: " << (world::is_3d(world3d) ? "true" : "false") << "\n";
  std::cout << "  dimensionality: " << world::dimensionality(world3d) << "\n";
  std::cout << "  Total points: " << world::get_total_size(world3d) << "\n";
}

void example_centered_domain() {
  print_section("Example 5: Centered Domain");

  // Domain centered at origin: spans [-5, 5] in each direction
  int n = 100;
  double L = 10.0; // Physical domain size
  double dx = L / n;
  double offset = -L / 2.0;

  auto world = world::create({n, n, n}, {offset, offset, offset}, {dx, dx, dx});

  std::cout << "100³ grid centered at origin:\n";
  auto lower = world::get_lower_bounds(world);
  auto upper = world::get_upper_bounds(world);
  std::cout << "  Lower bounds: " << lower[0] << ", " << lower[1] << ", " << lower[2]
            << "\n";
  std::cout << "  Upper bounds: " << upper[0] << ", " << upper[1] << ", " << upper[2]
            << "\n";

  // Center point
  Real3 center = world::to_coords(world, {n / 2, n / 2, n / 2});
  std::cout << "  Center position: " << center[0] << ", " << center[1] << ", "
            << center[2] << "\n";
}

void example_fft_optimal_sizes() {
  print_section("Example 6: FFT-Optimal Grid Sizes");

  std::cout << "FFT performs best with grid sizes that are:\n";
  std::cout << "  - Powers of 2 (fastest): 64, 128, 256, 512, 1024...\n";
  std::cout << "  - Products of small primes: 60 = 2²×3×5, 96 = 2⁵×3\n";
  std::cout << "\nRecommended sizes:\n";

  std::vector<int> optimal_sizes = {64, 96, 128, 192, 256, 384, 512, 768, 1024};
  for (int size : optimal_sizes) {
    auto world = world::create({size, size, size});
    size_t points = world::get_total_size(world);
    size_t mb = (points * sizeof(double)) / (1024 * 1024);
    std::cout << "  " << std::setw(4) << size << "³ → " << std::setw(12) << points
              << " points → " << std::setw(6) << mb << " MB per field\n";
  }
}

void example_anisotropic_grid() {
  print_section("Example 7: Anisotropic (Non-Uniform) Grid");

  // Different resolution in each direction
  // Example: thin film with fine z-resolution
  auto world = world::create({128, 128, 512},              // More points in z
                             {0, 0, 0}, {1.0, 1.0, 0.25}); // Finer spacing in z

  std::cout << "Thin film geometry (128 × 128 × 512):\n";
  auto size = world::get_size(world);
  auto spacing = world::get_spacing(world);

  std::cout << "  Grid dimensions: " << size[0] << " × " << size[1] << " × "
            << size[2] << "\n";
  std::cout << "  Spacing: " << spacing[0] << " × " << spacing[1] << " × "
            << spacing[2] << "\n";

  auto lower = world::get_lower_bounds(world);
  auto upper = world::get_upper_bounds(world);
  std::cout << "  Physical size: ";
  std::cout << (upper[0] - lower[0]) << " × ";
  std::cout << (upper[1] - lower[1]) << " × ";
  std::cout << (upper[2] - lower[2]) << " (x × y × z)\n";

  std::cout << "\nAspect ratios:\n";
  std::cout << "  Lx / Lz = " << (upper[0] - lower[0]) / (upper[2] - lower[2])
            << "\n";
  std::cout << "  nx / nz = " << static_cast<double>(size[0]) / size[2] << "\n";
}

int main() {
  std::cout << "OpenPFC World API Examples\n";
  std::cout << "==========================\n";
  std::cout << "\nThis example demonstrates the World class API for defining\n";
  std::cout << "computational domains in phase-field simulations.\n";

  try {
    example_basic_creation();
    example_custom_spacing();
    example_coordinate_transforms();
    example_dimensionality();
    example_centered_domain();
    example_fft_optimal_sizes();
    example_anisotropic_grid();

    print_section("Summary");
    std::cout << "\nKey takeaways:\n";
    std::cout << "  ✓ World defines the global simulation domain\n";
    std::cout << "  ✓ Use create() with size, origin, and spacing\n";
    std::cout << "  ✓ to_coords() and to_indices() for coordinate transforms\n";
    std::cout << "  ✓ Powers of 2 are optimal for FFT performance\n";
    std::cout << "  ✓ Works seamlessly with 1D, 2D, and 3D domains\n";
    std::cout << "\nSee include/openpfc/core/world.hpp for complete API.\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
