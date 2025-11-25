// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 08_discrete_field.cpp
 * @brief Comprehensive examples of the DiscreteField API
 *
 * This example demonstrates:
 * 1. Creating and initializing discrete fields
 * 2. Array-style indexing and data access
 * 3. Coordinate-space operations and transformations
 * 4. Interpolation at arbitrary coordinates
 * 5. Integration with Model fields and FFT
 *
 * Compile and run:
 *   g++ -std=c++17 -I/path/to/openpfc/include 08_discrete_field.cpp \
 *       -o 08_discrete_field
 *   ./08_discrete_field
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <openpfc/array.hpp>
#include <openpfc/discrete_field.hpp>

using namespace pfc;

//==============================================================================
// Helper functions for output formatting
//==============================================================================

void print_section(const std::string &title) {
  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << title << "\n";
  std::cout << std::string(70, '=') << "\n" << std::endl;
}

template <size_t D>
void print_array(const std::array<double, D> &arr, const std::string &name) {
  std::cout << name << " = [";
  for (size_t i = 0; i < D; i++) {
    std::cout << arr[i];
    if (i < D - 1) std::cout << ", ";
  }
  std::cout << "]\n";
}

template <size_t D>
void print_array_int(const std::array<int, D> &arr, const std::string &name) {
  std::cout << name << " = [";
  for (size_t i = 0; i < D; i++) {
    std::cout << arr[i];
    if (i < D - 1) std::cout << ", ";
  }
  std::cout << "]\n";
}

//==============================================================================
// SCENARIO 1: Creating and Initializing Discrete Fields
//==============================================================================

void demo_creation_and_initialization() {
  print_section("SCENARIO 1: Creating and Initializing Discrete Fields");

  // Create a 3D field with specified geometry
  DiscreteField<double, 3> field({32, 32, 32},    // dimensions: 32³ points
                                 {0, 0, 0},       // offset: no offset (full domain)
                                 {0.0, 0.0, 0.0}, // origin: starts at (0,0,0)
                                 {0.5, 0.5, 0.5}  // discretization: 0.5 spacing
  );

  std::cout << "Created DiscreteField<double, 3>:\n";
  std::cout << field << "\n\n";

  // Initialize with mathematical function - Method 1: 3D lambda
  std::cout << "Method 1: Initialize with 3D function f(x,y,z)\n";
  field.apply([](double x, double y, double z) {
    return std::sin(2.0 * M_PI * x / 16.0) * std::cos(2.0 * M_PI * y / 16.0);
  });

  // Sample some values
  std::cout << "Sample values after initialization:\n";
  std::cout << "  f(0, 0, 0) = " << field[{0, 0, 0}] << "\n";
  std::cout << "  f(4, 0, 0) = " << field[{4, 0, 0}] << "\n";
  std::cout << "  f(0, 4, 0) = " << field[{0, 4, 0}] << "\n";
  std::cout << "  f(4, 4, 0) = " << field[{4, 4, 0}] << "\n\n";

  // Initialize with 1D function - only uses x coordinate
  std::cout << "Method 2: Initialize with 1D function f(x)\n";
  field.apply([](double x) { return std::tanh((x - 8.0) / 2.0); });
  std::cout << "  f(0, any, any) = " << field[{0, 0, 0}] << "\n";
  std::cout << "  f(8, any, any) = " << field[{8, 0, 0}] << " (should be ~0)\n";
  std::cout << "  f(16, any, any) = " << field[{16, 0, 0}] << "\n\n";

  // Initialize with N-D function using std::array
  std::cout << "Method 3: Initialize with N-D function f(std::array<double,3>)\n";
  field.apply([](std::array<double, 3> coords) {
    double r2 =
        coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2];
    return std::exp(-r2 / 100.0);
  });
  std::cout << "  Radial Gaussian centered at origin\n";
  std::cout << "  f(0, 0, 0) = " << field[{0, 0, 0}] << " (maximum)\n";
  std::cout << "  f(10, 0, 0) = " << field[{20, 0, 0}] << " (x=10 physical)\n\n";

  // Create 2D field for comparison
  DiscreteField<double, 2> field2d({64, 64},               // 64² points
                                   {0, 0}, {-32.0, -32.0}, // centered origin
                                   {1.0, 1.0});

  field2d.apply([](double x, double y) {
    double r = std::sqrt(x * x + y * y);
    return std::sin(r) / (r + 0.01); // Sinc-like function
  });

  std::cout << "Created DiscreteField<double, 2> (2D field):\n";
  std::cout << "  Size: 64x64\n";
  std::cout << "  Origin: (-32, -32)\n";
  std::cout << "  f(0, 0) = " << field2d[{32, 32}] << " (center)\n";
  std::cout << "  f(-32, -32) = " << field2d[{0, 0}] << " (corner)\n";
}

//==============================================================================
// SCENARIO 2: Array-Style Indexing and Data Access
//==============================================================================

void demo_indexing() {
  print_section("SCENARIO 2: Array-Style Indexing and Data Access");

  DiscreteField<double, 3> field({16, 16, 16}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                 {1.0, 1.0, 1.0});

  // Initialize with constant
  std::fill(field.get_data().begin(), field.get_data().end(), 1.0);

  std::cout << "Access patterns:\n\n";

  // 1. Multi-dimensional index access
  std::cout << "1. Multi-dimensional indexing: field[{i,j,k}]\n";
  field[{5, 5, 5}] = 10.0;
  field[{10, 10, 10}] = 20.0;
  std::cout << "   Set field[{5,5,5}] = 10.0\n";
  std::cout << "   Set field[{10,10,10}] = 20.0\n";
  std::cout << "   field[{5,5,5}] = " << field[{5, 5, 5}] << "\n";
  std::cout << "   field[{10,10,10}] = " << field[{10, 10, 10}] << "\n\n";

  // 2. Linear index access
  std::cout << "2. Linear indexing: field[idx]\n";
  field[0] = 100.0; // First element
  field[1] = 200.0; // Second element
  std::cout << "   field[0] = " << field[0] << "\n";
  std::cout << "   field[1] = " << field[1] << "\n\n";

  // 3. Direct data access for performance
  std::cout << "3. Direct data access: field.get_data()\n";
  auto &data = field.get_data();
  std::cout << "   Total elements: " << data.size() << "\n";
  std::cout << "   Element type: std::vector<double>\n";

  // Compute statistics using STL algorithms
  double sum = std::accumulate(data.begin(), data.end(), 0.0);
  double mean = sum / data.size();
  double min_val = *std::min_element(data.begin(), data.end());
  double max_val = *std::max_element(data.begin(), data.end());

  std::cout << "   Mean: " << mean << "\n";
  std::cout << "   Min: " << min_val << "\n";
  std::cout << "   Max: " << max_val << "\n\n";

  // 4. Geometry accessors
  std::cout << "4. Geometry information:\n";
  print_array(field.get_origin(), "   Origin");
  print_array(field.get_discretization(), "   Discretization");
  print_array(field.get_coords_low(), "   Bounding box low");
  print_array(field.get_coords_high(), "   Bounding box high");
  print_array_int(field.get_size(), "   Size");
  print_array_int(field.get_offset(), "   Offset");
}

//==============================================================================
// SCENARIO 3: Coordinate-Space Operations
//==============================================================================

void demo_coordinate_operations() {
  print_section("SCENARIO 3: Coordinate-Space Operations");

  DiscreteField<double, 3> field({32, 32, 32}, {0, 0, 0},
                                 {5.0, 5.0, 5.0},  // origin at (5,5,5)
                                 {0.5, 0.5, 0.5}); // 0.5 spacing

  std::cout << "Field geometry:\n";
  std::cout << "  Grid size: 32³\n";
  std::cout << "  Physical origin: (5, 5, 5)\n";
  std::cout << "  Spacing: 0.5\n";
  std::cout << "  Physical domain: [5, 21) x [5, 21) x [5, 21)\n\n";

  // 1. Map indices to coordinates
  std::cout << "1. Index → Coordinate mapping:\n";
  std::vector<std::array<int, 3>> test_indices = {
      {0, 0, 0}, {10, 10, 10}, {31, 31, 31}};

  for (const auto &idx : test_indices) {
    auto coords = field.map_indices_to_coordinates(idx);
    std::cout << "   Index [" << idx[0] << "," << idx[1] << "," << idx[2]
              << "] → Coord [" << coords[0] << "," << coords[1] << "," << coords[2]
              << "]\n";
  }
  std::cout << "\n";

  // 2. Map coordinates to indices
  std::cout << "2. Coordinate → Index mapping (nearest):\n";
  std::vector<std::array<double, 3>> test_coords = {
      {5.0, 5.0, 5.0},   // Exactly at grid point
      {10.3, 12.7, 8.1}, // Between grid points
      {20.9, 20.9, 20.9} // Near boundary
  };

  for (const auto &coord : test_coords) {
    auto idx = field.map_coordinates_to_indices(coord);
    std::cout << "   Coord [" << coord[0] << "," << coord[1] << "," << coord[2]
              << "] → Index [" << idx[0] << "," << idx[1] << "," << idx[2] << "]\n";
  }
  std::cout << "\n";

  // 3. Bounds checking
  std::cout << "3. Bounds checking:\n";
  std::vector<std::array<double, 3>> test_bounds = {
      {10.0, 10.0, 10.0}, // Inside
      {5.0, 5.0, 5.0},    // Lower boundary (inclusive)
      {21.0, 10.0, 10.0}, // Upper boundary (exclusive)
      {4.9, 10.0, 10.0},  // Outside (below)
      {25.0, 10.0, 10.0}  // Outside (above)
  };

  for (const auto &coord : test_bounds) {
    bool in = field.inbounds(coord);
    std::cout << "   [" << coord[0] << "," << coord[1] << "," << coord[2] << "] → "
              << (in ? "INSIDE" : "OUTSIDE") << "\n";
  }
}

//==============================================================================
// SCENARIO 4: Interpolation
//==============================================================================

void demo_interpolation() {
  print_section("SCENARIO 4: Interpolation at Arbitrary Coordinates");

  // Create field with known analytical function
  DiscreteField<double, 3> field({64, 64, 64}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                 {1.0, 1.0, 1.0});

  // Initialize: f(x,y,z) = x² + y² + z²
  field.apply([](double x, double y, double z) { return x * x + y * y + z * z; });

  std::cout << "Field function: f(x,y,z) = x² + y² + z²\n";
  std::cout << "Interpolation method: Nearest-neighbor\n\n";

  // Test interpolation at various points
  std::cout << "Interpolation tests:\n";
  std::cout << "Query Coord\t\tInterpolated\tExpected\tError\n";
  std::cout << std::string(65, '-') << "\n";

  std::vector<std::array<double, 3>> query_points = {
      {10.0, 10.0, 10.0}, // Exactly on grid
      {10.3, 10.0, 10.0}, // Slightly off (→ 10)
      {10.5, 10.0, 10.0}, // Midpoint (→ 10 or 11)
      {10.7, 10.0, 10.0}, // Slightly off (→ 11)
      {5.2, 8.7, 12.1},   // Arbitrary point
      {20.0, 20.0, 20.0}, // On grid
      {31.8, 31.9, 31.7}  // Near boundary
  };

  for (const auto &query : query_points) {
    if (field.inbounds(query)) {
      double interp_val =
          pfc::interpolate(field, query); // Free function (preferred)
      double exact_val =
          query[0] * query[0] + query[1] * query[1] + query[2] * query[2];
      double error = std::abs(interp_val - exact_val);

      printf("[%.1f,%.1f,%.1f]\t%.2f\t\t%.2f\t\t%.2f\n", query[0], query[1],
             query[2], interp_val, exact_val, error);
    }
  }

  std::cout << "\nNote: Nearest-neighbor error depends on function curvature\n";
  std::cout << "      Error is typically < 0.5 * dx² * |∇²f| for smooth functions\n";

  // Example: Safe interpolation with bounds checking
  std::cout << "\nSafe interpolation pattern:\n";
  std::array<double, 3> test_point = {5.5, 10.2, 15.7};
  std::cout << "Query: [" << test_point[0] << "," << test_point[1] << ","
            << test_point[2] << "]\n";

  if (field.inbounds(test_point)) {
    double value = pfc::interpolate(field, test_point); // Free function (preferred)
    std::cout << "Result: " << value << " (IN BOUNDS)\n";
  } else {
    std::cout << "Result: OUT OF BOUNDS - cannot interpolate\n";
  }
}

//==============================================================================
// SCENARIO 5: Complex Fields and FFT Integration
//==============================================================================

void demo_complex_fields() {
  print_section("SCENARIO 5: Complex Fields and FFT Integration");

  using Complex = std::complex<double>;

  // Real-space field
  DiscreteField<double, 3> real_field({64, 64, 64}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                      {1.0, 1.0, 1.0});

  // Complex k-space field (after real-to-complex FFT)
  // Size is (nx, ny, nz/2+1) for real-to-complex transform
  DiscreteField<Complex, 3> kspace_field({64, 64, 33}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                         {1.0, 1.0, 1.0});

  std::cout << "Real-space field: 64 x 64 x 64 = " << real_field.get_data().size()
            << " points\n";
  std::cout << "K-space field: 64 x 64 x 33 = " << kspace_field.get_data().size()
            << " complex points\n";
  std::cout << "(33 = 64/2 + 1, due to Hermitian symmetry)\n\n";

  // Initialize real field with sine wave
  real_field.apply(
      [](double x, double y, double z) { return std::sin(2.0 * M_PI * x / 64.0); });

  // Simulate k-space operation: low-pass filter
  kspace_field.apply([](double kx, double ky, double kz) {
    double k2 = kx * kx + ky * ky + kz * kz;
    double cutoff = 10.0;
    if (k2 < cutoff * cutoff) {
      return Complex(1.0, 0.0); // Pass low frequencies
    } else {
      return Complex(0.0, 0.0); // Filter high frequencies
    }
  });

  std::cout << "Initialized real-space field with sine wave\n";
  std::cout << "Initialized k-space field with low-pass filter\n\n";

  // Demonstrate complex field operations
  std::cout << "Complex field operations:\n";
  std::cout << "  kspace_field[{0,0,0}] = " << kspace_field[{0, 0, 0}] << "\n";
  std::cout << "  kspace_field[{1,0,0}] = " << kspace_field[{1, 0, 0}] << "\n";
  std::cout << "  kspace_field[{10,0,0}] = " << kspace_field[{10, 0, 0}] << "\n\n";

  // Count non-zero k-space modes
  int non_zero = 0;
  for (const auto &val : kspace_field.get_data()) {
    if (std::abs(val) > 1e-10) non_zero++;
  }
  std::cout << "Non-zero k-space modes: " << non_zero << " / "
            << kspace_field.get_data().size() << "\n";
}

//==============================================================================
// Main: Run all scenarios
//==============================================================================

int main() {
  std::cout << "\n";
  std::cout
      << "╔════════════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║          OpenPFC DiscreteField API Examples                        ║\n";
  std::cout
      << "║                                                                    ║\n";
  std::cout
      << "║  Demonstrates discrete fields with coordinate mapping              ║\n";
  std::cout
      << "╚════════════════════════════════════════════════════════════════════╝\n";

  try {
    demo_creation_and_initialization();
    demo_indexing();
    demo_coordinate_operations();
    demo_interpolation();
    demo_complex_fields();

    std::cout << "\n";
    std::cout << "╔═════════════════════════════════════════════════════════════════"
                 "═══╗\n";
    std::cout << "║  Key Takeaways:                                                 "
                 "   ║\n";
    std::cout << "║                                                                 "
                 "   ║\n";
    std::cout
        << "║  1. DiscreteField bridges discrete grids and physical space       ║\n";
    std::cout << "║  2. Multiple initialization methods: apply() with lambdas       "
                 "   ║\n";
    std::cout << "║  3. Flexible indexing: multi-dimensional or linear              "
                 "   ║\n";
    std::cout << "║  4. Coordinate transformations: indices ↔ physical coords       "
                 "   ║\n";
    std::cout << "║  5. Interpolation: nearest-neighbor (check bounds first!)       "
                 "   ║\n";
    std::cout << "║  6. Works with real and complex fields (FFT integration)        "
                 "   ║\n";
    std::cout << "║                                                                 "
                 "   ║\n";
    std::cout << "║  Performance: Direct data access via get_data() for hot paths   "
                 "   ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════════"
                 "═══╝\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
