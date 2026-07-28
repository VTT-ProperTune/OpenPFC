// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file 08_discrete_field.cpp
 * @brief Comprehensive examples of the pfc::data::Field<T> API
 *
 * This example demonstrates the `pfc::data::Field<T>` class that provides
 * efficient storage and access for grid-based numerical fields with domain
 * geometry awareness.
 *
 * Demonstrated features:
 * 1. Creating and initializing fields with domain geometry
 * 2. Array-style indexing and data access
 * 3. Coordinate-space operations and transformations
 * 4. Complex fields and FFT integration
 * 5. Field iteration and STL algorithm compatibility
 *
 * Compile and run:
 *   g++ -std=c++20 -I/path/to/openpfc/include 08_discrete_field.cpp \
 *       -o 08_discrete_field
 *   ./08_discrete_field
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <numeric>
#include <openpfc/kernel/data/box3i.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/grid_field.hpp>

using namespace pfc;
using namespace pfc::data;

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
// SCENARIO 1: Creating and Initializing Fields
//==============================================================================

void demo_creation_and_initialization() {
  print_section("SCENARIO 1: Creating and Initializing Fields");

  // Create a domain and a 3D field covering the full domain
  auto domain = domain::create({32, 32, 32}, {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5});
  Field<double> field(domain, Box3i::from_bounds({0, 0, 0}, {31, 31, 31}), 0);

  std::cout << "Created Field<double>:\n";
  std::cout << "  Local size: [" << field.local_size()[0] << ", " << field.local_size()[1] << ", "
            << field.local_size()[2] << "]\n";
  std::cout << "  Origin: [" << domain.get_origin()[0] << ", " << domain.get_origin()[1] << ", "
            << domain.get_origin()[2] << "]\n";
  std::cout << "  Spacing: [" << domain.get_spacing()[0] << ", " << domain.get_spacing()[1] << ", "
            << domain.get_spacing()[2] << "]\n\n";

  // Initialize with mathematical function - Method 1: 3D lambda
  std::cout << "Method 1: Initialize with 3D function f(x,y,z)\n";
  field.apply([](double x, double y, double /*z*/) {
    return std::sin(2.0 * std::numbers::pi * x / 16.0) *
           std::cos(2.0 * std::numbers::pi * y / 16.0);
  });

  // Sample some values
  std::cout << "Sample values after initialization:\n";
  std::cout << "  f(0, 0, 0) = " << field(0, 0, 0) << "\n";
  std::cout << "  f(4, 0, 0) = " << field(4, 0, 0) << "\n";
  std::cout << "  f(0, 4, 0) = " << field(0, 4, 0) << "\n";
  std::cout << "  f(4, 4, 0) = " << field(4, 4, 0) << "\n\n";

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

  // Create 2D field for comparison (using 3D field with z-size=1)
  auto domain2d = domain::create({64, 64, 1}, {-32.0, -32.0, 0.0}, {1.0, 1.0, 1.0});
  Field<double> field2d(domain2d, Box3i::from_bounds({0, 0, 0}, {63, 63, 0}), 0);

  field2d.apply([](double x, double y, double /*z*/) {
    double r = std::sqrt(x * x + y * y);
    return std::sin(r) / (r + 0.01); // Sinc-like function
  });

  std::cout << "Created Field<double> (2D field in 3D, z-size=1):\n";
  std::cout << "  Size: 64x64x1\n";
  std::cout << "  Origin: (-32, -32, 0)\n";
  std::cout << "  f(0, 0, 0) = " << field2d(32, 32, 0) << " (center)\n";
  std::cout << "  f(-32, -32, 0) = " << field2d(0, 0, 0) << " (corner)\n";
}

//==============================================================================
// SCENARIO 2: Array-Style Indexing and Data Access
//==============================================================================

void demo_indexing() {
  print_section("SCENARIO 2: Array-Style Indexing and Data Access");

  auto domain = domain::create({16, 16, 16}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  Field<double> field(domain, Box3i::from_bounds({0, 0, 0}, {15, 15, 15}), 0);

  // Initialize with constant
  std::fill(field.data(), field.data() + field.size(), 1.0);

  std::cout << "Access patterns:\n\n";

  // 1. Multi-dimensional index access
  std::cout << "1. Multi-dimensional indexing: field(i, j, k)\n";
  field(5, 5, 5) = 10.0;
  field(10, 10, 10) = 20.0;
  std::cout << "   Set field(5,5,5) = 10.0\n";
  std::cout << "   Set field(10,10,10) = 20.0\n";
  std::cout << "   field(5,5,5) = " << field(5, 5, 5) << "\n";
  std::cout << "   field(10,10,10) = " << field(10, 10, 10) << "\n\n";

  // 2. Linear index access for reading
  std::cout << "2. Direct data access: field.data()\n";
  std::cout << "   Total elements: " << field.size() << "\n";
  std::cout << "   Element type: DataBuffer<T>*\n";

  // Compute statistics using STL algorithms
  auto *data = field.data();
  double sum = std::accumulate(data, data + field.size(), 0.0);
  double mean = sum / field.size();
  double min_val = *std::min_element(data, data + field.size());
  double max_val = *std::max_element(data, data + field.size());

  std::cout << "   Mean: " << mean << "\n";
  std::cout << "   Min: " << min_val << "\n";
  std::cout << "   Max: " << max_val << "\n\n";

  // 3. Geometry accessors
  std::cout << "3. Geometry information:\n";
  print_array(field.origin(), "   Origin");
  print_array(field.spacing(), "   Spacing");
  std::cout << "   Owned box: [" << field.box().low[0] << "," << field.box().low[1] << ","
            << field.box().low[2] << "] to [" << field.box().high[0] << ","
            << field.box().high[1] << "," << field.box().high[2] << "]\n";
  print_array_int(field.local_size(), "   Local size");
  std::cout << "   Domain size: [" << domain.get_size()[0] << "," << domain.get_size()[1] << ","
            << domain.get_size()[2] << "]\n";
}

//==============================================================================
// SCENARIO 3: Coordinate-Space Operations
//==============================================================================

void demo_coordinate_operations() {
  print_section("SCENARIO 3: Coordinate-Space Operations");

  auto domain = domain::create({32, 32, 32}, {5.0, 5.0, 5.0}, {0.5, 0.5, 0.5});
  Field<double> field(domain, Box3i::from_bounds({0, 0, 0}, {31, 31, 31}), 0);

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
    auto coords = field.coords(idx[0], idx[1], idx[2]);
    std::cout << "   Index [" << idx[0] << "," << idx[1] << "," << idx[2]
              << "] → Coord [" << coords[0] << "," << coords[1] << "," << coords[2]
              << "]\n";
  }
  std::cout << "\n";

  // 2. Map coordinates to indices (nearest grid point)
  std::cout << "2. Coordinate → Index mapping (nearest):\n";
  std::vector<std::array<double, 3>> test_coords = {
      {5.0, 5.0, 5.0},   // Exactly at grid point
      {10.3, 12.7, 8.1}, // Between grid points
      {20.9, 20.9, 20.9} // Near boundary
  };

  for (const auto &coord : test_coords) {
    int i = static_cast<int>((coord[0] - domain.get_origin()[0]) / domain.get_spacing()[0]);
    int j = static_cast<int>((coord[1] - domain.get_origin()[1]) / domain.get_spacing()[1]);
    int k = static_cast<int>((coord[2] - domain.get_origin()[2]) / domain.get_spacing()[2]);
    std::cout << "   Coord [" << coord[0] << "," << coord[1] << "," << coord[2]
              << "] → Index [" << i << "," << j << "," << k << "]\n";
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
    // Check if coordinate is within physical bounds
    bool in = (coord[0] >= domain.get_origin()[0] &&
               coord[0] < domain.get_origin()[0] + domain.get_size()[0] * domain.get_spacing()[0] &&
               coord[1] >= domain.get_origin()[1] &&
               coord[1] < domain.get_origin()[1] + domain.get_size()[1] * domain.get_spacing()[1] &&
               coord[2] >= domain.get_origin()[2] &&
               coord[2] < domain.get_origin()[2] + domain.get_size()[2] * domain.get_spacing()[2]);
    std::cout << "   [" << coord[0] << "," << coord[1] << "," << coord[2] << "] → "
              << (in ? "INSIDE" : "OUTSIDE") << "\n";
  }
}

//==============================================================================
// SCENARIO 4: Direct Access at Grid Points
//==============================================================================

void demo_interpolation() {
  print_section("SCENARIO 4: Direct Access at Grid Points");

  // Create field with known analytical function
  auto domain = domain::create({64, 64, 64}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  Field<double> field(domain, Box3i::from_bounds({0, 0, 0}, {63, 63, 63}), 0);

  // Initialize: f(x,y,z) = x² + y² + z²
  field.apply([](double x, double y, double z) { return x * x + y * y + z * z; });

  std::cout << "Field function: f(x,y,z) = x² + y² + z²\n";
  std::cout << "Access method: Direct lookup at nearest grid point\n\n";

  // Test access at various grid points
  std::cout << "Grid point access tests:\n";
  std::cout << "Grid Index\t\tPhysical Coord\t\tValue\t\tExpected\tError\n";
  std::cout << std::string(75, '-') << "\n";

  std::vector<std::array<int, 3>> grid_points = {
      {10, 10, 10}, {10, 5, 0}, {20, 20, 20}, {0, 0, 0}, {63, 63, 63}
  };

  for (const auto &idx : grid_points) {
    auto coords = field.coords(idx[0], idx[1], idx[2]);
    double value = field(idx[0], idx[1], idx[2]);
    double exact_val = coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2];
    double error = std::abs(value - exact_val);

    printf("[%d,%d,%d]\t\t[%.1f,%.1f,%.1f]\t\t%.2f\t\t%.2f\t\t%.2f\n",
           idx[0], idx[1], idx[2], coords[0], coords[1], coords[2],
           value, exact_val, error);
  }

  std::cout << "\nNote: The Field API provides direct access to grid-point values.\n";
  std::cout << "      For interpolation between grid points, use appropriate\n";
  std::cout << "      interpolation libraries or implement interpolation manually\n";
  std::cout << "      using the grid-point values and coordinate transformations.\n";

  // Example: Safe access with bounds checking
  std::cout << "\nSafe access pattern:\n";
  std::array<int, 3> test_index = {10, 20, 30};
  std::cout << "Grid index: [" << test_index[0] << "," << test_index[1] << ","
            << test_index[2] << "]\n";

  if (test_index[0] >= 0 && test_index[0] < field.local_size()[0] &&
      test_index[1] >= 0 && test_index[1] < field.local_size()[1] &&
      test_index[2] >= 0 && test_index[2] < field.local_size()[2]) {
    auto coords = field.coords(test_index[0], test_index[1], test_index[2]);
    double value = field(test_index[0], test_index[1], test_index[2]);
    std::cout << "Physical coord: [" << coords[0] << "," << coords[1] << "," << coords[2] << "]\n";
    std::cout << "Value: " << value << " (VALID INDEX)\n";
  } else {
    std::cout << "Result: OUT OF BOUNDS\n";
  }
}

//==============================================================================
// SCENARIO 5: Complex Fields and FFT Integration
//==============================================================================

void demo_complex_fields() {
  print_section("SCENARIO 5: Complex Fields and FFT Integration");

  using Complex = std::complex<double>;

  // Real-space field
  auto domain_real = domain::create({64, 64, 64}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  Field<double> real_field(domain_real, Box3i::from_bounds({0, 0, 0}, {63, 63, 63}), 0);

  // Complex k-space field (after real-to-complex FFT)
  // Size is (nx, ny, nz/2+1) for real-to-complex transform
  auto domain_k = domain::create({64, 64, 33}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  Field<Complex> kspace_field(domain_k, Box3i::from_bounds({0, 0, 0}, {63, 63, 32}), 0);

  std::cout << "Real-space field: 64 x 64 x 64 = " << real_field.size()
            << " points\n";
  std::cout << "K-space field: 64 x 64 x 33 = " << kspace_field.size()
            << " complex points\n";
  std::cout << "(33 = 64/2 + 1, due to Hermitian symmetry)\n\n";

  // Initialize real field with sine wave
  real_field.apply([](double x, double /*y*/, double /*z*/) {
    return std::sin(2.0 * std::numbers::pi * x / 64.0);
  });

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
  std::cout << "  kspace_field(0,0,0) = " << kspace_field(0, 0, 0) << "\n";
  std::cout << "  kspace_field(1,0,0) = " << kspace_field(1, 0, 0) << "\n";
  std::cout << "  kspace_field(10,0,0) = " << kspace_field(10, 0, 0) << "\n\n";

  // Count non-zero k-space modes
  int non_zero = 0;
  for (size_t i = 0; i < kspace_field.size(); ++i) {
    if (std::abs(kspace_field.data()[i]) > 1e-10) non_zero++;
  }
  std::cout << "Non-zero k-space modes: " << non_zero << " / "
            << kspace_field.size() << "\n";
}

//==============================================================================
// Main: Run all scenarios
//==============================================================================

int main() {
  std::cout << "\n";
  std::cout
      << "╔════════════════════════════════════════════════════════════════════╗\n";
  std::cout
      << "║          OpenPFC Field API Examples                                ║\n";
  std::cout
      << "║                                                                    ║\n";
  std::cout
      << "║  Demonstrates pfc::data::Field<T> for efficient grid-based         ║\n";
  std::cout
      << "║  numerical fields with domain geometry awareness                   ║\n";
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
        << "║  1. Field<T> bridges discrete grids and physical space             ║\n";
    std::cout << "║  2. Multiple initialization methods: apply() with lambdas       "
                 "   ║\n";
    std::cout << "║  3. Flexible indexing: multi-dimensional (i,j,k) or linear data()  "
                 "   ║\n";
    std::cout << "║  4. Coordinate transformations: indices ↔ physical coords       "
                 "   ║\n";
    std::cout << "║  5. Direct grid-point access (implement interpolation separately)  "
                 "   ║\n";
    std::cout << "║  6. Works with real and complex fields (FFT integration)        "
                 "   ║\n";
    std::cout << "║  7. Domain-aware with Box3i for decomposition support          "
                 "   ║\n";
    std::cout << "║                                                                 "
                 "   ║\n";
    std::cout << "║  Performance: Direct data access via data() for hot paths       "
                 "   ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════════"
                 "═══╝\n";

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
