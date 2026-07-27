// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/grid_field.hpp>
#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/domain/create.hpp>
#include <openpfc/kernel/decomposition/decomposition_factory.hpp>
#include <openpfc/kernel/field/field_factory.hpp>

using namespace pfc;
using Catch::Approx;

TEST_CASE("field_from_subdomain: slice access patterns (Domain version)",
          "[field_factory][domain][unit]") {
  const int nx = 8, ny = 6, nz = 4;
  auto world = domain::create_world({nx, ny, nz});
  auto decomp = decomposition::create(world, 1);
  
  auto field = pfc::data::field_from_subdomain<double>(decomp, 0, 1);
  
  // Initialize field with a simple pattern
  field.apply([](double x, double y, double z) { return x + 2.0 * y + 3.0 * z; });
  
  // Test slice extraction by fixing one dimension
  SECTION("XY plane slice at fixed z") {
    const int k = 2;
    double slice_sum = 0.0;
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        slice_sum += field(i, j, k);
      }
    }
    
    // Verify slice values match expected pattern
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        double expected = i + 2.0 * j + 3.0 * k;
        REQUIRE(field(i, j, k) == Approx(expected));
      }
    }
  }
  
  SECTION("XZ plane slice at fixed y") {
    const int j = 3;
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i < nx; ++i) {
        double expected = i + 2.0 * j + 3.0 * k;
        REQUIRE(field(i, j, k) == Approx(expected));
      }
    }
  }
  
  SECTION("YZ plane slice at fixed x") {
    const int i = 4;
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        double expected = i + 2.0 * j + 3.0 * k;
        REQUIRE(field(i, j, k) == Approx(expected));
      }
    }
  }
}

TEST_CASE("field_from_subdomain: slice boundary access (Domain version)",
          "[field_factory][domain][unit]") {
  const int nx = 6, ny = 6, nz = 6;
  auto world = domain::create_world({nx, ny, nz});
  auto decomp = decomposition::create(world, 1);
  
  auto field = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  
  // Initialize field coordinate-based pattern
  field.apply([](double x, double y, double z) { return x * y * z; });
  
  // Test boundary slices
  SECTION("z=0 plane slice") {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        REQUIRE(field(i, j, 0) == Approx(0.0)); // z=0 makes product zero
      }
    }
  }
  
  SECTION("x=0 plane slice") {
    for (int k = 0; k < nz; ++k) {
      for (int j = 0; j < ny; ++j) {
        REQUIRE(field(0, j, k) == Approx(0.0)); // x=0 makes product zero
      }
    }
  }
  
  SECTION("y=0 plane slice") {
    for (int k = 0; k < nz; ++k) {
      for (int i = 0; i < nx; ++i) {
        REQUIRE(field(i, 0, k) == Approx(0.0)); // y=0 makes product zero
      }
    }
  }
}

TEST_CASE("field_from_subdomain: mid-plane slice values (Domain version)",
          "[field_factory][domain][unit]") {
  const int nx = 8, ny = 8, nz = 8;
  auto world = domain::create_world(GridSize({nx, ny, nz}),
                                    PhysicalOrigin({0.0, 0.0, 0.0}),
                                    GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = decomposition::create(world, 1);

  auto field = pfc::data::field_from_subdomain<double>(decomp, 0, 0);

  // Initialize with radial pattern centered at origin
  field.apply([](double x, double y, double z) {
    return std::sqrt(x*x + y*y + z*z);
  });

  // Test that origin (0,0,0) has value 0.0
  REQUIRE(field(0, 0, 0) == Approx(0.0)); // At origin

  // Test mid-plane values (simple linear pattern for easier verification)
  field.apply([](double x, double y, double z) {
    return x + 2.0 * y + 3.0 * z; // Simple linear function
  });

  // Verify mid-plane values match expected pattern
  const int mid = 4;
  double expected_mid = 0.0 + 2.0 * 0.0 + 3.0 * mid; // x=0, y=0, z=mid
  REQUIRE(field(0, 0, mid) == Approx(expected_mid));

  // Test other mid-plane values
  double expected_mid_xy = mid + 2.0 * mid + 3.0 * 0.0; // x=mid, y=mid, z=0
  REQUIRE(field(mid, mid, 0) == Approx(expected_mid_xy));
}

TEST_CASE("field_from_subdomain: slice indexing consistency (Domain version)",
          "[field_factory][domain][unit]") {
  const int nx = 4, ny = 5, nz = 6;
  auto world = domain::create_world({nx, ny, nz});
  auto decomp = decomposition::create(world, 1);
  
  auto field = pfc::data::field_from_subdomain<double>(decomp, 0, 0);
  
  // Initialize with sequential values
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        field(i, j, k) = i + nx * (j + ny * k);
      }
    }
  }
  
  // Verify slice access matches 3D access
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        double expected = i + nx * (j + ny * k);
        REQUIRE(field(i, j, k) == Approx(expected));
        
        // Verify global index consistency
        std::size_t global_idx = field.idx(i, j, k);
        REQUIRE(field.data()[global_idx] == Approx(expected));
      }
    }
  }
}
