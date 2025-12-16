// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file bench_world_coords.cpp
 * @brief Microbenchmarks for World and coordinate system operations
 *
 * These benchmarks measure the performance of core coordinate transformation
 * operations. Since these functions are used in hot paths (inner loops of
 * field operations), they must be zero-cost abstractions.
 *
 * Expected performance characteristics:
 * - to_coords(): ~1-5 ns (should inline to simple arithmetic)
 * - to_indices(): ~1-5 ns (should inline to simple arithmetic)
 * - get_spacing/get_origin: ~0.5-2 ns (should inline to member access)
 *
 * Run with:
 *   ./build/tests/openpfc-tests "[world][benchmark]"
 *
 * For accurate results:
 *   - Use Release build: cmake -B build -DCMAKE_BUILD_TYPE=Release
 *   - Run on dedicated system (no background processes)
 *   - Check that operations actually inline (use -S flag to check assembly)
 */

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include "openpfc/core/world.hpp"

using namespace pfc;
using namespace pfc::types;

// ============================================================================
// Coordinate Transformation Benchmarks
// ============================================================================

TEST_CASE("World coordinate transformations - microbenchmarks",
          "[world][coords][benchmark]") {
  // Setup: Create a typical simulation world
  const Int3 size = {128, 128, 128};
  const Real3 origin = {0.0, 0.0, 0.0};
  const Real3 spacing = {0.1, 0.1, 0.1};
  const auto world =
      world::create(GridSize(size), PhysicalOrigin(origin), GridSpacing(spacing));

  // Test indices - use values that prevent compiler optimizations
  volatile int idx_base = 42; // volatile prevents constant folding
  const Int3 test_indices = {idx_base, idx_base + 10, idx_base + 20};

  SECTION("to_coords - index to physical coordinate conversion") {
    BENCHMARK("to_coords (single call)") { return to_coords(world, test_indices); };

    INFO("Expected: ~1-5 ns per call");
    INFO("This should inline to: origin[i] + indices[i] * spacing[i]");
  }

  SECTION("to_indices - physical coordinate to index conversion") {
    const Real3 test_coords = {4.2, 5.3, 6.4};

    BENCHMARK("to_indices (single call)") { return to_indices(world, test_coords); };

    INFO("Expected: ~1-5 ns per call");
    INFO("This should inline to: (coords[i] - origin[i]) / spacing[i]");
  }

  SECTION("Round-trip transformation") {
    BENCHMARK("to_coords → to_indices (round-trip)") {
      const auto coords = to_coords(world, test_indices);
      return to_indices(world, coords);
    };

    INFO("Expected: ~2-10 ns per round-trip");
    INFO("Verifies that both transformations inline properly");
  }
}

// ============================================================================
// Coordinate System Access Benchmarks
// ============================================================================

TEST_CASE("World accessor functions - microbenchmarks",
          "[world][accessors][benchmark]") {
  const auto world =
      world::create(GridSize({128, 128, 128}), PhysicalOrigin({1.0, 2.0, 3.0}),
                    GridSpacing({0.1, 0.1, 0.1}));

  SECTION("get_spacing (all dimensions)") {
    BENCHMARK("get_spacing (Real3)") { return get_spacing(world); };

    INFO("Expected: <1 ns (should inline to return m_cs.m_spacing)");
  }

  SECTION("get_spacing (single dimension)") {
    BENCHMARK("get_spacing (dimension 0)") { return get_spacing(world, 0); };

    INFO("Expected: <1 ns (should inline to return m_cs.m_spacing[0])");
  }

  SECTION("get_origin (all dimensions)") {
    BENCHMARK("get_origin (Real3)") { return get_origin(world); };

    INFO("Expected: <1 ns (should inline to return m_cs.m_offset)");
  }

  SECTION("get_origin (single dimension)") {
    BENCHMARK("get_origin (dimension 0)") { return get_origin(world, 0); };

    INFO("Expected: <1 ns (should inline to return m_cs.m_offset[0])");
  }

  SECTION("get_size") {
    BENCHMARK("get_size") { return get_size(world); };

    INFO("Expected: <1 ns (should inline to return m_size)");
  }

  SECTION("get_total_size") {
    BENCHMARK("get_total_size") { return get_total_size(world); };

    INFO("Expected: ~1-2 ns (size[0] * size[1] * size[2])");
  }
}

// ============================================================================
// Coordinate System Direct Access Benchmarks
// ============================================================================

TEST_CASE("CoordinateSystem direct operations - microbenchmarks",
          "[csys][benchmark]") {
  using namespace pfc::csys;

  // Create coordinate system directly
  const Real3 offset = {0.0, 0.0, 0.0};
  const Real3 spacing = {0.1, 0.1, 0.1};
  const Bool3 periodic = {true, true, true};
  const CartesianCS cs(offset, spacing, periodic);

  const Int3 test_indices = {42, 53, 64};

  SECTION("to_coords on bare CoordinateSystem") {
    BENCHMARK("to_coords (CartesianCS)") { return to_coords(cs, test_indices); };

    INFO("Expected: ~1-5 ns");
    INFO("Verifies no overhead from World wrapper");
  }

  SECTION("to_index on bare CoordinateSystem") {
    const Real3 test_coords = {4.2, 5.3, 6.4};

    BENCHMARK("to_index (CartesianCS)") { return to_index(cs, test_coords); };

    INFO("Expected: ~1-5 ns");
    INFO("Verifies no overhead from World wrapper");
  }

  SECTION("get_spacing on CoordinateSystem") {
    BENCHMARK("get_spacing (CartesianCS)") { return get_spacing(cs); };

    INFO("Expected: <1 ns (direct member access)");
  }

  SECTION("get_offset on CoordinateSystem") {
    BENCHMARK("get_offset (CartesianCS)") { return get_offset(cs); };

    INFO("Expected: <1 ns (direct member access)");
  }
}

// ============================================================================
// Loop-Based Realistic Usage Benchmarks
// ============================================================================

TEST_CASE("World operations in loops - realistic usage patterns",
          "[world][loop][benchmark]") {
  const auto world =
      world::create(GridSize({64, 64, 64}), PhysicalOrigin({0.0, 0.0, 0.0}),
                    GridSpacing({0.1, 0.1, 0.1}));
  const auto size = get_size(world);

  SECTION("Loop over all grid points - coordinate conversion") {
    BENCHMARK("Convert all grid indices to coordinates") {
      double sum = 0.0;
      for (int k = 0; k < size[2]; ++k) {
        for (int j = 0; j < size[1]; ++j) {
          for (int i = 0; i < size[0]; ++i) {
            const auto coords = to_coords(world, {i, j, k});
            sum += coords[0] + coords[1] + coords[2];
          }
        }
      }
      return sum;
    };

    INFO("Measures overhead in typical field initialization loop");
    INFO("Expected: <1 ms for 64³ grid (256K points)");
  }

  SECTION("Loop with coordinate-dependent calculation") {
    BENCHMARK("Gaussian initialization (coordinate-based)") {
      double sum = 0.0;
      const Real3 center = {3.2, 3.2, 3.2}; // Center of domain
      const double sigma = 1.0;

      for (int k = 0; k < size[2]; ++k) {
        for (int j = 0; j < size[1]; ++j) {
          for (int i = 0; i < size[0]; ++i) {
            const auto pos = to_coords(world, {i, j, k});
            const double dx = pos[0] - center[0];
            const double dy = pos[1] - center[1];
            const double dz = pos[2] - center[2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            sum += std::exp(-r2 / (2.0 * sigma * sigma));
          }
        }
      }
      return sum;
    };

    INFO("Simulates real initialization pattern");
    INFO("Coordinate transform should be negligible vs exp() cost");
  }

  SECTION("Sparse coordinate access pattern") {
    BENCHMARK("Convert 1000 random indices to coordinates") {
      double sum = 0.0;
      // Use deterministic "random" pattern to ensure reproducibility
      for (int n = 0; n < 1000; ++n) {
        const int i = (n * 37) % size[0];
        const int j = (n * 41) % size[1];
        const int k = (n * 43) % size[2];
        const auto coords = to_coords(world, {i, j, k});
        sum += coords[0] + coords[1] + coords[2];
      }
      return sum;
    };

    INFO("Measures overhead for sparse access patterns");
    INFO("Expected: <10 μs for 1000 conversions");
  }
}

// ============================================================================
// Compilation Overhead Test (compile-time check)
// ============================================================================

TEST_CASE("World zero-cost abstraction validation",
          "[world][zero-cost][benchmark]") {
  SECTION("Manual inline calculation (baseline)") {
    // This is what the compiler SHOULD reduce to_coords() to
    const Real3 offset = {0.0, 0.0, 0.0};
    const Real3 spacing = {0.1, 0.1, 0.1};
    const Int3 indices = {42, 53, 64};

    BENCHMARK("Manual coordinate calculation (baseline)") {
      Real3 result;
      result[0] = offset[0] + indices[0] * spacing[0];
      result[1] = offset[1] + indices[1] * spacing[1];
      result[2] = offset[2] + indices[2] * spacing[2];
      return result;
    };

    INFO("This is the theoretical minimum - raw arithmetic");
    INFO("to_coords() benchmark should match this closely");
  }

  SECTION("World abstraction (should match baseline)") {
    const auto world =
        world::create(GridSize({128, 128, 128}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.1, 0.1, 0.1}));
    const Int3 indices = {42, 53, 64};

    BENCHMARK("World to_coords (abstraction)") { return to_coords(world, indices); };

    INFO("Should be within ~10% of baseline if properly inlined");
    INFO("Significant difference indicates abstraction overhead");
  }
}

// ============================================================================
// Memory Access Pattern Benchmarks
// ============================================================================

TEST_CASE("World cache and memory access patterns", "[world][memory][benchmark]") {
  SECTION("Sequential world creation and destruction") {
    BENCHMARK("Create and destroy World (stack)") {
      auto world =
          world::create(GridSize({128}), PhysicalOrigin({128}), GridSpacing({128}));
      return get_total_size(world);
    };

    INFO("Measures constructor/destructor overhead");
    INFO("Expected: <100 ns (should be trivial for small struct)");
  }

  SECTION("World copy performance") {
    const auto world1 =
        world::create(GridSize({128}), PhysicalOrigin({128}), GridSpacing({128}));

    BENCHMARK("Copy World object") {
      auto world2 = world1; // Copy constructor
      return get_total_size(world2);
    };

    INFO("Measures copy constructor performance");
    INFO("World is value type - copies should be cheap");
  }

  SECTION("World equality comparison") {
    const auto world1 =
        world::create(GridSize({128, 128, 128}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.1, 0.1, 0.1}));
    const auto world2 =
        world::create(GridSize({128, 128, 128}), PhysicalOrigin({0.0, 0.0, 0.0}),
                      GridSpacing({0.1, 0.1, 0.1}));

    BENCHMARK("World equality comparison") { return world1 == world2; };

    INFO("Measures comparison operator performance");
    INFO("Expected: <10 ns (compare 3 Real3 + 1 Int3)");
  }
}
