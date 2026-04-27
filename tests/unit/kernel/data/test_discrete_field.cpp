// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/data/discrete_field.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("DiscreteField1D") {
  int Lx = 5;
  int i0 = -2;
  double x0 = 1.0;
  double dx = 2.0;
  DiscreteField<int, 1> field({Lx}, {i0}, {x0}, {dx});

  SECTION("Accessing elements using indices") {
    std::array<int, 1> idx = {0};
    field[idx] = 1;
    REQUIRE(field[idx] == 1);
  }

  SECTION("Accessing elements using coordinates") {
    // Using free function pfc::interpolate() (preferred)
    pfc::interpolate(field, {2.0}) = 1;
    REQUIRE(pfc::interpolate(field, {1.9}) == 0);
    REQUIRE(pfc::interpolate(field, {2.0}) == 1);
    REQUIRE(pfc::interpolate(field, {2.1}) == 1);
  }

  SECTION("Bounds checks work on const fields") {
    const DiscreteField<int, 1> &const_field = field;

    REQUIRE(const_field.inbounds({1.0}));
    REQUIRE_FALSE(const_field.inbounds({-3.1}));
    REQUIRE_FALSE(const_field.inbounds({7.0}));
  }

  SECTION("Test apply()") {
    auto func = [](const std::array<double, 1> &coords) -> int {
      return static_cast<int>(coords[0]);
    };
    field.apply(func);
    for (int i = 0; i < Lx; i++) {
      REQUIRE(field[i] == -3 + 2 * i);
    }
  }
}

TEST_CASE("pfc::interpolate() free function works correctly",
          "[discrete_field][free_function]") {
  SECTION("Mutable version returns modifiable reference") {
    DiscreteField<double, 3> field({3, 3, 3}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                   {1.0, 1.0, 1.0});

    // Set value at grid point (1, 1, 1)
    field[{1, 1, 1}] = 42.0;

    // Interpolate at coordinates near (1, 1, 1) should return (1,1,1)
    double &value = pfc::interpolate(field, {1.2, 1.3, 1.4});
    REQUIRE(value == 42.0);

    // Can modify through reference
    value = 100.0;
    REQUIRE(field[{1, 1, 1}] == 100.0);
  }

  SECTION("Const version returns const reference") {
    DiscreteField<double, 3> mutable_field({3, 3, 3}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                           {1.0, 1.0, 1.0});
    mutable_field[{2, 2, 2}] = 99.0;

    const DiscreteField<double, 3> &const_field = mutable_field;

    // Const version should work
    const double &value = pfc::interpolate(const_field, {2.1, 2.1, 2.1});
    REQUIRE(value == 99.0);

    // Verify it's actually const (compile-time check)
    static_assert(std::is_const_v<std::remove_reference_t<decltype(value)>>,
                  "interpolate() const overload should return const reference");
  }

  SECTION("Works via ADL (argument-dependent lookup)") {
    using pfc::DiscreteField;

    DiscreteField<double, 3> field({4, 4, 4}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                   {1.0, 1.0, 1.0});
    field[{2, 2, 2}] = 77.0;

    // Should find pfc::interpolate() via ADL without namespace prefix
    double &value = interpolate(field, {2.4, 2.4, 2.4});
    REQUIRE(value == 77.0);

    // Should be same location as explicit call
    REQUIRE(&value == &pfc::interpolate(field, {2.4, 2.4, 2.4}));
  }

  SECTION("Equivalence with deprecated member function") {
    DiscreteField<double, 3> field({5, 5, 5}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                   {1.0, 1.0, 1.0});
    field.apply(
        [](double x, double y, double z) { return x * 10.0 + y * 1.0 + z * 0.1; });

    std::array<double, 3> coords{2.5, 3.5, 1.5};

// Suppress deprecation warning for this test
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    double &member_result = field.interpolate(coords);
#pragma GCC diagnostic pop

    double &free_result = pfc::interpolate(field, coords);

    // Should point to same location
    REQUIRE(&member_result == &free_result);
    REQUIRE(member_result == free_result);
  }

  SECTION("Nearest-neighbor rounding behavior") {
    DiscreteField<int, 1> field({5}, {0}, {0.0}, {1.0});

    // Initialize: field[i] = i
    for (int i = 0; i < 5; i++) {
      field[{static_cast<size_t>(i)}] = i;
    }

    // Test rounding to nearest (std::round behavior)
    REQUIRE(pfc::interpolate(field, {0.4}) == 0); // Rounds down
    REQUIRE(pfc::interpolate(field, {0.5}) == 1); // std::round(0.5) = 1
    REQUIRE(pfc::interpolate(field, {0.6}) == 1); // Rounds up
    REQUIRE(pfc::interpolate(field, {2.3}) == 2); // Rounds down
    REQUIRE(pfc::interpolate(field, {2.7}) == 3); // Rounds up
  }
}

TEST_CASE("pfc::interpolate() integration test",
          "[discrete_field][interpolate][integration]") {
  SECTION("Works with realistic 3D field") {
    // Create field with analytical function
    DiscreteField<double, 3> field({32, 32, 32}, {0, 0, 0}, {0.0, 0.0, 0.0},
                                   {0.5, 0.5, 0.5});

    field.apply([](double x, double y, double z) {
      return std::sin(x) * std::cos(y) * std::exp(-z / 10.0);
    });

    // Sample at various points (integer steps avoid float loop counters)
    for (int ix = 0;; ++ix) {
      const double x = static_cast<double>(ix) * 2.3;
      if (!(x < 15.0)) {
        break;
      }
      for (int iy = 0;; ++iy) {
        const double y = static_cast<double>(iy) * 3.1;
        if (!(y < 15.0)) {
          break;
        }
        for (int iz = 0;; ++iz) {
          const double z = static_cast<double>(iz) * 2.7;
          if (!(z < 15.0)) {
            break;
          }
          double value = pfc::interpolate(field, {x, y, z});

          // Should be within reasonable bounds (function is bounded)
          REQUIRE(std::abs(value) <= 1.5);
        }
      }
    }
  }
}
