#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <openpfc/discrete_field.hpp>

using namespace Catch::Matchers;
// using namespace pfc;
using namespace pfc::utils;

TEST_CASE("DiscreteField1D") {
  int Lx = 5;
  int i0 = -2;
  double x0 = 1.0;
  double dx = 2.0;
  DiscreteField<int, 1> field({Lx}, {i0}, {x0}, {dx});

  SECTION("Accessing elements using indices") {
    field({0}) = 1;
    REQUIRE(field({0}) == 1);
  }

  SECTION("Accessing elements using coordinates") {
    field.interpolate({2.0}) = 1;
    REQUIRE(field.interpolate({1.9}) == 0);
    REQUIRE(field.interpolate({2.0}) == 1);
    REQUIRE(field.interpolate({2.1}) == 1);
  }

  SECTION("Test apply()") {
    auto func = [](const std::array<double, 1> &coords) -> int { return static_cast<int>(coords[0]); };
    field.apply(func);
    for (int i = 0; i < Lx; i++) REQUIRE(field[i] == -3 + 2 * i);
  }
}
