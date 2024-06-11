/*

OpenPFC, a simulation software for the phase field crystal method.
Copyright (C) 2024 VTT Technical Research Centre of Finland Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

*/

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <iostream>
#include <openpfc/discrete_field.hpp>

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
