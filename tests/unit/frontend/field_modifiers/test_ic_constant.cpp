// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/types.hpp>
#include <openpfc/kernel/simulation/initial_conditions/constant.hpp>

using namespace pfc;

TEST_CASE("Constant Field Modifier") {

  SECTION("Density value") {
    Constant c(1.0);
    REQUIRE(c.get_density() == 1.0);
    c.set_density(2.5);
    REQUIRE(c.get_density() == 2.5);
  }

  SECTION("Apply field modifier") {
    auto domain = pfc::domain::create(pfc::Int3{8, 1, 1});
    auto box = pfc::domain::index_box(domain);
    std::vector<double> psi(8, 0.0);

    Constant c(1.0);
    c.apply(psi, domain, box);
    bool values_match = true;
    for (const auto &value : psi) {
      values_match &= value == 1.0;
    }
    REQUIRE(values_match);
  }
}

TEST_CASE("IC Constant - Domain box size", "[ic_constant]") {
  auto domain = pfc::domain::create(pfc::Int3{8, 8, 8});
  REQUIRE(pfc::domain::get_size(domain) == Int3{8, 8, 8});
  REQUIRE(pfc::domain::index_box(domain).count() == 512);
}
