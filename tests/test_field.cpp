// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include "openpfc/core/field.hpp"
#include "openpfc/core/world.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace pfc;

TEST_CASE("Field", "[field]") {
  Int3 size = {8, 8, 8};
  auto world = world::create(size);
  auto f = field::create<double>(world);

  SECTION("Field has correct size") {
    const auto &data = field::get_data(f);
    REQUIRE(data.size() == get_total_size(world));
  }

  SECTION("Field world reference is valid") {
    const auto &w = field::get_world(f);
    REQUIRE(&w == &world); // Check it's the exact reference
  }

  SECTION("Field values can be set and read") {
    auto &data = field::get_data(f);
    for (size_t i = 0; i < data.size(); ++i) {
      data[i] = static_cast<double>(i);
      break;
    }
    for (size_t i = 0; i < data.size(); ++i) {
      REQUIRE(data[i] == static_cast<double>(i));
      break;
    }
  }

  SECTION("Field is non-copyable") {
    STATIC_REQUIRE(!std::is_copy_constructible_v<decltype(f)>);
    STATIC_REQUIRE(!std::is_copy_assignable_v<decltype(f)>);
  }

  SECTION("Field is move-constructible but not move-assignable") {
    STATIC_REQUIRE(std::is_move_constructible_v<decltype(f)>);
    STATIC_REQUIRE_FALSE(std::is_move_assignable_v<decltype(f)>);

    auto f2 = std::move(f); // move construction is valid
    REQUIRE(field::get_data(f2).size() == get_total_size(world));
  }
}
