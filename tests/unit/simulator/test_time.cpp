// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "openpfc/time.hpp"

using namespace Catch::Matchers;
using namespace pfc;

constexpr double TOLERANCE = 1e-10;

TEST_CASE("Time - initialization", "[time][unit]") {
  SECTION("with save interval") {
    Time t({0.0, 10.0, 1.0}, 2.0);

    REQUIRE_THAT(t.get_t0(), WithinAbs(0.0, TOLERANCE));
    REQUIRE_THAT(t.get_t1(), WithinAbs(10.0, TOLERANCE));
    REQUIRE_THAT(t.get_dt(), WithinAbs(1.0, TOLERANCE));
    REQUIRE_THAT(t.get_saveat(), WithinAbs(2.0, TOLERANCE));
    REQUIRE_THAT(t.get_current(), WithinAbs(0.0, TOLERANCE));
    REQUIRE(t.get_increment() == 0);
  }

  SECTION("without save interval") {
    Time t({0.0, 5.0, 0.5});

    REQUIRE_THAT(t.get_t0(), WithinAbs(0.0, TOLERANCE));
    REQUIRE_THAT(t.get_t1(), WithinAbs(5.0, TOLERANCE));
    REQUIRE_THAT(t.get_dt(), WithinAbs(0.5, TOLERANCE));
    REQUIRE_THAT(t.get_saveat(), WithinAbs(0.5, TOLERANCE));
    REQUIRE_THAT(t.get_current(), WithinAbs(0.0, TOLERANCE));
    REQUIRE(t.get_increment() == 0);
  }
}

TEST_CASE("Time - validation errors", "[time][unit]") {
  SECTION("negative start time") {
    REQUIRE_THROWS_AS(Time({-1.0, 10.0, 1.0}), std::invalid_argument);
  }

  SECTION("zero time step") {
    REQUIRE_THROWS_AS(Time({0.0, 10.0, 0.0}), std::invalid_argument);
  }

  SECTION("save interval exceeds end time") {
    REQUIRE_THROWS_AS(Time({0.0, 10.0, 1.0}, 15.0), std::invalid_argument);
  }

  SECTION("extremely small time step") {
    REQUIRE_THROWS_AS(Time({0.0, 10.0, 1e-10}), std::invalid_argument);
  }

  SECTION("start time equals end time") {
    REQUIRE_THROWS_AS(Time({5.0, 5.0, 1.0}), std::invalid_argument);
  }
}

TEST_CASE("Time - increment overflow", "[time][unit]") {
  Time t({0.0, 1e6, 1.0});

  t.set_increment(static_cast<int>(1e6 + 1));

  REQUIRE(t.done());
  REQUIRE_THAT(t.get_current(), WithinAbs(1e6, TOLERANCE));
}

TEST_CASE("Time - completion detection", "[time][unit]") {
  Time t({0.0, 10.0, 1.0});

  REQUIRE_FALSE(t.done());

  SECTION("complete when increment reaches maximum") {
    t.set_increment(10);
    REQUIRE(t.done());
  }

  SECTION("incomplete when increment is less than maximum") {
    t.set_increment(5);
    REQUIRE_FALSE(t.done());
  }
}

TEST_CASE("Time - save condition", "[time][unit]") {
  Time t({0.0, 10.0, 1.0});

  SECTION("save at first increment") { REQUIRE(t.do_save()); }

  SECTION("save at last increment") {
    t.set_increment(10);
    REQUIRE(t.do_save());
  }

  SECTION("save at aligned intervals") {
    t.set_saveat(2.5);

    t.set_increment(5);
    REQUIRE(t.do_save());

    t.set_increment(6);
    REQUIRE_FALSE(t.do_save());
  }

  SECTION("no save when interval is zero") {
    t.set_saveat(0.0);
    REQUIRE_FALSE(t.do_save());
  }

  SECTION("no save when interval non-aligned with dt") {
    t.set_saveat(1.3);
    t.set_increment(3);
    REQUIRE_FALSE(t.do_save());
  }
}
