// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <sstream>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <openpfc/kernel/simulation/time.hpp>

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

TEST_CASE("Time - set_increment rejects negative values", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  REQUIRE_THROWS_AS(t.set_increment(-1), std::invalid_argument);
}

TEST_CASE("Time - get_current invariant after next()", "[time][unit]") {
  Time t({0.0, 1.0, 0.1}, 0.0);
  for (int n = 0; n <= 15; ++n) {
    const double raw = t.get_t0() + static_cast<double>(n) * t.get_dt();
    const double expected = std::min(raw, t.get_t1());
    REQUIRE_THAT(t.get_current(), WithinAbs(expected, TOLERANCE));
    if (n < 15) {
      t.next();
    }
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

  SECTION("end time before start time") {
    REQUIRE_THROWS_AS(Time({10.0, 5.0, 1.0}), std::invalid_argument);
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

TEST_CASE("Time::set_dt accepts positive values", "[time]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  // Test initial dt
  REQUIRE_THAT(time.get_dt(), WithinAbs(0.1, TOLERANCE));

  // Set new dt and verify
  time.set_dt(0.05);
  REQUIRE_THAT(time.get_dt(), WithinAbs(0.05, TOLERANCE));

  // Set another dt and verify
  time.set_dt(0.2);
  REQUIRE_THAT(time.get_dt(), WithinAbs(0.2, TOLERANCE));

  // Test with very small positive dt
  time.set_dt(1e-8);
  REQUIRE_THAT(time.get_dt(), WithinAbs(1e-8, TOLERANCE));

  // Test with large dt
  time.set_dt(100.0);
  REQUIRE_THAT(time.get_dt(), WithinAbs(100.0, TOLERANCE));
}

TEST_CASE("Time::set_dt rejects zero", "[time]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  REQUIRE_THROWS_AS(time.set_dt(0.0), std::invalid_argument);

  // Verify dt is unchanged after failed set
  REQUIRE_THAT(time.get_dt(), WithinAbs(0.1, TOLERANCE));
}

TEST_CASE("Time::set_dt rejects negative values", "[time]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  // Test negative dt
  REQUIRE_THROWS_AS(time.set_dt(-0.1), std::invalid_argument);
  REQUIRE_THROWS_AS(time.set_dt(-1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(time.set_dt(-1e-10), std::invalid_argument);

  // Verify dt is unchanged after failed set
  REQUIRE_THAT(time.get_dt(), WithinAbs(0.1, TOLERANCE));
}

TEST_CASE("TimeStateGuard - restores dt and increment on destruction",
          "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  // Modify time state
  time.set_dt(0.05);
  time.set_increment(5);

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.05, TOLERANCE));
  REQUIRE(time.get_increment() == 5);

  {
    TimeStateGuard guard(time);
    // Guard has captured dt=0.05, increment=5

    // Modify time state again
    time.set_dt(0.02);
    time.set_increment(10);

    REQUIRE_THAT(time.get_dt(), WithinAbs(0.02, TOLERANCE));
    REQUIRE(time.get_increment() == 10);
  }
  // Guard destroyed: dt and increment restored

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.05, TOLERANCE));
  REQUIRE(time.get_increment() == 5);
}

TEST_CASE("TimeStateGuard - commit disables restoration", "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  time.set_dt(0.05);
  time.set_increment(5);

  {
    TimeStateGuard guard(time);

    time.set_dt(0.02);
    time.set_increment(10);

    guard.commit(); // Mark step as accepted
  }
  // Guard destroyed but committed: no restoration

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.02, TOLERANCE));
  REQUIRE(time.get_increment() == 10);
}

TEST_CASE("TimeStateGuard - multiple scopes stack correctly", "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  {
    TimeStateGuard outer(time);
    time.set_dt(0.05);
    time.set_increment(5);

    {
      TimeStateGuard inner(time);
      time.set_dt(0.02);
      time.set_increment(10);

      REQUIRE_THAT(time.get_dt(), WithinAbs(0.02, TOLERANCE));
      REQUIRE(time.get_increment() == 10);
    }
    // Inner destroyed: restore to outer's captured state

    REQUIRE_THAT(time.get_dt(), WithinAbs(0.05, TOLERANCE));
    REQUIRE(time.get_increment() == 5);
  }
  // Outer destroyed: restore to original state

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.1, TOLERANCE));
  REQUIRE(time.get_increment() == 0);
}

TEST_CASE("TimeStateGuard - move semantics preserve state", "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  time.set_dt(0.05);
  time.set_increment(5);

  {
    TimeStateGuard original(time);
    time.set_dt(0.02);
    time.set_increment(10);

    // Move construct
    TimeStateGuard moved(std::move(original));

    // Original is now committed (no restoration)
    REQUIRE(original.committed());

    // Modify time again
    time.set_dt(0.01);
    time.set_increment(15);

    REQUIRE_THAT(time.get_dt(), WithinAbs(0.01, TOLERANCE));
    REQUIRE(time.get_increment() == 15);
  }
  // End of scope: moved guard destroyed, should restore to original captured state
  // (dt=0.05, increment=5)

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.05, TOLERANCE));
  REQUIRE(time.get_increment() == 5);
}

TEST_CASE("TimeStateGuard - move assignment restores before transfer",
          "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  time.set_dt(0.05);
  time.set_increment(5);

  TimeStateGuard source(time);
  time.set_dt(0.02);
  time.set_increment(10);

  // Create target guard
  time.set_dt(0.03);
  time.set_increment(8);

  TimeStateGuard target(time);
  time.set_dt(0.015);
  time.set_increment(12);

  // Target should restore to dt=0.03, increment=8 before taking over
  target = std::move(source);

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.03, TOLERANCE));
  REQUIRE(time.get_increment() == 8);

  // Source is now committed (no restoration)
  REQUIRE(source.committed());
}

TEST_CASE("TimeStateGuard - committed() reflects state", "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  TimeStateGuard guard(time);

  REQUIRE_FALSE(guard.committed());

  guard.commit();

  REQUIRE(guard.committed());
}

TEST_CASE("TimeStateGuard - with time state modifications", "[time][unit]") {
  Time time({0.0, 10.0, 0.1}, 1.0);

  {
    TimeStateGuard guard(time);

    // Modify dt multiple times
    time.set_dt(0.05);
    time.set_dt(0.02);
    time.set_dt(0.01);

    // Modify increment multiple times
    time.set_increment(5);
    time.set_increment(10);
    time.set_increment(15);
  }
  // Should restore to original state

  REQUIRE_THAT(time.get_dt(), WithinAbs(0.1, TOLERANCE));
  REQUIRE(time.get_increment() == 0);
}

TEST_CASE("Time - stage tracking initializes to 0/1", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 2.0);

  REQUIRE(t.get_stage() == 0);
  REQUIRE(t.get_stage_count() == 1);
}

TEST_CASE("Time - set_stage_count and set_stage accept valid values",
          "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);

  t.set_stage_count(4);
  REQUIRE(t.get_stage_count() == 4);

  t.set_stage(0);
  REQUIRE(t.get_stage() == 0);

  t.set_stage(3);
  REQUIRE(t.get_stage() == 3);
}

TEST_CASE("Time - set_stage rejects out-of-range values", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.set_stage_count(2);

  REQUIRE_THROWS_AS(t.set_stage(-1), std::invalid_argument);
  REQUIRE_THROWS_AS(t.set_stage(2), std::invalid_argument);
}

TEST_CASE("Time - set_stage_count rejects values below 1", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);

  REQUIRE_THROWS_AS(t.set_stage_count(0), std::invalid_argument);
  REQUIRE_THROWS_AS(t.set_stage_count(-1), std::invalid_argument);
}

TEST_CASE("Time - stage is independent from increment", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.set_stage_count(4);
  t.set_stage(2);

  t.next();
  t.next();

  REQUIRE(t.get_increment() == 2);
  REQUIRE(t.get_stage() == 2); // next() does not touch stage
}

TEST_CASE("Time - non-member stage functions mirror member functions",
          "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);

  pfc::time::set_stage_count(t, 3);
  REQUIRE(pfc::time::stage_count(t) == 3);

  pfc::time::set_stage(t, 1);
  REQUIRE(pfc::time::stage(t) == 1);
}

TEST_CASE("Time - operator<< includes stage information", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.set_stage_count(4);
  t.set_stage(1);

  std::ostringstream oss;
  oss << t;

  REQUIRE_THAT(oss.str(), ContainsSubstring("stage = 1/4"));
}
