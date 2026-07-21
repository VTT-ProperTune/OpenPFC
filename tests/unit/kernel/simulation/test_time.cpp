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

TEST_CASE("Time - get_step_count returns zero initially", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  REQUIRE(t.get_step_count() == 0);
}

TEST_CASE("Time - get_step_count increments with next", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.next();
  REQUIRE(t.get_step_count() == 1);
  t.next();
  t.next();
  REQUIRE(t.get_step_count() == 3);
}

TEST_CASE("Time - get_step_count matches get_increment", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.next();
  t.next();
  REQUIRE(t.get_step_count() == t.get_increment());
}

TEST_CASE("test_time_step_counters_initialization", "[time][unit]") {
  SECTION("one-arg constructor") {
    Time t({0.0, 10.0, 1.0});
    REQUIRE(t.get_accepted_steps() == 0);
    REQUIRE(t.get_rejected_steps() == 0);
  }

  SECTION("two-arg constructor with saveat") {
    Time t({0.0, 10.0, 1.0}, 2.0);
    REQUIRE(t.get_accepted_steps() == 0);
    REQUIRE(t.get_rejected_steps() == 0);
  }

  SECTION("three-arg constructor with method") {
    Time t({0.0, 10.0, 1.0}, 2.0, IntegratorMethod::rk2_heun);
    REQUIRE(t.get_accepted_steps() == 0);
    REQUIRE(t.get_rejected_steps() == 0);
  }
}

TEST_CASE("test_time_increment_step_success", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.increment_step_success();
  REQUIRE(t.get_accepted_steps() == 1);
  REQUIRE(t.get_rejected_steps() == 0);
}

TEST_CASE("test_time_increment_step_rejection", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.increment_step_rejection();
  REQUIRE(t.get_rejected_steps() == 1);
  REQUIRE(t.get_accepted_steps() == 0);
}

TEST_CASE("test_time_get_accepted_and_rejected_steps", "[time][unit]") {
  Time t({0.0, 10.0, 1.0}, 0.0);
  t.increment_step_success();
  t.increment_step_success();
  t.increment_step_rejection();
  t.increment_step_success();
  t.increment_step_rejection();
  t.increment_step_rejection();
  REQUIRE(t.get_accepted_steps() == 3);
  REQUIRE(t.get_rejected_steps() == 3);
}

TEST_CASE("test_time_state_guard_restores_counters", "[time][unit]") {
  SECTION("uncommitted guard restores counters") {
    Time time({0.0, 10.0, 0.1}, 1.0);
    time.increment_step_success();
    time.increment_step_rejection();
    REQUIRE(time.get_accepted_steps() == 1);
    REQUIRE(time.get_rejected_steps() == 1);

    {
      TimeStateGuard guard(time);
      time.increment_step_success();
      time.increment_step_success();
      time.increment_step_rejection();
      REQUIRE(time.get_accepted_steps() == 3);
      REQUIRE(time.get_rejected_steps() == 2);
    }

    REQUIRE(time.get_accepted_steps() == 1);
    REQUIRE(time.get_rejected_steps() == 1);
  }

  SECTION("committed guard preserves counters") {
    Time time({0.0, 10.0, 0.1}, 1.0);
    time.increment_step_success();
    REQUIRE(time.get_accepted_steps() == 1);
    REQUIRE(time.get_rejected_steps() == 0);

    {
      TimeStateGuard guard(time);
      time.increment_step_success();
      time.increment_step_rejection();
      guard.commit();
      REQUIRE(time.get_accepted_steps() == 2);
      REQUIRE(time.get_rejected_steps() == 1);
    }

    REQUIRE(time.get_accepted_steps() == 2);
    REQUIRE(time.get_rejected_steps() == 1);
  }
}

TEST_CASE("Time - attempt accepted time immutable on reject", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 0.0);
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.0, TOLERANCE));
  REQUIRE_THAT(t.get_current(), WithinAbs(0.0, TOLERANCE));

  t.begin_attempt(0.5);
  REQUIRE(t.attempt_active());
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.0, TOLERANCE));
  REQUIRE_THAT(t.get_current(), WithinAbs(0.0, TOLERANCE));
  REQUIRE_THAT(t.get_attempted_dt(), WithinAbs(0.5, TOLERANCE));

  t.reject_attempt();
  REQUIRE_FALSE(t.attempt_active());
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.0, TOLERANCE));
  REQUIRE(t.get_increment() == 0);

  for (int i = 0; i < 5; ++i) {
    t.begin_attempt(0.25);
    t.reject_attempt();
  }
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.0, TOLERANCE));
  REQUIRE(t.get_increment() == 0);
}

TEST_CASE("Time - clip_attempt_dt terminal bound", "[time][unit]") {
  Time t({0.0, 1.0, 0.1}, 0.0);
  t.set_increment(8); // accepted = 0.8
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.8, TOLERANCE));

  const double clipped = t.clip_attempt_dt(1.0);
  REQUIRE_THAT(clipped, WithinAbs(0.2, TOLERANCE));
  REQUIRE_THAT(t.get_accepted_time() + clipped, WithinAbs(1.0, TOLERANCE));
  REQUIRE(clipped < 1.0);

  REQUIRE_THROWS_AS(t.clip_attempt_dt(0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(t.clip_attempt_dt(-0.1), std::invalid_argument);
}

TEST_CASE("Time - clip_attempt_dt saveat alignment", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 1.0);
  t.set_increment(1); // accepted = 0.5, next save = 1.0
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.5, TOLERANCE));

  const double clipped = t.clip_attempt_dt(0.8);
  REQUIRE_THAT(clipped, WithinAbs(0.5, TOLERANCE));
  REQUIRE_THAT(t.get_accepted_time() + clipped, WithinAbs(1.0, TOLERANCE));
}

TEST_CASE("Time - clip_attempt_dt saveat<=0 skips alignment", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 0.0);
  t.set_increment(1); // accepted = 0.5
  // With saveat disabled, a large candidate only hits the terminal bound.
  const double clipped = t.clip_attempt_dt(3.0);
  REQUIRE_THAT(clipped, WithinAbs(3.0, TOLERANCE));

  Time near_end({0.0, 1.0, 0.1}, 0.0);
  near_end.set_increment(8); // accepted = 0.8
  REQUIRE_THAT(near_end.clip_attempt_dt(1.0), WithinAbs(0.2, TOLERANCE));
}

TEST_CASE("Time - commit_attempt advances by attempted_dt", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 1.0);
  t.set_increment(1); // accepted = 0.5
  const int steps_before = t.get_step_count();

  t.begin_attempt(0.8); // clipped to 0.5 (next saveat)
  const double attempted = t.get_attempted_dt();
  REQUIRE_THAT(attempted, WithinAbs(0.5, TOLERANCE));

  t.commit_attempt();
  REQUIRE_FALSE(t.attempt_active());
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.5 + attempted, TOLERANCE));
  REQUIRE(t.get_step_count() == steps_before + 1);
  REQUIRE(t.get_accepted_steps() == 0); // counters stay caller-owned
}

TEST_CASE("Time - do_save keys off accepted time after reject", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 1.0);
  t.set_increment(1); // accepted = 0.5 — not a save point (saveat=1)
  const bool save_before = t.do_save();
  REQUIRE_FALSE(save_before);

  t.begin_attempt(0.5);
  REQUIRE_FALSE(t.do_save()); // still keyed off accepted time
  t.reject_attempt();
  REQUIRE(t.do_save() == save_before);
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(0.5, TOLERANCE));
}

TEST_CASE("Time - set_dt does not rewrite accepted time", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 0.0);
  t.next();
  t.next();
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(1.0, TOLERANCE));
  REQUIRE(t.get_increment() == 2);

  t.set_dt(0.1);
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(1.0, TOLERANCE));
  REQUIRE_THAT(t.get_current(), WithinAbs(1.0, TOLERANCE));
  REQUIRE(t.get_increment() == 2);

  t.begin_attempt(0.2);
  t.set_dt(0.05);
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(1.0, TOLERANCE));
  t.reject_attempt();
  REQUIRE_THAT(t.get_accepted_time(), WithinAbs(1.0, TOLERANCE));
}

TEST_CASE("TimeStateGuard - restores accepted time and attempt flags after begin_attempt",
          "[time][unit]") {
  Time time({0.0, 10.0, 0.5}, 0.0);
  time.next();
  REQUIRE_THAT(time.get_accepted_time(), WithinAbs(0.5, TOLERANCE));

  {
    TimeStateGuard guard(time);
    time.begin_attempt(0.25);
    REQUIRE(time.attempt_active());
    REQUIRE_THAT(time.get_attempted_dt(), WithinAbs(0.25, TOLERANCE));
  }

  REQUIRE_FALSE(time.attempt_active());
  REQUIRE_THAT(time.get_accepted_time(), WithinAbs(0.5, TOLERANCE));
  REQUIRE(time.get_increment() == 1);
}

TEST_CASE("TimeStateGuard - restores accepted time after commit with attempted_dt != dt",
          "[time][unit]") {
  Time time({0.0, 1.0, 0.5}, 0.0);
  // Pre-guard accepted at 0.0; commit a clipped terminal step with attempted != dt.
  {
    TimeStateGuard guard(time);
    time.begin_attempt(2.0); // clips to remaining 1.0
    REQUIRE_THAT(time.get_attempted_dt(), WithinAbs(1.0, TOLERANCE));
    REQUIRE(time.get_attempted_dt() != time.get_dt());
    time.commit_attempt();
    REQUIRE_THAT(time.get_accepted_time(), WithinAbs(1.0, TOLERANCE));
    REQUIRE(time.get_increment() == 1);
    // Uncommitted: must restore accepted via friend assign after set_increment,
    // not t0 + n * dt reconstruction (which would also be 0 here, so also
    // exercise a mid-span case below).
  }
  REQUIRE_THAT(time.get_accepted_time(), WithinAbs(0.0, TOLERANCE));
  REQUIRE(time.get_increment() == 0);

  // Mid-span: commit clipped saveat step so accepted != t0 + n * dt after set_dt.
  Time mid({0.0, 10.0, 0.5}, 1.0);
  mid.set_increment(1); // accepted = 0.5
  const double pre_guard = mid.get_accepted_time();
  {
    TimeStateGuard guard(mid);
    mid.begin_attempt(0.8); // clips to 0.5 → accepted becomes 1.0 on commit
    mid.commit_attempt();
    REQUIRE_THAT(mid.get_accepted_time(), WithinAbs(1.0, TOLERANCE));
    mid.set_dt(0.2); // if restore used only set_increment, accepted would be
                     // t0 + 2 * 0.2 = 0.4, not the pre-guard 0.5
  }
  REQUIRE_THAT(mid.get_accepted_time(), WithinAbs(pre_guard, TOLERANCE));
  REQUIRE(mid.get_increment() == 1);
  REQUIRE_FALSE(mid.attempt_active());
}

TEST_CASE("Time - attempt API throws on misuse", "[time][unit]") {
  Time t({0.0, 10.0, 0.5}, 0.0);
  REQUIRE_THROWS_AS(t.get_attempted_dt(), std::logic_error);
  REQUIRE_THROWS_AS(t.commit_attempt(), std::logic_error);
  REQUIRE_THROWS_AS(t.reject_attempt(), std::logic_error);

  t.begin_attempt(0.1);
  REQUIRE_THROWS_AS(t.begin_attempt(0.1), std::logic_error);
  t.reject_attempt();
}

TEST_CASE("Time - non-member attempt helpers mirror members", "[time][unit]") {
  Time t({0.0, 5.0, 0.5}, 0.0);
  REQUIRE_THAT(pfc::time::accepted_time(t), WithinAbs(0.0, TOLERANCE));
  REQUIRE_THAT(pfc::time::clip_attempt_dt(t, 10.0), WithinAbs(5.0, TOLERANCE));

  pfc::time::begin_attempt(t, 1.0);
  REQUIRE(pfc::time::attempt_active(t));
  REQUIRE_THAT(pfc::time::attempted_dt(t), WithinAbs(1.0, TOLERANCE));
  pfc::time::commit_attempt(t);
  REQUIRE_THAT(pfc::time::accepted_time(t), WithinAbs(1.0, TOLERANCE));
}
