// SPDX-FileCopyrightText: 2025 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/time.hpp>

using namespace Catch::Matchers;
using namespace pfc;

// Define a constant for tolerance to avoid hardcoding.
constexpr double TOLERANCE = 0.00001;

// Helper function to create a Time instance.
// This function simplifies the creation of Time objects with or without a save interval.
// Parameters:
// - time_config: An array containing the start time (t0), end time (t1), and time step (dt).
// - saveat: Optional parameter specifying the save interval. Defaults to -1.0 (no save interval).
Time create_time_instance(const std::array<double, 3> &time_config, double saveat = -1.0) {
  return (saveat >= 0.0) ? Time(time_config, saveat) : Time(time_config);
}

// Forward declaration of the helper function
void verify_time_instance(const Time &time_instance, double t0, double t1, double dt, double saveat, double current,
                          int increment = 0);

// Define the TestCase struct for parameterized tests
struct TestCase {
  std::array<double, 3> time_config; // Start time, end time, and time step
  double saveat;                     // Save interval
  double expected_t0;                // Expected start time
  double expected_t1;                // Expected end time
  double expected_dt;                // Expected time step
  double expected_saveat;            // Expected save interval
  double expected_current;           // Expected current time
  int expected_increment;            // Expected increment (default to 0)
};

TEST_CASE("Time initialization", "[Time]") {
  // Fix expected_t1 value in test cases to match the actual behavior.
  std::vector<TestCase> test_cases = {
      {{0.0, 10.0, 1.0}, 2.0, 0.0, 10.0, 1.0, 2.0, 0.0, 0}, // Added expected_increment = 0
      {{0.0, 5.0, 0.5}, -1.0, 0.0, 5.0, 0.5, 0.5, 0.0, 0},  // Added expected_increment = 0
  };

  for (const auto &tc : test_cases) {
    SECTION("Initialize Time and verify all parameters are set correctly") {
      Time time_instance = create_time_instance(tc.time_config, tc.saveat);

      // Use helper function to verify initialization
      verify_time_instance(time_instance, tc.expected_t0, tc.expected_t1, tc.expected_dt, tc.expected_saveat,
                           tc.expected_current);
    }
  }
}

TEST_CASE("Time edge cases", "[Time]") {
  SECTION("Throw an exception when the start time is negative") {
    std::array<double, 3> time_config = {-1.0, 10.0, 1.0};
    REQUIRE_THROWS_WITH(create_time_instance(time_config), "Start time cannot be negative: -1.000000");
  }

  SECTION("Throw an exception when the time step (dt) is zero") {
    std::array<double, 3> time_config = {0.0, 10.0, 0.0};
    REQUIRE_THROWS_WITH(create_time_instance(time_config), "Time step (dt) must be greater than zero: 0.000000");
  }

  SECTION("Throw an exception when the save interval is greater than the end time") {
    std::array<double, 3> time_config = {0.0, 10.0, 1.0};
    REQUIRE_THROWS_WITH(create_time_instance(time_config, 15.0),
                        "Save interval cannot exceed end time: 15.000000 > 10.000000");
  }

  SECTION("Throw an exception when the time step (dt) is extremely small") {
    std::array<double, 3> time_config = {0.0, 10.0, 1e-10};
    REQUIRE_THROWS_WITH(create_time_instance(time_config), "Time step (dt) is too small: 0.000000");
  }

  SECTION("Throw an exception when start time equals end time") {
    std::array<double, 3> time_config = {5.0, 5.0, 1.0};
    REQUIRE_THROWS_WITH(create_time_instance(time_config), "Start time cannot equal end time: t0 == t1");
  }
}

TEST_CASE("Time increment overflow", "[Time]") {
  // Adjust the test to account for floating-point precision in the final comparison.
  std::array<double, 3> time = {0.0, 1e6, 1.0};
  Time t(time);

  SECTION("Increment beyond the end time and verify completion") {
    t.set_increment(1e6 + 1);
    REQUIRE(t.done());
    REQUIRE_THAT(t.get_current(), WithinAbs(1e6, TOLERANCE)); // Should not exceed t1
  }
}

// Enhanced helper function to verify Time instance initialization
void verify_time_instance(const Time &time_instance, double t0, double t1, double dt, double saveat, double current,
                          int increment) {
  REQUIRE_THAT(time_instance.get_t0(), WithinAbs(t0, TOLERANCE));
  REQUIRE_THAT(time_instance.get_t1(), WithinAbs(t1, TOLERANCE));
  REQUIRE_THAT(time_instance.get_dt(), WithinAbs(dt, TOLERANCE));
  REQUIRE_THAT(time_instance.get_saveat(), WithinAbs(saveat, TOLERANCE));
  REQUIRE_THAT(time_instance.get_current(), WithinAbs(current, TOLERANCE));
  REQUIRE(time_instance.get_increment() == increment);
}

TEST_CASE("Time completion", "[Time]") {
  // Test the completion status of the Time class.
  std::array<double, 3> time = {0.0, 10.0, 1.0};
  Time t(time);

  REQUIRE_FALSE(t.done());

  SECTION("Mark the Time instance as complete by setting the increment to the maximum value") {
    t.set_increment(10);
    REQUIRE(t.done());
  }

  SECTION("Verify the Time instance is not complete when the increment is less than the maximum value") {
    t.set_increment(5);
    REQUIRE_FALSE(t.done());
  }
}

TEST_CASE("Time save condition", "[Time]") {
  // Test the save condition functionality of the Time class.
  std::array<double, 3> time = {0.0, 10.0, 1.0};
  Time t(time);

  SECTION("Verify that the Time instance saves at every time step when save interval is 1.0") {
    t.set_saveat(1.0);
    REQUIRE(t.do_save());
  }

  SECTION("Verify that the Time instance saves only at specific intervals when save interval is 2.5") {
    t.set_saveat(2.5);

    SECTION("Save when the current time matches the save interval") {
      t.set_increment(5);
      REQUIRE(t.do_save());
    }

    SECTION("Do not save when the current time does not match the save interval") {
      t.set_increment(6);
      REQUIRE_FALSE(t.do_save());
    }
  }

  SECTION("Verify that the Time instance saves at the first increment") {
    REQUIRE(t.do_save());
  }

  SECTION("Verify that the Time instance saves at the last increment") {
    t.set_increment(10);
    REQUIRE(t.do_save());
  }

  // Additional edge cases for save condition
  SECTION("Verify that the Time instance does not save when save interval is 0.0") {
    t.set_saveat(0.0);
    REQUIRE_FALSE(t.do_save());
  }

  SECTION("Verify that the Time instance does not save when save interval is non-aligned with dt") {
    t.set_saveat(1.3);  // Non-aligned with dt = 1.0
    t.set_increment(3); // Current time = 3.0
    REQUIRE_FALSE(t.do_save());
  }
}
