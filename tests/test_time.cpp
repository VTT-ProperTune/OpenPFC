#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/time.hpp>

using namespace Catch::Matchers;
using namespace pfc;

TEST_CASE("Time initialization", "[Time]") {
  SECTION("Initialize with save interval") {
    std::array<double, 3> time = {0.0, 10.0, 1.0};
    double saveat = 2.0;
    Time t(time, saveat);

    REQUIRE_THAT(t.get_t0(), WithinAbs(0.0, 0.00001));
    REQUIRE_THAT(t.get_t1(), WithinAbs(10.0, 0.00001));
    REQUIRE_THAT(t.get_dt(), WithinAbs(1.0, 0.00001));
    REQUIRE(t.get_increment() == 0);
    REQUIRE_THAT(t.get_saveat(), WithinAbs(2.0, 0.00001));
    REQUIRE_THAT(t.get_current(), WithinAbs(0.0, 0.00001));
  }

  SECTION("Initialize without save interval") {
    std::array<double, 3> time = {0.0, 5.0, 0.5};
    Time t(time);

    REQUIRE_THAT(t.get_t0(), WithinAbs(0.0, 0.00001));
    REQUIRE_THAT(t.get_t1(), WithinAbs(5.0, 0.00001));
    REQUIRE_THAT(t.get_dt(), WithinAbs(0.5, 0.00001));
    REQUIRE(t.get_increment() == 0);
    REQUIRE_THAT(t.get_saveat(), WithinAbs(0.5, 0.00001));
    REQUIRE_THAT(t.get_current(), WithinAbs(0.0, 0.00001));
  }
}

TEST_CASE("Time increment", "[Time]") {
  std::array<double, 3> time = {0.0, 10.0, 1.0};
  Time t(time);

  REQUIRE(t.get_increment() == 0);

  SECTION("Increment by 1") {
    t.next();
    REQUIRE(t.get_increment() == 1);
    REQUIRE_THAT(t.get_current(), WithinAbs(1.0, 0.00001));
  }

  SECTION("Increment by 5") {
    t.set_increment(5);
    REQUIRE(t.get_increment() == 5);
    REQUIRE_THAT(t.get_current(), WithinAbs(5.0, 0.00001));
  }
}

TEST_CASE("Time completion", "[Time]") {
  std::array<double, 3> time = {0.0, 10.0, 1.0};
  Time t(time);

  REQUIRE_FALSE(t.done());

  SECTION("Complete time interval") {
    t.set_increment(10);
    REQUIRE(t.done());
  }

  SECTION("Incomplete time interval") {
    t.set_increment(5);
    REQUIRE_FALSE(t.done());
  }
}

TEST_CASE("Time save condition", "[Time]") {
  std::array<double, 3> time = {0.0, 10.0, 1.0};
  Time t(time);

  SECTION("Save at each time step") {
    t.set_saveat(1.0);
    REQUIRE(t.do_save());
  }

  SECTION("Save at specific intervals") {
    t.set_saveat(2.5);

    SECTION("Save at current time") {
      t.set_increment(5);
      REQUIRE(t.do_save());
    }

    SECTION("Do not save at current time") {
      t.set_increment(6);
      REQUIRE_FALSE(t.do_save());
    }
  }

  SECTION("Save at the first increment") {
    REQUIRE(t.do_save());
  }

  SECTION("Save at the last increment") {
    t.set_increment(10);
    REQUIRE(t.do_save());
  }
}
