#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openpfc/world.hpp>

using namespace Catch::Matchers;

TEST_CASE("World - Construction and Accessors", "[world]") {
  // Test case for World construction and accessors

  SECTION("Construct World with dimensions, origin, and discretization") {
    std::array<int, 3> dimensions = {100, 200, 300};
    std::array<double, 3> origin = {1.0, 2.0, 3.0};
    std::array<double, 3> discretization = {0.1, 0.2, 0.3};

    pfc::World world(dimensions, origin, discretization);

    // Check the dimensions
    REQUIRE(world.get_Lx() == dimensions[0]);
    REQUIRE(world.get_Ly() == dimensions[1]);
    REQUIRE(world.get_Lz() == dimensions[2]);

    // Check the origin coordinates
    REQUIRE(world.get_x0() == origin[0]);
    REQUIRE(world.get_y0() == origin[1]);
    REQUIRE(world.get_z0() == origin[2]);

    // Check the discretization parameters
    REQUIRE(world.get_dx() == discretization[0]);
    REQUIRE(world.get_dy() == discretization[1]);
    REQUIRE(world.get_dz() == discretization[2]);

    // Check the size
    std::array<int, 3> size = world.get_size();
    REQUIRE(size == dimensions);
  }

  SECTION("Construct World with dimensions (default origin and discretization)") {
    std::array<int, 3> dimensions = {100, 200, 300};
    pfc::World world(dimensions);

    // Check the dimensions
    REQUIRE(world.get_Lx() == dimensions[0]);
    REQUIRE(world.get_Ly() == dimensions[1]);
    REQUIRE(world.get_Lz() == dimensions[2]);

    // Check the default origin coordinates (0.0, 0.0, 0.0)
    REQUIRE_THAT(world.get_x0(), WithinAbs(0.0, 0.00001));
    REQUIRE_THAT(world.get_y0(), WithinAbs(0.0, 0.00001));
    REQUIRE_THAT(world.get_z0(), WithinAbs(0.0, 0.00001));

    // Check the default discretization parameters (1.0, 1.0, 1.0)
    REQUIRE_THAT(world.get_dx(), WithinAbs(1.0, 0.00001));
    REQUIRE_THAT(world.get_dy(), WithinAbs(1.0, 0.00001));
    REQUIRE_THAT(world.get_dz(), WithinAbs(1.0, 0.00001));

    // Check the size
    std::array<int, 3> size = world.get_size();
    REQUIRE(size == dimensions);
  }
}

TEST_CASE("World - Conversion to heffte::box3d<int>", "[world]") {
  // Test case for World conversion to heffte::box3d<int>

  std::array<int, 3> dimensions = {100, 200, 300};
  pfc::World world(dimensions);

  SECTION("Conversion to heffte::box3d<int>") {
    heffte::box3d<int> box = world;

    // Check the box dimensions
    REQUIRE(box.size[0] == dimensions[0]);
    REQUIRE(box.size[1] == dimensions[1]);
    REQUIRE(box.size[2] == dimensions[2]);
  }
}

TEST_CASE("World - Invalid Construction", "[world]") {
  // Test case for invalid World construction

  SECTION("Construct World with invalid dimensions") {
    std::array<int, 3> dimensions = {-100, 200, 300};
    REQUIRE_THROWS_AS(pfc::World(dimensions), std::invalid_argument);
  }

  SECTION("Construct World with invalid discretization") {
    std::array<int, 3> dimensions = {100, 200, 300};
    std::array<double, 3> discretization = {0.1, 0.2, -0.3};
    REQUIRE_THROWS_AS(pfc::World(dimensions, {}, discretization), std::invalid_argument);
  }
}
