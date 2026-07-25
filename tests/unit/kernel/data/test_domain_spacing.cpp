// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>
#include <openpfc/kernel/data/world.hpp>

using pfc::Bool3;
using pfc::Int3;
using pfc::Real3;
namespace domain = pfc::domain;
namespace world = pfc::world;

TEST_CASE("Domain::from_bounds periodic spacing matches documented formula and world::from_bounds",
          "[domain][unit]") {
    double lower = 0.0;
    double upper = 1.0;
    int size = 10;
    Bool3 periodic = {true, true, true};  // all axes periodic

    auto domain_result = domain::from_bounds({size, size, size}, {lower, lower, lower}, {upper, upper, upper}, periodic);
    auto world_result = world::from_bounds({size, size, size}, {lower, lower, lower}, {upper, upper, upper}, periodic);

    double expected_spacing = (upper - lower) / size;  // periodic formula

    REQUIRE(domain_result.spacing[0] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[1] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[2] == Catch::Approx(expected_spacing));
    
    REQUIRE(world_result.domain_.spacing[0] == Catch::Approx(domain_result.spacing[0]));
    REQUIRE(world_result.domain_.spacing[1] == Catch::Approx(domain_result.spacing[1]));
    REQUIRE(world_result.domain_.spacing[2] == Catch::Approx(domain_result.spacing[2]));
}

TEST_CASE("Domain::from_bounds non-periodic spacing matches documented formula and world::from_bounds",
          "[domain][unit]") {
    double lower = 0.0;
    double upper = 1.0;
    int size = 10;
    Bool3 periodic = {false, false, false};  // all axes non-periodic

    auto domain_result = domain::from_bounds({size, size, size}, {lower, lower, lower}, {upper, upper, upper}, periodic);
    auto world_result = world::from_bounds({size, size, size}, {lower, lower, lower}, {upper, upper, upper}, periodic);

    double expected_spacing = (upper - lower) / (size - 1);  // non-periodic formula

    REQUIRE(domain_result.spacing[0] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[1] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[2] == Catch::Approx(expected_spacing));
    
    REQUIRE(world_result.domain_.spacing[0] == Catch::Approx(domain_result.spacing[0]));
    REQUIRE(world_result.domain_.spacing[1] == Catch::Approx(domain_result.spacing[1]));
    REQUIRE(world_result.domain_.spacing[2] == Catch::Approx(domain_result.spacing[2]));
}
