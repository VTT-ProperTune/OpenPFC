// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <openpfc/kernel/data/domain.hpp>

using pfc::Bool3;
using pfc::Int3;
using pfc::Real3;
namespace domain = pfc::domain;

TEST_CASE("Domain::from_bounds periodic spacing matches documented formula",
          "[domain][unit]") {
    double lower = 0.0;
    double upper = 1.0;
    int size = 10;
    Bool3 periodic = {true, true, true};

    auto domain_result = domain::from_bounds({size, size, size}, {lower, lower, lower},
                                             {upper, upper, upper}, periodic);

    double expected_spacing = (upper - lower) / size;

    REQUIRE(domain_result.spacing[0] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[1] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[2] == Catch::Approx(expected_spacing));
}

TEST_CASE("Domain::from_bounds non-periodic spacing matches documented formula",
          "[domain][unit]") {
    double lower = 0.0;
    double upper = 1.0;
    int size = 10;
    Bool3 periodic = {false, false, false};

    auto domain_result = domain::from_bounds({size, size, size}, {lower, lower, lower},
                                             {upper, upper, upper}, periodic);

    double expected_spacing = (upper - lower) / (size - 1);

    REQUIRE(domain_result.spacing[0] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[1] == Catch::Approx(expected_spacing));
    REQUIRE(domain_result.spacing[2] == Catch::Approx(expected_spacing));
}
