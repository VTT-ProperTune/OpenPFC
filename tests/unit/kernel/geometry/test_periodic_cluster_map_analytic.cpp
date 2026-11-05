// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

// This translation unit must link when FFTW3 is disabled. It intentionally
// tests only the analytic map, Jacobian and coordinate formula.
#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/geometry/periodic_cluster_map.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>

TEST_CASE("the periodic coordinate map has a positive periodic Jacobian",
          "[geometry][cluster][analytic]") {
  using namespace pfc::geometry;
  REQUIRE_THROWS_AS(periodic_cluster_map(0.0, 0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_cluster_map(1.0, 1.0), std::invalid_argument);
  REQUIRE_THROWS_AS(periodic_cluster_map(1.0, -1.0), std::invalid_argument);
  for (double a : {-0.8, 0.0, 0.6}) {
    const auto map = periodic_cluster_map(2.0, a);
    const double h = map.period / 32.0;
    for (int i = 0; i < 32; ++i) {
      const double xi = cluster_computational(map, i, 32);
      const double jac = cluster_jacobian(map, xi);
      REQUIRE(jac > 0.0);
      REQUIRE(std::abs(cluster_jacobian(map, xi + map.period) - jac) < 1e-12);
      REQUIRE(std::abs(cluster_physical(map, xi + map.period) -
                       cluster_physical(map, xi) - map.period) < 1e-12);
    }
    // The map's periodic Jacobian integrates to its original period.
    double jacobian_sum = 0.0;
    for (int i = 0; i < 32; ++i) {
      jacobian_sum += cluster_jacobian(map, i * h);
    }
    REQUIRE(std::abs(h * jacobian_sum - map.period) < 1e-12);
  }
}
