// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <burgers/model.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;

TEST_CASE("Burgers rejects invalid grid sizes before allocating",
          "[burgers][no_fft]") {
  REQUIRE_THROWS_AS(burgers::Simulation(burgers::Method::fvm, -1, 0.1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(burgers::Simulation(burgers::Method::fdm2, 0, 0.1),
                    std::invalid_argument);
}

TEST_CASE("Burgers FD2 and FVM periodic tendencies conserve mass",
          "[burgers][no_fft]") {
  for (const auto method : {burgers::Method::fdm2, burgers::Method::fvm}) {
    burgers::Simulation s(method, 64, 0.1);
    s.set_initial_sine();
    std::vector<double> rhs(64);
    s.rate(s.state(), rhs);
    double sum = 0.0;
    for (double value : rhs) sum += value;
    REQUIRE_THAT(s.spacing() * sum, WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("Burgers rejects a viscous CFL violation without changing state",
          "[burgers][no_fft]") {
  burgers::Simulation s(burgers::Method::fvm, 64, 10.0);
  s.set_initial_sine();
  const auto before = s.state();
  const double bound = s.stability_limit();
  REQUIRE(std::isfinite(bound));
  REQUIRE(bound > 0.0);
  REQUIRE_THROWS_AS(s.advance(2.0 * bound, 1), std::invalid_argument);
  REQUIRE(s.state() == before);
}
