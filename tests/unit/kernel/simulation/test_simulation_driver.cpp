// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/simulation/simulation_driver.hpp>
#include <openpfc/kernel/simulation/simulation_state.hpp>
#include <openpfc/kernel/simulation/time.hpp>

using Catch::Matchers::WithinAbs;
using pfc::sim::SimulationDriver;

TEST_CASE("sim::run matches Simulator step ordering", "[simulation_driver][unit]") {
  pfc::Time time({0.0, 0.05, 0.01}, 0.02);
  int nstart = 0;
  int napply = 0;
  int nstep = 0;
  int nsave = 0;
  double last_t = -1.0;

  pfc::sim::run(
      time,
      [&](double t) {
        ++nstep;
        last_t = t;
      },
      [&](pfc::Time &) { ++nstart; }, [&](pfc::Time &) { ++napply; },
      [&](const pfc::Time &) { ++nsave; });

  REQUIRE(nstart == 1);
  REQUIRE(nstep == 5);
  REQUIRE(napply == 5);
  REQUIRE(napply == nstep);
  REQUIRE_THAT(last_t, WithinAbs(0.05, 1e-12));
  REQUIRE(pfc::time::done(time));
  // t=0 (increment 0), 0.02, 0.04, and t1 via done().
  REQUIRE(nsave == 4);
}

TEST_CASE("SimulationDriver holds Time and optional SimulationState",
          "[simulation_driver][unit]") {
  pfc::Time time({0.0, 0.02, 0.01}, 0.0);
  pfc::SimulationState state;
  SimulationDriver driver(time, &state);
  REQUIRE(&driver.time() == &time);
  REQUIRE(driver.state() == &state);

  int nstep = 0;
  driver.run([&](double) { ++nstep; });
  REQUIRE(nstep == 2);
  REQUIRE(pfc::time::done(driver.time()));
}
