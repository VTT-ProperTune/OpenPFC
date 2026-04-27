// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <openpfc/kernel/mpi/timer.hpp>

#include <stdexcept>

TEST_CASE("mpi timer rejects toc without active lap", "[mpi][timer]") {
  pfc::mpi::timer t;

  REQUIRE_THROWS_AS(t.toc(), std::logic_error);

  t.tic();
  [[maybe_unused]] const double elapsed = t.toc();
  REQUIRE_THROWS_AS(t.toc(), std::logic_error);
}

TEST_CASE("mpi timer reset clears duration and active lap", "[mpi][timer]") {
  pfc::mpi::timer t;

  t.tic();
  t.reset();

  REQUIRE(t.duration() == 0.0);
  REQUIRE_THROWS_AS(t.toc(), std::logic_error);
}
