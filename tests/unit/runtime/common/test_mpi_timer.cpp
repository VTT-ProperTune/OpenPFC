// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <chrono>
#include <sstream>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mpi.h>

#include <openpfc/runtime/common/mpi_timer.hpp>

using pfc::runtime::MpiTimer;
using pfc::runtime::print_timing_summary;
using pfc::runtime::tic;
using pfc::runtime::toc;

namespace {

inline void busy_sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

} // namespace

TEST_CASE("MpiTimer: unlabeled tic/toc returns nonneg elapsed", "[runtime][timer]") {
  MpiTimer timer{MPI_COMM_SELF};
  tic(timer);
  busy_sleep_ms(2);
  const double elapsed = toc(timer);

  REQUIRE(elapsed >= 0.0);
  REQUIRE(elapsed < 1.0);
}

TEST_CASE("MpiTimer: labeled tic/toc accumulates per section", "[runtime][timer]") {
  MpiTimer timer{MPI_COMM_SELF};

  tic(timer, "alpha");
  busy_sleep_ms(2);
  const double a1 = toc(timer, "alpha");

  tic(timer, "beta");
  busy_sleep_ms(1);
  const double b1 = toc(timer, "beta");

  tic(timer, "alpha");
  busy_sleep_ms(2);
  const double a2 = toc(timer, "alpha");

  REQUIRE(a1 >= 0.0);
  REQUIRE(a2 >= 0.0);
  REQUIRE(b1 >= 0.0);

  REQUIRE(timer.sections.count("alpha") == 1);
  REQUIRE(timer.sections.count("beta") == 1);

  const double alpha_total = timer.sections.at("alpha").second;
  const double beta_total = timer.sections.at("beta").second;

  REQUIRE(alpha_total >= a1 + a2 - 1e-9);
  REQUIRE(beta_total >= b1 - 1e-9);
  REQUIRE(alpha_total > beta_total);
}

TEST_CASE("MpiTimer: labeled overloads do not touch unlabeled t_start",
          "[runtime][timer]") {
  MpiTimer timer{MPI_COMM_SELF};
  tic(timer);
  const double saved = timer.t_start;
  tic(timer, "section");
  busy_sleep_ms(1);
  toc(timer, "section");

  REQUIRE(timer.t_start == saved);
}

TEST_CASE("MpiTimer: print_timing_summary returns sorted (label, max) pairs",
          "[runtime][timer]") {
  MpiTimer timer{MPI_COMM_SELF};

  tic(timer, "small");
  busy_sleep_ms(1);
  toc(timer, "small");

  tic(timer, "big");
  busy_sleep_ms(5);
  toc(timer, "big");

  std::ostringstream oss;
  const auto summary = print_timing_summary(timer, /*print_rank=*/0, oss);

  REQUIRE(summary.size() == 2);
  REQUIRE(summary[0].first == "big");
  REQUIRE(summary[1].first == "small");
  REQUIRE(summary[0].second >= summary[1].second);

  const std::string out = oss.str();
  REQUIRE(out.find("section breakdown") != std::string::npos);
  REQUIRE(out.find("big") != std::string::npos);
  REQUIRE(out.find("small") != std::string::npos);
}

TEST_CASE("MpiTimer: print_timing_summary on empty timer prints nothing",
          "[runtime][timer]") {
  MpiTimer timer{MPI_COMM_SELF};
  std::ostringstream oss;
  const auto summary = print_timing_summary(timer, 0, oss);
  REQUIRE(summary.empty());
  REQUIRE(oss.str().empty());
}
