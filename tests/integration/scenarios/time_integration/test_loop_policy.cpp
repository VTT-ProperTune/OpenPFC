// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_loop_policy.cpp
 * @brief Serial and parallel for_each_interior agree on a small grid.
 *
 * OpenPFC #423. The serial policy must not depend on omp_set_num_threads.
 */

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include <openpfc/kernel/simulation/for_each_interior.hpp>

namespace {

struct Sample {
  double value = 0.0;
};

struct GridEval {
  int n = 0;
  const std::vector<double> *u = nullptr;
  void prepare() {}
  int imin() const { return 0; }
  int imax() const { return n; }
  int jmin() const { return 0; }
  int jmax() const { return n; }
  int kmin() const { return 0; }
  int kmax() const { return 1; }
  std::size_t idx(int i, int j, int) const {
    return static_cast<std::size_t>(i + j * n);
  }
  Sample operator()(int i, int j, int k) const { return {(*u)[idx(i, j, k)]}; }
};

struct Decay {
  double rhs(double, const Sample &g) const { return -g.value; }
};

} // namespace

TEST_CASE("for_each_interior serial matches parallel on a small decay",
          "[loop-policy]") {
  constexpr int n = 8;
  std::vector<double> u(static_cast<std::size_t>(n * n));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      u[static_cast<std::size_t>(i + j * n)] = 0.1 * i - 0.05 * j;

  GridEval eval{n, &u};
  const Decay model;
  std::vector<double> serial(u.size(), 1.0);
  std::vector<double> parallel(u.size(), 1.0);
  pfc::sim::for_each_interior(model, eval, serial.data(), 0.0,
                              pfc::sim::LoopPolicy::Serial);
  pfc::sim::for_each_interior(model, eval, parallel.data(), 0.0,
                              pfc::sim::LoopPolicy::Parallel);

  for (std::size_t q = 0; q < u.size(); ++q) {
    REQUIRE(serial[q] == parallel[q]);
    REQUIRE(serial[q] == -u[q]);
  }
}
