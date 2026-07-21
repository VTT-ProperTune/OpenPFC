// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/embedded_rk.hpp>

using pfc::sim::steppers::EmbeddedRKStepper;
using pfc::sim::steppers::make_embedded_rk23;
using pfc::sim::steppers::make_embedded_rk45;
using pfc::sim::steppers::make_rk4_classical;

namespace {

struct CountingDecayRhs {
  unsigned int *evals{nullptr};
  void operator()(double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) const {
    if (evals != nullptr) {
      ++(*evals);
    }
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -u[i];
    }
  }
};

struct DecayRhs {
  void operator()(double /*t*/, std::vector<double> &u,
                  std::vector<double> &du) const {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -u[i];
    }
  }
};

[[nodiscard]] double max_abs_error(const std::vector<double> &error) {
  double m = 0.0;
  for (double e : error) {
    m = std::max(m, std::fabs(e));
  }
  return m;
}

} // namespace

TEST_CASE("rejects_non_embedded_tableau", "[embedded_rk][unit]") {
  auto tableau = make_rk4_classical<double>();
  REQUIRE_FALSE(tableau.has_embedded());
  unsigned int evals = 0;
  CountingDecayRhs rhs{&evals};
  REQUIRE_THROWS_AS(EmbeddedRKStepper(1, tableau, rhs), std::invalid_argument);
  try {
    EmbeddedRKStepper(1, tableau, rhs);
    FAIL("expected std::invalid_argument");
  } catch (const std::invalid_argument &ex) {
    const std::string msg = ex.what();
    const bool mentions_embedded =
        msg.find("embedded") != std::string::npos ||
        msg.find("has_embedded") != std::string::npos;
    REQUIRE(mentions_embedded);
  }
}

TEST_CASE("accepted_state_unchanged", "[embedded_rk][unit]") {
  auto tableau = make_embedded_rk23<double>();
  DecayRhs rhs;
  EmbeddedRKStepper stepper(3, tableau, rhs);

  std::vector<double> u{1.25, -0.5, 3.0};
  const std::vector<double> fingerprint = u;

  auto result = stepper.attempt(/*t=*/0.0, /*dt=*/0.1, u);
  REQUIRE(result.success);
  REQUIRE(u == fingerprint);
  REQUIRE(result.u_high.size() == u.size());
  REQUIRE(result.u_low.size() == u.size());
  REQUIRE(result.error.size() == u.size());
  // Non-trivial RHS should move candidates away from the accepted state.
  REQUIRE(result.u_high[0] != Catch::Approx(u[0]));
  REQUIRE(result.error[0] != Catch::Approx(0.0));
}

TEST_CASE("rhs_eval_count_equals_stage_count", "[embedded_rk][unit]") {
  SECTION("Bogacki-Shampine 3(2)") {
    auto tableau = make_embedded_rk23<double>();
    unsigned int evals = 0;
    CountingDecayRhs rhs{&evals};
    EmbeddedRKStepper stepper(1, tableau, rhs);
    std::vector<double> u{1.0};
    auto result = stepper.attempt(0.0, 0.05, u);
    REQUIRE(result.success);
    REQUIRE(result.rhs_evals == tableau.stage_count());
    REQUIRE(result.rhs_evals == 4u);
    REQUIRE(evals == 4u);
  }
  SECTION("Dormand-Prince 5(4)") {
    auto tableau = make_embedded_rk45<double>();
    unsigned int evals = 0;
    CountingDecayRhs rhs{&evals};
    EmbeddedRKStepper stepper(1, tableau, rhs);
    std::vector<double> u{1.0};
    auto result = stepper.attempt(0.0, 0.05, u);
    REQUIRE(result.success);
    REQUIRE(result.rhs_evals == tableau.stage_count());
    REQUIRE(result.rhs_evals == 7u);
    REQUIRE(evals == 7u);
  }
}

TEST_CASE("error_norm_decreases_under_dt_refinement", "[embedded_rk][unit]") {
  auto tableau = make_embedded_rk23<double>();
  DecayRhs rhs;
  EmbeddedRKStepper stepper(1, tableau, rhs);
  std::vector<double> u{1.0};

  const double dt = 0.2;
  auto coarse = stepper.attempt(0.0, dt, u);
  REQUIRE(coarse.success);
  const double err_coarse = max_abs_error(coarse.error);

  auto fine = stepper.attempt(0.0, dt / 2.0, u);
  REQUIRE(fine.success);
  const double err_fine = max_abs_error(fine.error);

  REQUIRE(err_fine < err_coarse);
  REQUIRE(err_coarse > 0.0);
}

TEST_CASE("bs32_and_dp54_end_to_end", "[embedded_rk][unit]") {
  DecayRhs rhs;
  std::vector<double> u{1.0, 2.0};

  {
    auto tableau = make_embedded_rk23<double>();
    EmbeddedRKStepper stepper(u.size(), tableau, rhs);
    auto result = stepper.attempt(0.0, 0.1, u);
    REQUIRE(result.success);
    REQUIRE(result.t0 == Catch::Approx(0.0));
    REQUIRE(result.dt == Catch::Approx(0.1));
    REQUIRE(result.t1 == Catch::Approx(0.1));
    REQUIRE(result.u_high.size() == u.size());
    REQUIRE(result.u_low.size() == u.size());
    REQUIRE(result.error.size() == u.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
      REQUIRE(std::isfinite(result.u_high[i]));
      REQUIRE(std::isfinite(result.u_low[i]));
      REQUIRE(std::isfinite(result.error[i]));
      REQUIRE(result.error[i] ==
              Catch::Approx(result.u_high[i] - result.u_low[i]));
    }
  }

  {
    auto tableau = make_embedded_rk45<double>();
    EmbeddedRKStepper stepper(u.size(), tableau, rhs);
    auto result = stepper.attempt(0.0, 0.1, u);
    REQUIRE(result.success);
    REQUIRE(result.rhs_evals == 7u);
    REQUIRE(result.u_high.size() == u.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
      REQUIRE(std::isfinite(result.u_high[i]));
      REQUIRE(std::isfinite(result.u_low[i]));
      REQUIRE(std::isfinite(result.error[i]));
    }
  }
}
