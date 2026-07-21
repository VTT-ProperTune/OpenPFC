// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <openpfc/kernel/simulation/steppers/integrator_base.hpp>

#include <cmath>
#include <memory>
#include <vector>

using namespace pfc::sim::steppers;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

RhsFunction make_decay_rhs(double lambda) {
  return [lambda](double /*t*/, const std::vector<double> &u,
                  std::vector<double> &du) {
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = -lambda * u[i];
    }
  };
}

std::vector<double> decay_analytic(const std::vector<double> &u0, double lambda,
                                   double t) {
  std::vector<double> out = u0;
  for (double &v : out) {
    v *= std::exp(-lambda * t);
  }
  return out;
}

} // namespace

TEST_CASE("EulerIntegrator via IntegratorBase pointer", "[integrator_base]") {
  constexpr double dt = 0.01;
  constexpr double lambda = 1.0;
  constexpr std::size_t n = 3;
  constexpr double t_final = 0.1;
  const int steps = static_cast<int>(t_final / dt);
  std::vector<double> u0{1.0, 2.0, 3.0};

  SECTION("construction and order") {
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<EulerIntegrator>(make_decay_rhs(lambda), dt, n);
    REQUIRE(integ->get_order() == 1);
  }

  SECTION("step counting") {
    auto u = u0;
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<EulerIntegrator>(make_decay_rhs(lambda), dt, n);
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
      t = integ->step(t, u);
    }
    REQUIRE_THAT(t, WithinAbs(t_final, 1e-12));
    REQUIRE(integ->get_accepted_steps() == static_cast<std::size_t>(steps));
    REQUIRE(integ->get_rejected_steps() == 0);
  }

  SECTION("rollback restores t and u") {
    auto u = u0;
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<EulerIntegrator>(make_decay_rhs(lambda), dt, n);
    double t = 0.0;
    const auto u_before = u;
    const double t_before = t;
    t = integ->step(t, u);
    integ->rollback(t, u);
    REQUIRE_THAT(t, WithinAbs(t_before, 1e-15));
    for (std::size_t i = 0; i < u.size(); ++i) {
      REQUIRE_THAT(u[i], WithinAbs(u_before[i], 1e-15));
    }
    REQUIRE(integ->get_rejected_steps() == 1);
  }

  SECTION("accuracy vs analytic") {
    auto u = u0;
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<EulerIntegrator>(make_decay_rhs(lambda), dt, n);
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
      t = integ->step(t, u);
    }
    const auto analytic = decay_analytic(u0, lambda, t_final);
    for (std::size_t i = 0; i < u.size(); ++i) {
      REQUIRE_THAT(u[i], WithinRel(analytic[i], 0.1));
    }
  }
}

TEST_CASE("RK2HeunIntegrator via IntegratorBase pointer",
          "[integrator_base]") {
  constexpr double dt = 0.01;
  constexpr double lambda = 1.0;
  constexpr std::size_t n = 3;
  constexpr double t_final = 0.1;
  const int steps = static_cast<int>(t_final / dt);
  std::vector<double> u0{1.0, 2.0, 3.0};

  SECTION("construction and order") {
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<RK2HeunIntegrator>(make_decay_rhs(lambda), dt, n);
    REQUIRE(integ->get_order() == 2);
  }

  SECTION("step counting") {
    auto u = u0;
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<RK2HeunIntegrator>(make_decay_rhs(lambda), dt, n);
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
      t = integ->step(t, u);
    }
    REQUIRE_THAT(t, WithinAbs(t_final, 1e-12));
    REQUIRE(integ->get_accepted_steps() == static_cast<std::size_t>(steps));
  }

  SECTION("rollback restores t and u") {
    auto u = u0;
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<RK2HeunIntegrator>(make_decay_rhs(lambda), dt, n);
    double t = 0.0;
    const auto u_before = u;
    t = integ->step(t, u);
    integ->rollback(t, u);
    REQUIRE_THAT(t, WithinAbs(0.0, 1e-15));
    for (std::size_t i = 0; i < u.size(); ++i) {
      REQUIRE_THAT(u[i], WithinAbs(u_before[i], 1e-15));
    }
    REQUIRE(integ->get_rejected_steps() == 1);
  }

  SECTION("accuracy vs analytic") {
    auto u = u0;
    std::unique_ptr<IntegratorBase> integ =
        std::make_unique<RK2HeunIntegrator>(make_decay_rhs(lambda), dt, n);
    double t = 0.0;
    for (int i = 0; i < steps; ++i) {
      t = integ->step(t, u);
    }
    const auto analytic = decay_analytic(u0, lambda, t_final);
    for (std::size_t i = 0; i < u.size(); ++i) {
      REQUIRE_THAT(u[i], WithinRel(analytic[i], 0.01));
    }
  }
}

TEST_CASE("Polymorphic integrator swapping", "[integrator_base]") {
  constexpr double dt = 0.01;
  constexpr double lambda = 1.0;
  constexpr std::size_t n = 3;
  constexpr double t_final = 0.1;
  const int steps = static_cast<int>(t_final / dt);
  std::vector<double> u_euler{1.0, 2.0, 3.0};
  std::vector<double> u_rk2 = u_euler;

  std::unique_ptr<IntegratorBase> integ =
      std::make_unique<EulerIntegrator>(make_decay_rhs(lambda), dt, n);
  double t = 0.0;
  for (int i = 0; i < steps; ++i) {
    t = integ->step(t, u_euler);
  }

  integ = std::make_unique<RK2HeunIntegrator>(make_decay_rhs(lambda), dt, n);
  t = 0.0;
  for (int i = 0; i < steps; ++i) {
    t = integ->step(t, u_rk2);
  }

  const auto analytic = decay_analytic({1.0, 2.0, 3.0}, lambda, t_final);
  for (std::size_t i = 0; i < n; ++i) {
    const double e_err = std::abs(u_euler[i] - analytic[i]);
    const double r_err = std::abs(u_rk2[i] - analytic[i]);
    REQUIRE(r_err < e_err);
  }
}
