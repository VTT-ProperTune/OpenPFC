// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

/**
 * @file test_multifield_stepper.cpp
 * @brief Multi-field contract validation for MultiEulerStepper.
 *
 * @details
 * Validates MultiEulerStepper<Rhs, 2>'s public interface (constructor and
 * step()) against a simple two-field harmonic rotation system:
 *
 *     du/dt =  v
 *     dv/dt = -u
 *
 * whose exact solution is a pure rotation:
 *
 *     u(t) = u0*cos(t) + v0*sin(t)
 *     v(t) = v0*cos(t) - u0*sin(t)
 *
 * This is a compact, dependency-free stand-in for a real two-field model
 * (no Field/gradient machinery needed) chosen so the test can focus purely
 * on the stepper's tuple-protocol contract: field ordering, simultaneous
 * (not sequential) updates, and cross-field coupling (each field's RHS
 * reads the *other* field's pre-step value). The same rhs/step structure
 * is the template other multi-field steppers (RK2/RK4) can be validated
 * against by swapping in their own stepper class.
 */

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <tuple>
#include <vector>

#include <openpfc/kernel/simulation/steppers/euler.hpp>

namespace {

// rhs(t, u_pack, du_pack) fills du_pack in place from the *pre-step* values
// in u_pack -- this is what makes the update simultaneous rather than
// sequential: du/u is computed entirely before either field is mutated.
struct RotationRhs {
  void operator()(
      double /*t*/, std::tuple<std::vector<double> &, std::vector<double> &> u_pack,
      std::tuple<std::vector<double> &, std::vector<double> &> du_pack) const {
    const auto &u = std::get<0>(u_pack);
    const auto &v = std::get<1>(u_pack);
    auto &du = std::get<0>(du_pack);
    auto &dv = std::get<1>(du_pack);
    for (std::size_t i = 0; i < u.size(); ++i) {
      du[i] = v[i];
      dv[i] = -u[i];
    }
  }
};

} // namespace

TEST_CASE("MultiEulerStepper validates multi-field contract",
          "[integration][multifield]") {
  const double dt = 0.001;
  const double u0 = 1.0;
  const double v0 = 0.25;

  std::vector<double> u{u0};
  std::vector<double> v{v0};

  pfc::sim::steppers::MultiEulerStepper<RotationRhs, 2> stepper(dt, {1, 1},
                                                                RotationRhs{});

  SECTION("field tuple ordering is preserved across steps") {
    // Field 0 (u) and field 1 (v) must each keep receiving their own
    // increment -- not swapped -- across repeated calls to step().
    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
      t = stepper.step(t, u, v);
    }
    // u started positive and only mixes with v through -u/+v coupling;
    // if the tuple order were swapped, u and v would diverge from the
    // exact rotation below far more than the Euler discretization error.
    const double u_exact = u0 * std::cos(t) + v0 * std::sin(t);
    const double v_exact = v0 * std::cos(t) - u0 * std::sin(t);
    REQUIRE(std::abs(u[0] - u_exact) < 1e-5);
    REQUIRE(std::abs(v[0] - v_exact) < 1e-5);
  }

  SECTION("both fields update simultaneously, not sequentially") {
    // A sequential (Gauss-Seidel-style) update would use the *already
    // updated* u when computing dv, giving a different result than the
    // simultaneous update the contract requires. Check one step by hand.
    double t = stepper.step(0.0, u, v);
    REQUIRE(t == 0.001);
    const double expected_u = u0 + dt * v0;
    const double expected_v = v0 - dt * u0; // uses pre-step u0, not expected_u
    REQUIRE(std::abs(u[0] - expected_u) < 1e-12);
    REQUIRE(std::abs(v[0] - expected_v) < 1e-12);
  }

  SECTION("cross-field coupling is evaluated at a consistent stage") {
    // Every step's rhs must read both fields' values from the *same*
    // instant (the start of that step) -- run two steps and verify the
    // second step's increments are consistent with rhs applied to the
    // state left by the first step, not some stale or partially-updated
    // mixture.
    double t = stepper.step(0.0, u, v);
    const double u_after_1 = u[0];
    const double v_after_1 = v[0];
    t = stepper.step(t, u, v);
    const double expected_u2 = u_after_1 + dt * v_after_1;
    const double expected_v2 = v_after_1 - dt * u_after_1;
    REQUIRE(std::abs(u[0] - expected_u2) < 1e-12);
    REQUIRE(std::abs(v[0] - expected_v2) < 1e-12);
  }

  SECTION("ten steps against the analytical rotation solution") {
    double t = 0.0;
    for (int i = 0; i < 10; ++i) {
      t = stepper.step(t, u, v);
    }
    const double u_exact = u0 * std::cos(t) + v0 * std::sin(t);
    const double v_exact = v0 * std::cos(t) - u0 * std::sin(t);
    // Measured forward-Euler error at dt=0.001, 10 steps is ~5e-6; 1e-5
    // gives comfortable margin without masking a real regression.
    REQUIRE(std::abs(u[0] - u_exact) < 1e-5);
    REQUIRE(std::abs(v[0] - v_exact) < 1e-5);
  }
}
