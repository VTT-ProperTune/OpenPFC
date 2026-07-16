// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>
#include <openpfc/kernel/simulation/time.hpp>

#include <openpfc/kernel/data/world.hpp>
#include <openpfc/kernel/decomposition/decomposition.hpp>
#include <openpfc/kernel/field/local_field.hpp>
#include <openpfc/kernel/simulation/steppers/euler.hpp>

using namespace pfc::sim::steppers;
using Catch::Approx;

TEST_CASE("EulerStepper accumulates correctly with constant RHS",
          "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  std::vector<double> u(n, 0.0);

  // Mock RHS: fills du[i] = 2.0 for all i
  auto rhs = [](double /*t*/, std::vector<double> & /*u*/, std::vector<double> &du) {
    for (std::size_t i = 0; i < du.size(); ++i) {
      du[i] = 2.0;
    }
  };

  EulerStepper stepper(dt, n, rhs);

  SECTION("First step: u[i] should be dt * du[i] = 0.1 * 2.0 = 0.2") {
    stepper.step(0.0, u);
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u[i] == Approx(dt * 2.0)); // 0.2
    }
  }

  SECTION("Second step: u[i] should accumulate to 2 * dt * du[i] = 0.4") {
    stepper.step(0.0, u); // t=0.0, u becomes [0.2, 0.2, ...]
    stepper.step(dt, u);  // t=0.1, u becomes [0.4, 0.4, ...]
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u[i] == Approx(2.0 * dt * 2.0)); // 0.4
    }
  }
}

TEST_CASE("EulerStepper returns correct new time", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 5;
  std::vector<double> u(n, 0.0);

  // RHS that does nothing (focus on time return)
  auto dummy_rhs = [](double /*t*/, std::vector<double> & /*u*/,
                      std::vector<double> &du) {
    std::fill(du.begin(), du.end(), 0.0);
  };

  EulerStepper stepper(dt, n, dummy_rhs);

  double t = 0.0;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.0 -> 0.1

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.1 -> 0.2

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.2 -> 0.3

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // t=0.3 -> 0.4
}

TEST_CASE("EulerStepper reuses du buffer across steps", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  std::vector<double> u(n, 0.0);

  // Functor-based RHS that only writes to one index per step
  struct SelectiveRhs {
    std::size_t step_counter = 0;
    void operator()(double /*t*/, std::vector<double> & /*u*/,
                    std::vector<double> &du) {
      // Write to index 0 on even steps, index 1 on odd steps
      std::size_t write_idx = (step_counter % 2 == 0) ? 0 : 1;
      for (std::size_t i = 0; i < du.size(); ++i) {
        du[i] = (i == write_idx) ? 1.0 : 0.0; // Only one index gets 1.0
      }
      ++step_counter;
    }
  };

  SelectiveRhs rhs;
  EulerStepper stepper(dt, n, rhs);

  SECTION("Step 0: only du[0] written, others preserve 0.0 value-init") {
    stepper.step(0.0, u);
    REQUIRE(u[0] == Approx(dt * 1.0)); // Only index 0 advanced
    for (std::size_t i = 1; i < n; ++i) {
      REQUIRE(u[i] == Approx(0.0)); // Others stay at 0.0
    }
  }

  SECTION("Step 1: only du[1] written, du[0] from previous step ignored in "
          "accumulation") {
    stepper.step(0.0, u);              // Step 0: writes du[0]=1.0, others 0.0
    stepper.step(dt, u);               // Step 1: writes du[1]=1.0, others 0.0
    REQUIRE(u[0] == Approx(dt * 1.0)); // Only advanced once
    REQUIRE(u[1] == Approx(dt * 1.0)); // Advanced on step 1
    for (std::size_t i = 2; i < n; ++i) {
      REQUIRE(u[i] == Approx(0.0)); // Never advanced
    }
  }
}

TEST_CASE("MultiEulerStepper updates fields independently", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  constexpr std::size_t N = 2;
  std::vector<double> u1(n, 0.0); // First field
  std::vector<double> u2(n, 0.0); // Second field

  // RHS that fills du1[i]=1.0, du2[i]=2.0 for all i
  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
    }
  };

  std::array<std::size_t, N> local_sizes = {n, n};
  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);

  SECTION("First step: each field accumulates independently") {
    stepper.step(0.0, u1, u2);
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u1[i] == Approx(dt * 1.0)); // 0.1
      REQUIRE(u2[i] == Approx(dt * 2.0)); // 0.2
    }
  }

  SECTION("Second step: both fields accumulate independently again") {
    stepper.step(0.0, u1, u2); // t=0.0
    stepper.step(dt, u1, u2);  // t=0.1
    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE(u1[i] == Approx(2.0 * dt * 1.0)); // 0.2
      REQUIRE(u2[i] == Approx(2.0 * dt * 2.0)); // 0.4
    }
  }
}

#include <openpfc/kernel/simulation/steppers/butcher_tableau.hpp>
#include <openpfc/kernel/simulation/steppers/explicit_rk.hpp>

TEST_CASE("ExplicitRKStepper accumulates correctly with constant RHS (RK4)",
          "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  std::vector<double> u(n, 0.0);

  auto rhs = [](double /*t*/, std::vector<double> & /*u*/, std::vector<double> &du) {
    for (std::size_t i = 0; i < du.size(); ++i) {
      du[i] = 2.0;
    }
  };

  auto tableau = make_rk4_classical<double>();
  ExplicitRKStepper stepper(dt, n, tableau, rhs);

  // Verify RK4: u += dt * sum(b_i * k_i) where k_i = 2.0 for all stages
  // sum(b_i) = 1/6 + 1/3 + 1/3 + 1/6 = 1.0 for classical RK4
  stepper.step(0.0, u);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u[i] == Approx(dt * 2.0)); // 0.2
  }

  stepper.step(dt, u);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u[i] == Approx(2.0 * dt * 2.0)); // 0.4
  }
}

TEST_CASE("ExplicitRKStepper returns correct new time (RK2 midpoint)",
          "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 5;
  std::vector<double> u(n, 0.0);

  auto dummy_rhs = [](double /*t*/, std::vector<double> & /*u*/,
                      std::vector<double> &du) {
    std::fill(du.begin(), du.end(), 0.0);
  };

  auto tableau = make_rk2_midpoint<double>();
  ExplicitRKStepper stepper(dt, n, tableau, dummy_rhs);

  double t = 0.0;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // 0.0 -> 0.1

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // 0.1 -> 0.2

  t += dt;
  REQUIRE(stepper.step(t, u) == Approx(t + dt)); // 0.2 -> 0.3
}

TEST_CASE("MultiExplicitRKStepper updates fields independently (RK4)",
          "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;
  constexpr std::size_t N = 2;
  std::vector<double> u1(n, 0.0);
  std::vector<double> u2(n, 0.0);

  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
    }
  };

  std::array<std::size_t, N> local_sizes = {n, n};
  auto tableau = make_rk4_classical<double>();
  MultiExplicitRKStepper<decltype(rhs), N> stepper(dt, local_sizes, tableau, rhs);

  stepper.step(0.0, u1, u2);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u1[i] == Approx(dt * 1.0)); // 0.1
    REQUIRE(u2[i] == Approx(dt * 2.0)); // 0.2
  }

  stepper.step(dt, u1, u2);
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u1[i] == Approx(2.0 * dt * 1.0)); // 0.2
    REQUIRE(u2[i] == Approx(2.0 * dt * 2.0)); // 0.4
  }
}

TEST_CASE("ExplicitRKStepper factory with LocalField", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;

  // Mock model with rhs(double t, const G&) -> double
  struct MockModel {
    double rhs(double /*t*/, double /*g*/) const { return 3.0; }
  };

  // Mock evaluator with required interface
  struct MockEval {
    std::size_t size() const { return n; }
    void prepare() {}
    int imin() const { return 0; }
    int imax() const { return static_cast<int>(n); }
    int jmin() const { return 0; }
    int jmax() const { return 1; }
    int kmin() const { return 0; }
    int kmax() const { return 1; }
    std::size_t idx(int ix, int /*iy*/, int /*iz*/) const {
      return static_cast<std::size_t>(ix);
    }
    double operator()(int /*ix*/, int /*iy*/, int /*iz*/) const { return 0.0; }
  };

  MockModel model;
  MockEval eval;

  // Create LocalField using named constructor
  auto world = pfc::world::create(pfc::GridSize({static_cast<int>(n), 1, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  pfc::field::LocalField<double> u = pfc::field::LocalField<double>::from_subdomain(
      decomp, /*rank=*/0, /*halo_width=*/0);

  auto tableau = make_rk4_classical<double>();
  auto stepper = create(u, eval, model, dt, tableau);

  std::vector<double> u_vec(n, 0.0);
  stepper.step(0.0, u_vec);

  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u_vec[i] == Approx(dt * 3.0)); // RK4 sum(b_i) = 1.0
  }
}

TEST_CASE("MultiExplicitRKStepper factory with tuple", "[stepper][unit]") {
  constexpr double dt = 0.1;
  constexpr std::size_t n = 10;

  // Mock multi-field model returning tuple
  struct MockModel {
    std::tuple<double, double> rhs(double /*t*/, double /*g*/) const {
      return {1.0, 2.0};
    }
  };

  // Mock composite evaluator
  struct MockEval {
    std::size_t size() const { return n; }
    void prepare() {}
    int imin() const { return 0; }
    int imax() const { return static_cast<int>(n); }
    int jmin() const { return 0; }
    int jmax() const { return 1; }
    int kmin() const { return 0; }
    int kmax() const { return 1; }
    std::size_t idx(int ix, int /*iy*/, int /*iz*/) const {
      return static_cast<std::size_t>(ix);
    }
    double operator()(int /*ix*/, int /*iy*/, int /*iz*/) const { return 0.0; }
  };

  MockModel model;
  MockEval eval;

  // Create LocalFields using named constructor
  auto world = pfc::world::create(pfc::GridSize({static_cast<int>(n), 1, 1}),
                                  pfc::PhysicalOrigin({0.0, 0.0, 0.0}),
                                  pfc::GridSpacing({1.0, 1.0, 1.0}));
  auto decomp = pfc::decomposition::create(world, /*nparts=*/1);
  pfc::field::LocalField<double> u1 = pfc::field::LocalField<double>::from_subdomain(
      decomp, /*rank=*/0, /*halo_width=*/0);
  pfc::field::LocalField<double> u2 = pfc::field::LocalField<double>::from_subdomain(
      decomp, /*rank=*/0, /*halo_width=*/0);

  auto fields = std::tie(u1, u2);
  auto tableau = make_rk4_classical<double>();
  auto stepper = create(fields, eval, model, dt, tableau);

  std::vector<double> u1_vec(n, 0.0);
  std::vector<double> u2_vec(n, 0.0);
  stepper.step(0.0, u1_vec, u2_vec);

  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(u1_vec[i] == Approx(dt * 1.0)); // 0.1
    REQUIRE(u2_vec[i] == Approx(dt * 2.0)); // 0.2
  }
}

// -----------------------------------------------------------------------------
// Convergence-order tests.
//
// The tests above only check exactness for a constant RHS, which any
// consistent method of any order satisfies -- it cannot distinguish a
// correct RK4 implementation from one with a coefficient bug that happens
// to still integrate a constant exactly. These tests instead integrate a
// genuine linear ODE (exponential decay, du/dt = -lambda*u) to a fixed end
// time at two step sizes related by a factor of 2, and check that the
// final error shrinks by ~2^order when dt halves -- the standard way to
// verify an RK method's actual order of accuracy.
// -----------------------------------------------------------------------------

TEST_CASE("ExplicitRKStepper RK2 midpoint shows second-order convergence",
          "[stepper][unit][convergence]") {
  constexpr double lambda = 1.0;
  constexpr double t_final = 1.0;
  auto rhs = [](double /*t*/, std::vector<double> &u, std::vector<double> &du) {
    du[0] = -lambda * u[0];
  };

  auto run = [&](double dt) {
    std::vector<double> u = {1.0};
    auto tableau = make_rk2_midpoint<double>();
    ExplicitRKStepper stepper(dt, 1, tableau, rhs);
    double t = 0.0;
    const int steps = static_cast<int>(std::lround(t_final / dt));
    for (int i = 0; i < steps; ++i) {
      t = stepper.step(t, u);
    }
    return std::abs(u[0] - std::exp(-lambda * t));
  };

  const double dt_coarse = 0.02;
  const double err_coarse = run(dt_coarse);
  const double err_fine = run(dt_coarse / 2.0);

  REQUIRE(err_coarse > 1e-8); // sanity: coarse error must be measurable
  REQUIRE(err_fine > 0.0);
  const double ratio = err_coarse / err_fine;
  // Second order: halving dt should reduce error by ~4x. Allow generous
  // slack (3x-5x) for the asymptotic regime not being exact at this dt.
  REQUIRE(ratio > 3.0);
  REQUIRE(ratio < 5.0);
}

TEST_CASE("ExplicitRKStepper RK4 classical shows fourth-order convergence",
          "[stepper][unit][convergence]") {
  constexpr double lambda = 1.0;
  constexpr double t_final = 1.0;
  auto rhs = [](double /*t*/, std::vector<double> &u, std::vector<double> &du) {
    du[0] = -lambda * u[0];
  };

  auto run = [&](double dt) {
    std::vector<double> u = {1.0};
    auto tableau = make_rk4_classical<double>();
    ExplicitRKStepper stepper(dt, 1, tableau, rhs);
    double t = 0.0;
    const int steps = static_cast<int>(std::lround(t_final / dt));
    for (int i = 0; i < steps; ++i) {
      t = stepper.step(t, u);
    }
    return std::abs(u[0] - std::exp(-lambda * t));
  };

  const double dt_coarse = 0.1;
  const double err_coarse = run(dt_coarse);
  const double err_fine = run(dt_coarse / 2.0);

  REQUIRE(err_coarse > 1e-10); // sanity: coarse error must be measurable
  REQUIRE(err_fine > 0.0);
  const double ratio = err_coarse / err_fine;
  // Fourth order: halving dt should reduce error by ~16x. Allow generous
  // slack (10x-24x) for the asymptotic regime not being exact at this dt.
  REQUIRE(ratio > 10.0);
  REQUIRE(ratio < 24.0);
}

// -----------------------------------------------------------------------------
// Checkpoint/rollback protocol tests for EulerStepper and MultiEulerStepper
// -----------------------------------------------------------------------------

TEST_CASE("test_euler_stepper_save_state_copies_to_checkpoint", "[stepper][unit]") {
  const int n = 100;

  // Define RHS callable matching EulerStepper requirements
  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  std::vector<double> u(n, 1.0);
  EulerStepper stepper(0.1, n, rhs);  // dt=0.1, local_size=n, rhs=rhs

  stepper.save_state(u);

  // Modify u after checkpoint
  for (auto& val : u) val = 2.0;

  // Verify checkpoint still holds original values (access via friend declaration or verify through restore)
  std::vector<double> u_restored(n);
  stepper.restore_state(u_restored);

  for (int i = 0; i < n; ++i) {
    REQUIRE(u_restored[i] == 1.0);
  }
}

TEST_CASE("test_euler_stepper_restore_state_copies_from_checkpoint", "[stepper][unit]") {
  const int n = 100;

  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  std::vector<double> u(n, 1.0);
  std::vector<double> u_initial = u;
  EulerStepper stepper(0.1, n, rhs);

  stepper.save_state(u);

  // Modify u
  for (auto& val : u) val = 2.0;

  stepper.restore_state(u);

  REQUIRE(u == u_initial);
}

TEST_CASE("test_euler_stepper_can_rollback_returns_true", "[stepper][unit]") {
  const int n = 100;

  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  EulerStepper stepper(0.1, n, rhs);
  REQUIRE(stepper.can_rollback() == true);
}

TEST_CASE("test_euler_stepper_checkpoint_preserves_exact_values", "[stepper][unit]") {
  const int n = 100;

  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  std::vector<double> u(n);
  for (int i = 0; i < n; ++i) {
    u[i] = i * 0.1 + 0.5;  // Use non-trivial values
  }
  std::vector<double> u_initial = u;
  EulerStepper stepper(0.1, n, rhs);

  stepper.save_state(u);

  // Modify u
  for (auto& val : u) val = -1.0;

  stepper.restore_state(u);

  REQUIRE(u == u_initial);
}

TEST_CASE("test_euler_stepper_checkpoint_buffer_correct_sizing", "[stepper][unit]") {
  const int n = 100;

  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  EulerStepper stepper(0.1, n, rhs);
  std::vector<double> u(n, 1.0);

  // Save should work correctly with buffer sized to match m_du
  REQUIRE_NOTHROW(stepper.save_state(u));

  // Restore should work correctly
  REQUIRE_NOTHROW(stepper.restore_state(u));
}

TEST_CASE("test_euler_stepper_checkpoint_save_restore_cycle", "[stepper][unit]") {
  const int n = 100;

  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  std::vector<double> u(n, 1.0);
  std::vector<double> u_initial = u;
  EulerStepper stepper(0.1, n, rhs);

  REQUIRE(stepper.can_rollback());

  stepper.save_state(u);

  // Modify u
  for (auto& val : u) val = 2.0;

  stepper.restore_state(u);

  REQUIRE(u == u_initial);
}

TEST_CASE("test_euler_stepper_checkpoint_rollback_pattern", "[stepper][unit]") {
  const int n = 100;

  // Define RHS callable matching EulerStepper requirements
  auto rhs = [](double /*t*/, std::vector<double>& u, std::vector<double>& du) {
    std::fill(du.begin(), du.end(), 1.0);
  };

  std::vector<double> u(n, 1.0);
  std::vector<double> u_initial = u;
  EulerStepper stepper(0.1, n, rhs);  // dt=0.1, local_size=n, rhs=rhs
  pfc::Time time({0.0, 1.0, 0.1});  // t0=0.0, t1=1.0, dt=0.1

  // Save state before stepping
  stepper.save_state(u);

  // Perform a step
  stepper.step(0.0, u);

  // Simulate error check: if error > tolerance, rollback
  double error = 1.0;  // Simulated large error
  double tolerance = 0.01;

  if (stepper.can_rollback() && error > tolerance) {
    stepper.restore_state(u);
    // Time manipulation removed to avoid increment going negative
    // For checkpoint testing, verify state restoration only
  }

  // Verify state was restored
  REQUIRE(u == u_initial);
}

TEST_CASE("test_multi_euler_stepper_save_state_captures_all_fields", "[stepper][unit]") {
  const int n = 100;
  constexpr std::size_t N = 3;
  std::array<std::size_t, N> local_sizes = {n, n, n};
  double dt = 0.1;

  // Define RHS callable matching MultiEulerStepper requirements
  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    auto &du3 = std::get<2>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
      du3[i] = 3.0;
    }
  };

  std::vector<double> u1(n, 1.0);
  std::vector<double> u2(n, 2.0);
  std::vector<double> u3(n, 3.0);
  std::vector<double> u1_initial = u1;
  std::vector<double> u2_initial = u2;
  std::vector<double> u3_initial = u3;

  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);

  stepper.save_state(u1, u2, u3);

  // Modify all fields
  for (auto& val : u1) val = 10.0;
  for (auto& val : u2) val = 20.0;
  for (auto& val : u3) val = 30.0;

  // Restore to verify checkpoint captured all fields
  stepper.restore_state(u1, u2, u3);

  REQUIRE(u1 == u1_initial);
  REQUIRE(u2 == u2_initial);
  REQUIRE(u3 == u3_initial);
}

TEST_CASE("test_multi_euler_stepper_restore_state_restores_all_fields", "[stepper][unit]") {
  const int n = 100;
  constexpr std::size_t N = 3;
  std::array<std::size_t, N> local_sizes = {n, n, n};
  double dt = 0.1;

  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    auto &du3 = std::get<2>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
      du3[i] = 3.0;
    }
  };

  std::vector<double> u1(n, 1.0);
  std::vector<double> u2(n, 2.0);
  std::vector<double> u3(n, 3.0);
  std::vector<double> u1_initial = u1;
  std::vector<double> u2_initial = u2;
  std::vector<double> u3_initial = u3;

  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);

  stepper.save_state(u1, u2, u3);

  // Modify all fields
  for (auto& val : u1) val = 10.0;
  for (auto& val : u2) val = 20.0;
  for (auto& val : u3) val = 30.0;

  stepper.restore_state(u1, u2, u3);

  REQUIRE(u1 == u1_initial);
  REQUIRE(u2 == u2_initial);
  REQUIRE(u3 == u3_initial);
}

TEST_CASE("test_multi_euler_stepper_can_rollback_returns_true", "[stepper][unit]") {
  const int n = 100;
  constexpr std::size_t N = 3;
  std::array<std::size_t, N> local_sizes = {n, n, n};
  double dt = 0.1;

  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    auto &du3 = std::get<2>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
      du3[i] = 3.0;
    }
  };

  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);
  REQUIRE(stepper.can_rollback() == true);
}

TEST_CASE("test_multi_euler_stepper_checkpoint_save_restore_cycle", "[stepper][unit]") {
  const int n = 100;
  constexpr std::size_t N = 3;
  std::array<std::size_t, N> local_sizes = {n, n, n};
  double dt = 0.1;

  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    auto &du3 = std::get<2>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
      du3[i] = 3.0;
    }
  };

  std::vector<double> u1(n, 1.0);
  std::vector<double> u2(n, 2.0);
  std::vector<double> u3(n, 3.0);
  std::vector<double> u1_initial = u1;
  std::vector<double> u2_initial = u2;
  std::vector<double> u3_initial = u3;

  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);

  REQUIRE(stepper.can_rollback());

  stepper.save_state(u1, u2, u3);

  // Modify all fields
  for (auto& val : u1) val = 10.0;
  for (auto& val : u2) val = 20.0;
  for (auto& val : u3) val = 30.0;

  stepper.restore_state(u1, u2, u3);

  REQUIRE(u1 == u1_initial);
  REQUIRE(u2 == u2_initial);
  REQUIRE(u3 == u3_initial);
}

TEST_CASE("test_multi_euler_stepper_checkpoint_independent_fields", "[stepper][unit]") {
  const int n = 100;
  constexpr std::size_t N = 2;
  std::array<std::size_t, N> local_sizes = {n, n};
  double dt = 0.1;

  auto rhs = [](double /*t*/, auto & /*u_pack*/, auto &du_pack) {
    auto &du1 = std::get<0>(du_pack);
    auto &du2 = std::get<1>(du_pack);
    for (std::size_t i = 0; i < n; ++i) {
      du1[i] = 1.0;
      du2[i] = 2.0;
    }
  };

  std::vector<double> u1(n, 1.0);
  std::vector<double> u2(n, 2.0);
  std::vector<double> u1_initial = u1;
  std::vector<double> u2_initial = u2;

  MultiEulerStepper<decltype(rhs), N> stepper(dt, local_sizes, rhs);

  stepper.save_state(u1, u2);

  // Modify only u1
  for (auto& val : u1) val = 10.0;
  // u2 remains unchanged

  stepper.restore_state(u1, u2);

  REQUIRE(u1 == u1_initial);
  REQUIRE(u2 == u2_initial);
}
